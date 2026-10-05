// SPDX-License-Identifier: Apache-2.0
// Adapted from SGLang's elementwise/kvcache.cuh; see LICENSES/sglang.txt in the
// repository root.
// https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/kernels/jit/csrc/elementwise/kvcache.cuh
// Uses Loop-SGLang's JIT utilities and warp-copy helpers.

#include <loopsgl/tensor.h>
#include <loopsgl/utils.cuh>
#include <loopsgl/utils.h>
#include <loopsgl/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>

namespace {

struct StoreKernelParams {
  void *__restrict__ k_cache;
  void *__restrict__ v_cache;
  const void *__restrict__ indices;
  const void *__restrict__ k;
  const void *__restrict__ v;
  std::size_t k_cache_stride;
  std::size_t v_cache_stride;
  std::size_t k_input_stride;
  std::size_t v_input_stride;
  std::size_t indices_stride;
  std::size_t length;
  int64_t size_limit;
  int64_t reserved_skip_index;
};

template <std::size_t kBytes>
__always_inline __device__ void
copy_kv_warp(void *__restrict__ k_dst, void *__restrict__ v_dst,
             const void *__restrict__ k_src, const void *__restrict__ v_src) {
  using namespace device;
  using Package =
      warp::details::mem_package_t<kBytes,
                                   warp::details::resolve_unit_size(kBytes)>;
  constexpr auto kPackages = kBytes / sizeof(Package);
  constexpr auto kLoops = kPackages / kWarpThreads;
  const auto lane = threadIdx.x % kWarpThreads;
  auto dst_k = static_cast<Package *>(k_dst);
  auto dst_v = static_cast<Package *>(v_dst);
  auto src_k = static_cast<const Package *>(k_src);
  auto src_v = static_cast<const Package *>(v_src);

  // Issue both loads before either store to interleave independent K/V copies.
#pragma unroll kLoops
  for (std::size_t i = 0; i < kLoops; ++i) {
    const auto j = i * kWarpThreads + lane;
    const auto k = src_k[j];
    const auto v = src_v[j];
    dst_k[j] = k;
    dst_v[j] = v;
  }
  if constexpr (kPackages % kWarpThreads != 0) {
    const auto j = kLoops * kWarpThreads + lane;
    if (j < kPackages) {
      const auto k = src_k[j];
      const auto v = src_v[j];
      dst_k[j] = k;
      dst_v[j] = v;
    }
  }
}

template <std::size_t kKBytes, std::size_t kVBytes>
__always_inline __device__ void
copy_kv_rows(void *__restrict__ k_dst, void *__restrict__ v_dst,
             const void *__restrict__ k_src, const void *__restrict__ v_src) {
  using namespace device;
  constexpr auto kCommon = kKBytes < kVBytes ? kKBytes : kVBytes;
  constexpr auto kTail = (kKBytes < kVBytes ? kVBytes : kKBytes) - kCommon;
  constexpr auto kTailOrCommon = kTail == 0 ? kCommon : kTail;
  constexpr auto kCommonAlign = warp::details::resolve_unit_size(kCommon);
  constexpr auto kTailAlign = warp::details::resolve_unit_size(kTailOrCommon);
  // Both split offsets must support the common vector width, and the tail
  // starts at kCommon with its own vector alignment.
  constexpr bool kCanInterleave = kKBytes % kCommonAlign == 0 &&
                                  kVBytes % kCommonAlign == 0 &&
                                  kCommon % kTailAlign == 0;

  if constexpr (kCanInterleave) {
    if constexpr (kTail > 0) {
      // Independent strides need not be multiples of the row width.
      const auto common_pointers = reinterpret_cast<std::uintptr_t>(k_src) |
                                   reinterpret_cast<std::uintptr_t>(v_src) |
                                   reinterpret_cast<std::uintptr_t>(k_dst) |
                                   reinterpret_cast<std::uintptr_t>(v_dst);
      const auto tail_pointers =
          kKBytes > kVBytes ? reinterpret_cast<std::uintptr_t>(k_src) |
                                  reinterpret_cast<std::uintptr_t>(k_dst)
                            : reinterpret_cast<std::uintptr_t>(v_src) |
                                  reinterpret_cast<std::uintptr_t>(v_dst);
      if (common_pointers % kCommonAlign != 0 ||
          tail_pointers % kTailAlign != 0) {
        warp::copy<kKBytes>(k_dst, k_src);
        warp::copy<kVBytes>(v_dst, v_src);
        return;
      }
    }
    copy_kv_warp<kCommon>(k_dst, v_dst, k_src, v_src);
    if constexpr (kTail > 0) {
      if constexpr (kKBytes > kVBytes) {
        warp::copy<kTail>(pointer::offset(k_dst, kCommon),
                          pointer::offset(k_src, kCommon));
      } else {
        warp::copy<kTail>(pointer::offset(v_dst, kCommon),
                          pointer::offset(v_src, kCommon));
      }
    }
  } else {
    warp::copy<kKBytes>(k_dst, k_src);
    warp::copy<kVBytes>(v_dst, v_src);
  }
}

template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
          std::size_t kKBytes, std::size_t kVBytes, unsigned kSplit,
          std::integral T>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void store_kv_cache(
    const __grid_constant__ StoreKernelParams params) {
  using namespace device;
  constexpr auto kWarpsPerBlock = kNumThreads / kWarpThreads;
  static_assert(kNumThreads % kWarpThreads == 0);
  const auto warp_id = threadIdx.x / kWarpThreads + blockIdx.x * kWarpsPerBlock;
  const auto item_id = warp_id / kSplit;
  const auto split_id = warp_id % kSplit;
  PDL::wait<kUsePDL>();

  if (item_id < params.length) {
    const auto pos =
        static_cast<const T *>(params.indices)[item_id * params.indices_stride];
    assert(pos >= 0 && pos < params.size_limit);
    if (pos != params.reserved_skip_index) {
      constexpr auto kKSplitBytes = kKBytes / kSplit;
      constexpr auto kVSplitBytes = kVBytes / kSplit;
      const auto dst_k = pointer::offset(
          params.k_cache, pos * params.k_cache_stride, split_id * kKSplitBytes);
      const auto dst_v = pointer::offset(
          params.v_cache, pos * params.v_cache_stride, split_id * kVSplitBytes);
      const auto src_k = pointer::offset(
          params.k, item_id * params.k_input_stride, split_id * kKSplitBytes);
      const auto src_v = pointer::offset(
          params.v, item_id * params.v_input_stride, split_id * kVSplitBytes);
      copy_kv_rows<kKSplitBytes, kVSplitBytes>(dst_k, dst_v, src_k, src_v);
    }
  }
  PDL::launch<kUsePDL>();
}

template <std::size_t k_row_bytes, std::size_t v_row_bytes,
          std::size_t num_threads = 128, std::size_t max_concurrency = 1,
          bool use_pdl = false>
struct StoreKernel {
  static_assert(k_row_bytes > 0 && k_row_bytes % 4 == 0);
  static_assert(v_row_bytes > 0 && v_row_bytes % 4 == 0);

  template <unsigned split, typename T>
  static constexpr auto kernel =
      store_kv_cache<num_threads, max_concurrency, use_pdl, k_row_bytes,
                     v_row_bytes, split, T>;

  template <typename T> static auto get_kernel(int num_split) {
    using namespace host;
    if constexpr (k_row_bytes % 512 == 0 && v_row_bytes % 512 == 0) {
      if (num_split == 4)
        return kernel<4, T>;
    }
    if constexpr (k_row_bytes % 256 == 0 && v_row_bytes % 256 == 0) {
      if (num_split == 2)
        return kernel<2, T>;
    }
    RuntimeCheck(num_split == 1, "Unsupported KV row split: ", num_split);
    return kernel<1, T>;
  }

  static void run(const tvm::ffi::TensorView k_cache,
                  const tvm::ffi::TensorView v_cache,
                  const tvm::ffi::TensorView indices,
                  const tvm::ffi::TensorView k, const tvm::ffi::TensorView v,
                  int num_split, int64_t size_limit,
                  int64_t reserved_skip_index) {
    using namespace host;
    auto DK = SymbolicSize{"key_width"};
    auto DV = SymbolicSize{"value_width"};
    auto L = SymbolicSize{"batch_size"};
    auto N = SymbolicSize{"cache_rows"};
    auto SK = SymbolicSize{"key_cache_stride"};
    auto SV = SymbolicSize{"value_cache_stride"};
    auto KS = SymbolicSize{"key_input_stride"};
    auto VS = SymbolicSize{"value_input_stride"};
    auto IS = SymbolicSize{"indices_stride"};
    auto indices_dtype = SymbolicDType{};
    auto dtype = SymbolicDType{};
    auto device = SymbolicDevice{};

    TensorMatcher({N, DK})
        .with_strides({SK, 1})
        .with_device<kDLCUDA>(device)
        .with_dtype(dtype)
        .verify(k_cache);
    TensorMatcher({N, DV})
        .with_strides({SV, 1})
        .with_device<kDLCUDA>(device)
        .with_dtype(dtype)
        .verify(v_cache);
    TensorMatcher({L, DK})
        .with_strides({KS, 1})
        .with_device<kDLCUDA>(device)
        .with_dtype(dtype)
        .verify(k);
    TensorMatcher({L, DV})
        .with_strides({VS, 1})
        .with_device<kDLCUDA>(device)
        .with_dtype(dtype)
        .verify(v);
    TensorMatcher({L})
        .with_strides({IS})
        .with_device<kDLCUDA>(device)
        .with_dtype<int32_t, int64_t>(indices_dtype)
        .verify(indices);

    const auto dtype_size = dtype_bytes(dtype.unwrap());
    RuntimeCheck(k_row_bytes == dtype_size * DK.unwrap());
    RuntimeCheck(v_row_bytes == dtype_size * DV.unwrap());
    RuntimeCheck(size_limit > 0 && size_limit <= N.unwrap());

    const auto params = StoreKernelParams{
        .k_cache = k_cache.data_ptr(),
        .v_cache = v_cache.data_ptr(),
        .indices = indices.data_ptr(),
        .k = k.data_ptr(),
        .v = v.data_ptr(),
        .k_cache_stride = static_cast<std::size_t>(SK.unwrap() * dtype_size),
        .v_cache_stride = static_cast<std::size_t>(SV.unwrap() * dtype_size),
        .k_input_stride = static_cast<std::size_t>(KS.unwrap() * dtype_size),
        .v_input_stride = static_cast<std::size_t>(VS.unwrap() * dtype_size),
        .indices_stride = static_cast<std::size_t>(IS.unwrap()),
        .length = static_cast<std::size_t>(L.unwrap()),
        .size_limit = size_limit,
        .reserved_skip_index = reserved_skip_index,
    };
    const auto launch = indices_dtype.unwrap().bits == 32
                            ? get_kernel<int32_t>(num_split)
                            : get_kernel<int64_t>(num_split);
    constexpr auto kWarpsPerBlock = num_threads / 32;
    const auto num_blocks = div_ceil(params.length * num_split, kWarpsPerBlock);
    LaunchKernel(num_blocks, num_threads, device.unwrap())
        .with_attr(use_pdl)(launch, params);
  }
};

} // namespace
