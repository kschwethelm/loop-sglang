from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit(do_not_specialize=["size"])
def _ouro_exit_gate_kernel(
    logits,
    previous,
    steps,
    signal,
    next_state,
    size,
    logits_stride: tl.constexpr,
    previous_stride: tl.constexpr,
    steps_stride: tl.constexpr,
    threshold: tl.constexpr,
    min_steps: tl.constexpr,
    depth_offset: tl.constexpr,
    tensor_steps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    value = tl.load(logits + offsets * logits_stride, mask, 0).to(tl.float32)
    old_state = tl.load(previous + offsets * previous_stride, mask, 0).to(tl.float32)
    # Preserve subnormal hazards that libdevice division flushes under Triton's FTZ mode.
    probability = tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[1.0, 1.0 + libdevice.exp(-value)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    state = old_state + (1.0 - old_state) * probability
    if tensor_steps:
        step = tl.load(steps + offsets * steps_stride, mask, 0)
    else:
        step = steps
    eligible = step + depth_offset >= min_steps
    tl.store(next_state + offsets, state, mask)
    tl.store(signal + offsets, (state >= threshold) & eligible, mask)


def ouro_exit_gate(
    logits: torch.Tensor,
    previous_probability: torch.Tensor,
    loop_step: int | torch.Tensor,
    *,
    threshold: float,
    min_loop_steps: int,
    lookahead: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate FP32 exit hazards and apply threshold and depth eligibility on CUDA."""
    assert logits.is_cuda and previous_probability.device == logits.device
    assert logits.ndim == 1 and previous_probability.shape == logits.shape
    tensor_steps = isinstance(loop_step, torch.Tensor)
    step_stride = 0
    if tensor_steps:
        assert loop_step.device == logits.device
        assert loop_step.numel() == 1 or loop_step.shape == logits.shape
        step_stride = loop_step.stride(0) if loop_step.numel() != 1 else 0
    state = torch.empty(logits.shape, device=logits.device, dtype=torch.float32)
    signal = torch.empty(logits.shape, device=logits.device, dtype=torch.bool)
    if logits.numel():
        _ouro_exit_gate_kernel[(triton.cdiv(logits.numel(), 256),)](
            logits,
            previous_probability,
            loop_step,
            signal,
            state,
            logits.numel(),
            logits.stride(0),
            previous_probability.stride(0),
            step_stride,
            threshold,
            min_loop_steps,
            1 + int(lookahead),
            tensor_steps,
            256,
            enable_fp_fusion=False,
        )
    return signal, state
