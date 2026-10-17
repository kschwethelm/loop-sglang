import torch

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo


class NoPrefixCacheHandle(BaseCacheHandle):
    empty_tensor: torch.Tensor  # should be set by NoPrefixCache

    def __init__(self) -> None:
        super().__init__(cached_len=0)

    def get_matched_indices(self) -> torch.Tensor:
        return self.empty_tensor


class NoPrefixCache(BasePrefixCache):
    """Implement the prefix-cache interface without retaining or matching prefixes."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        NoPrefixCacheHandle.empty_tensor = self.empty_tensor
        super().__init__()

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        pass

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        return MatchResult(NoPrefixCacheHandle())

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        return InsertResult(0, NoPrefixCacheHandle())

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        raise NotImplementedError("NoPrefixCache has no cached prefixes to evict.")

    def reset(self) -> None:
        pass

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=0, protected_size=0)

    def check_integrity(self) -> None:
        pass
