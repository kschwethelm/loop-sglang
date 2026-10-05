from __future__ import annotations

import os

import pytest
import torch
from loopsgl.kernel import test_tensor as check_tensor


@pytest.mark.skipif(
    not os.environ.get("SLURM_JOB_ID") or torch.cuda.device_count() < 2,
    reason="Tensor contract check requires two CUDA devices in a SLURM allocation",
)
def test_tensor_contract() -> None:
    x = torch.empty((12, 2048), dtype=torch.int32, device="cpu")[:, :1024]
    y = torch.empty((12, 1024), dtype=torch.int64, device="cuda:1")
    check_tensor(x, y)
