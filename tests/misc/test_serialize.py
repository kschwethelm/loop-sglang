from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from loopsgl.core import SamplingParams
from loopsgl.message import BatchBackendMsg, UserMsg
from loopsgl.message.utils import deserialize_type, serialize_type


@dataclass
class A:
    x: int
    y: str
    z: List[A]
    w: torch.Tensor


def test_serialize_deserialize() -> None:
    t = torch.tensor([1, 2, 3], dtype=torch.int32)
    x = A(10, "hello", [A(20, "world", [], t)], t)
    data = serialize_type(x)
    y = deserialize_type({"A": A}, data)
    assert isinstance(y, A)
    assert (y.x, y.y) == (10, "hello")
    torch.testing.assert_close(y.w, t)
    assert len(y.z) == 1
    assert isinstance(y.z[0], A)
    assert (y.z[0].x, y.z[0].y, y.z[0].z) == (20, "world", [])
    torch.testing.assert_close(y.z[0].w, t)

    params = SamplingParams(temperature=0.7, max_tokens=17, ignore_eos=True)
    u = BatchBackendMsg([UserMsg(uid=42, input_ids=t, sampling_params=params)])
    result = u.decoder(u.encoder())
    assert isinstance(result, BatchBackendMsg)
    assert len(result.data) == 1
    message = result.data[0]
    assert isinstance(message, UserMsg)
    assert message.uid == 42
    torch.testing.assert_close(message.input_ids, t)
    assert message.sampling_params == params
