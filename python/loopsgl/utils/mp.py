# Portions adapted from SGLang's srt/utils/common.py.
# Copyright 2023-2024 SGLang Team.
# Licensed under the Apache License, Version 2.0.
# See LICENSES/sglang.txt in the repository root.

from __future__ import annotations

import pickle
from typing import Callable, Dict, Generic, List, TypeVar

import msgpack
import numpy as np
import torch
import torch.distributed as dist
import zmq
import zmq.asyncio

T = TypeVar("T")


def broadcast_pyobj(
    data: List[T] | None,
    rank: int,
    dist_group: torch.distributed.ProcessGroup | None = None,
    src: int = 0,
) -> List[T]:
    """Broadcast a Python list through a CPU process group.

    rank and src are global ranks, including when dist_group is a subgroup.
    Empty lists require only a zero-size broadcast.
    """
    if rank == src:
        assert data is not None
        if len(data) == 0:
            tensor_size = torch.tensor([0], dtype=torch.long, device="cpu")
            dist.broadcast(tensor_size, src=src, group=dist_group)
        else:
            serialized_data = pickle.dumps(data)
            tensor_data = torch.ByteTensor(np.frombuffer(serialized_data, dtype=np.uint8).copy())
            tensor_size = torch.tensor([len(serialized_data)], dtype=torch.long, device="cpu")
            dist.broadcast(tensor_size, src=src, group=dist_group)
            dist.broadcast(tensor_data, src=src, group=dist_group)
        return data

    tensor_size = torch.tensor([0], dtype=torch.long, device="cpu")
    dist.broadcast(tensor_size, src=src, group=dist_group)
    size = int(tensor_size.item())
    if size == 0:
        return []

    tensor_data = torch.empty(size, dtype=torch.uint8, device="cpu")
    dist.broadcast(tensor_data, src=src, group=dist_group)
    return pickle.loads(bytes(tensor_data.numpy()))


class ZmqPushQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put(self, obj: T):
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close(linger=0)
        self.context.term()


class ZmqAsyncPushQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    async def put(self, obj: T):
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        await self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close(linger=0)
        self.context.term()


class ZmqPullQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    def get(self) -> T:
        event = self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def get_raw(self) -> bytes:
        return self.socket.recv()

    def decode(self, raw: bytes) -> T:
        return self.decoder(msgpack.unpackb(raw, raw=False))

    def empty(self) -> bool:
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        self.socket.close(linger=0)
        self.context.term()


class ZmqAsyncPullQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    async def get(self) -> T:
        event = await self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def stop(self):
        self.socket.close(linger=0)
        self.context.term()


class ZmqPubQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put_raw(self, raw: bytes):
        self.socket.send(raw, copy=False)

    def put(self, obj: T):
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close(linger=0)
        self.context.term()


class ZmqSubQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.decoder = decoder

    def get(self) -> T:
        event = self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def empty(self) -> bool:
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        self.socket.close(linger=0)
        self.context.term()
