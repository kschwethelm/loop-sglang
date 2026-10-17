from __future__ import annotations

from typing import TYPE_CHECKING, Final, List

import msgpack
import torch

from loopsgl.managers.io_struct import (
    BaseReq,
    BatchTokenIDOutput,
    SchedulerControlOutput,
    TokenIDOutput,
)
from loopsgl.utils import ZmqPullQueue, ZmqPushQueue, broadcast_pyobj, init_logger

if TYPE_CHECKING:
    from loopsgl.managers.scheduler_components.config import SchedulerConfig

logger = init_logger(__name__)


class SchedulerIOMixin:
    """
    Mixin class for Scheduler I/O operations.

    This class handles communication with the frontend tokenizer manager and detokenizer.

    Public Utilities:
        receive_msg: Function to receive messages from the tokenizer.
        send_result: Function to send generated tokens to the detokenizer.
        sync_all_ranks: Function to synchronize all ranks on CPU side.
    """

    def __init__(
        self,
        config: SchedulerConfig,
        tp_cpu_group: torch.distributed.ProcessGroup,
    ) -> None:
        tp_info = config.tp_info
        self.tp_cpu_group: Final = tp_cpu_group
        if tp_info.is_primary():
            self.send_to_tokenizer: Final = ZmqPushQueue(
                config.zmq_frontend_addr, create=False, encoder=BaseReq.encoder
            )
            self.recv_from_tokenizer: Final = ZmqPullQueue(
                config.zmq_backend_addr,
                create=True,
                decoder=BaseReq.decoder,
            )
            self.send_to_detokenizer: Final = ZmqPushQueue(
                config.zmq_detokenizer_addr,
                create=True,
                encoder=BaseReq.encoder,
            )

        recv = self._recv_msg_single_rank
        send = self._reply_detokenizer_rank0
        if tp_info.size > 1:
            recv = self._recv_msg_multi_rank
            if not tp_info.is_primary():
                send = self._reply_detokenizer_rank1

        self.receive_msg = recv
        self.send_result = send

    def sync_all_ranks(self) -> None:
        self.tp_cpu_group.barrier().wait()

    def _recv_msg_single_rank(self, blocking: bool = False) -> List[BaseReq]:
        pending_msgs: List[BaseReq] = []
        if blocking:
            pending_msgs.append(self.recv_from_tokenizer.get())
        while not self.recv_from_tokenizer.empty():
            pending_msgs.append(self.recv_from_tokenizer.get())
        return pending_msgs

    def _recv_msg_multi_rank(self, blocking: bool = False) -> List[BaseReq]:
        """Broadcast one complete raw-message list through the CPU group."""
        raw_messages: List[bytes] = []
        if self.config.tp_info.is_primary():
            if blocking:
                raw_messages.append(self.recv_from_tokenizer.get_raw())
            while not self.recv_from_tokenizer.empty():
                raw_messages.append(self.recv_from_tokenizer.get_raw())
        raw_messages = broadcast_pyobj(raw_messages, self.config.tp_info.rank, self.tp_cpu_group)
        return [BaseReq.decoder(msgpack.unpackb(raw, raw=False)) for raw in raw_messages]

    def _reply_detokenizer_rank0(self, reply: List[TokenIDOutput]) -> None:
        num_reply = len(reply)
        logger.debug_rank0(f"Sending outputs to detokenizer: {num_reply} messages")
        if num_reply == 1:
            self.send_to_detokenizer.put(reply[0])
        elif num_reply > 1:
            self.send_to_detokenizer.put(BatchTokenIDOutput(data=reply))  # type: ignore

    def _reply_detokenizer_rank1(self, reply: List[TokenIDOutput]) -> None:
        _ = reply  # do nothing for non-primary ranks

    def send_control_result(self, reply: SchedulerControlOutput) -> None:
        """Return scheduler control replies from the primary rank to the frontend."""
        if self.config.tp_info.is_primary():
            self.send_to_tokenizer.put(reply)
