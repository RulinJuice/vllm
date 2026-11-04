# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import mmap
from typing import TYPE_CHECKING

import torch

from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor

from .types import PLEPool, PLEResult

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch


class PLEReader:
    """Stage this rank's packed token rows from the EngineCore result pool."""

    def __init__(
        self, pool: PLEPool, rows: torch.Tensor, ready: torch.cuda.Event
    ) -> None:
        self._device_rows = rows.view(torch.uint8)
        if (
            str(rows.dtype) != pool.dtype
            or self._device_rows.shape[1] != pool.token_bytes
        ):
            raise ValueError("PLE result pool does not match the worker row layout")
        self._token_bytes = pool.token_bytes
        self._ready = ready
        self._consumer_done = torch.cuda.Event()
        self._consumer_done.record()
        self._copy_stream = torch.cuda.Stream(device=rows.device)
        self._results: dict[str, PLEResult] = {}
        self._last_copy: dict[int, torch.cuda.Event] = {}
        with open(pool.path, "r+b") as pool_file:
            self._memory = mmap.mmap(pool_file.fileno(), pool.size)
        self._host_rows = torch.frombuffer(self._memory, dtype=torch.uint8)
        try:
            pin_tensor(self._host_rows)
        except BaseException:
            del self._host_rows
            self._memory.close()
            raise

    def attach(self, req_id: str, result: PLEResult) -> None:
        self._results[req_id] = result

    def remove_request(self, req_id: str) -> None:
        self._results.pop(req_id, None)

    def stage(self, batch: "InputBatch") -> None:
        copies = []
        for i, req_id in enumerate(batch.req_ids):
            result = self._results[req_id]
            start = int(batch.num_computed_prefill_tokens_np[i])
            length = int(batch.num_scheduled_tokens[i])
            if not result.start <= start <= start + length <= result.end:
                raise RuntimeError(
                    f"Scheduled tokens are outside the PLE result for {req_id}"
                )
            src = result.offset + (start - result.start) * self._token_bytes
            dst = int(batch.query_start_loc_np[i])
            copies.append((result.result_id, src, dst, length))

        self._copy_stream.wait_event(self._consumer_done)
        copy_done = torch.cuda.Event()
        with torch.cuda.stream(self._copy_stream):
            for result_id, src, dst, length in copies:
                host = self._host_rows[src : src + length * self._token_bytes]
                device = self._device_rows[dst : dst + length]
                device.copy_(host.view_as(device), non_blocking=True)
                self._last_copy[result_id] = copy_done
            self._device_rows[batch.num_tokens : batch.num_tokens_after_padding].zero_()
            copy_done.record()
            self._ready.record()

    def finish_forward(self) -> None:
        # The next step may reuse the static device rows only after this one.
        self._consumer_done.record()

    def acknowledge(self, result_ids: list[int]) -> list[int]:
        ready = []
        for result_id in result_ids:
            event = self._last_copy.get(result_id)
            if event is None or event.query():
                self._last_copy.pop(result_id, None)
                ready.append(result_id)
        return ready

    def close(self) -> None:
        self._copy_stream.synchronize()
        error = torch.cuda.cudart().cudaHostUnregister(self._host_rows.data_ptr())
        if error.value != 0:
            raise RuntimeError(f"cudaHostUnregister failed: {error}")
        del self._host_rows
        self._memory.close()
