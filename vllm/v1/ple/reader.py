# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import mmap
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor

from .types import PLEReaderConfig, PLEResult, PreparedRows

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch


@dataclass(eq=False)
class _Registration:
    start: int
    end: int
    view: torch.Tensor
    users: set[str] = field(default_factory=set)


class PLEReader:
    """Map stable local storage and copy descriptors without provider calls."""

    def __init__(
        self,
        config: PLEReaderConfig,
        rows: torch.Tensor,
        ready: torch.cuda.Event,
        *,
        rank: int = 0,
    ) -> None:
        self._device_rows = rows.view(torch.uint8)
        if (
            str(rows.dtype) != config.dtype
            or self._device_rows.shape[1] != config.token_bytes
        ):
            raise ValueError("PLE pool does not match the worker row layout")
        self._config = config
        self._token_bytes = config.token_bytes
        self._ready = ready
        self._consumer_done = torch.cuda.Event()
        self._consumer_done.record()
        self._copy_stream = torch.cuda.Stream(device=rows.device)
        self._results: dict[str, PreparedRows] = {}
        self._last_copy: dict[str, torch.cuda.Event] = {}
        self._registrations: list[_Registration] = []
        pool = config.pool
        fd = (
            os.open(pool.path, os.O_RDWR)
            if pool.path is not None
            else pool.handles[rank].detach()
        )
        try:
            self._memory = mmap.mmap(fd, pool.size)
        finally:
            os.close(fd)
        self._host_rows = torch.frombuffer(self._memory, dtype=torch.uint8)
        try:
            if pool.registration == "pool":
                pin_tensor(self._host_rows)
        except BaseException:
            del self._host_rows
            self._memory.close()
            raise

    def _register(self, rows: PreparedRows) -> None:
        if (previous := self._results.get(rows.result_id)) is not None:
            if previous != rows:
                raise ValueError("PLE result ID was reused with different storage")
            return
        pool = self._config.pool
        if rows.pool_id != pool.pool_id or any(
            c.offset < 0 or c.nbytes <= 0 or c.offset + c.nbytes > pool.size
            for c in rows.chunks
        ):
            raise ValueError("Invalid PLE pool generation or byte range")
        if pool.registration == "region":
            for chunk in rows.chunks:
                start = chunk.offset // mmap.PAGESIZE * mmap.PAGESIZE
                end = min(
                    pool.size,
                    (chunk.offset + chunk.nbytes + mmap.PAGESIZE - 1)
                    // mmap.PAGESIZE
                    * mmap.PAGESIZE,
                )
                # CUDA registration is page-based. Share registrations when
                # independently owned results occupy the same mapped page.
                for registration in sorted(self._registrations, key=lambda r: r.start):
                    if registration.end <= start or registration.start >= end:
                        continue
                    if start < registration.start:
                        self._pin(start, registration.start, rows.result_id)
                    registration.users.add(rows.result_id)
                    start = min(end, registration.end)
                if start < end:
                    self._pin(start, end, rows.result_id)
        self._results[rows.result_id] = rows

    def _pin(self, start: int, end: int, result_id: str) -> None:
        view = self._host_rows[start:end]
        pin_tensor(view)
        self._registrations.append(_Registration(start, end, view, {result_id}))

    @staticmethod
    def _unpin(view: torch.Tensor) -> None:
        error = torch.cuda.cudart().cudaHostUnregister(view.data_ptr())
        if error.value != 0:
            raise RuntimeError(f"cudaHostUnregister failed: {error}")

    def stage(self, batch: "InputBatch", results: dict[str, PLEResult]) -> None:
        copies = []
        for i, req_id in enumerate(batch.req_ids):
            result = results[req_id]
            start = int(batch.num_computed_prefill_tokens_np[i])
            length = int(batch.num_scheduled_tokens[i])
            if not result.start <= start <= start + length <= result.end:
                raise RuntimeError(
                    f"Scheduled tokens are outside the PLE result for {req_id}"
                )
            if (
                sum(c.nbytes for c in result.rows.chunks)
                != (result.end - result.start) * self._token_bytes
            ):
                raise ValueError("PLE byte length does not match the token range")
            self._register(result.rows)
            begin = (start - result.start) * self._token_bytes
            end = begin + length * self._token_bytes
            destination = int(batch.query_start_loc_np[i]) * self._token_bytes
            position = 0
            for chunk in result.rows.chunks:
                lo, hi = max(begin, position), min(end, position + chunk.nbytes)
                if lo < hi:
                    copies.append(
                        (
                            result.rows.result_id,
                            chunk.offset + lo - position,
                            destination + lo - begin,
                            hi - lo,
                        )
                    )
                position += chunk.nbytes

        self._copy_stream.wait_event(self._consumer_done)
        copy_done = torch.cuda.Event()
        with torch.cuda.stream(self._copy_stream):
            device = self._device_rows.flatten()
            for result_id, src, dst, size in copies:
                device[dst : dst + size].copy_(
                    self._host_rows[src : src + size], non_blocking=True
                )
                self._last_copy[result_id] = copy_done
            self._device_rows[batch.num_tokens : batch.num_tokens_after_padding].zero_()
            copy_done.record()
            self._ready.record()

    def finish_forward(self) -> None:
        self._consumer_done.record()

    def release(self, result_ids: set[str]) -> set[str]:
        ready = set()
        for result_id in result_ids:
            event = self._last_copy.get(result_id)
            if event is not None and not event.query():
                continue
            for registration in list(self._registrations):
                registration.users.discard(result_id)
                if not registration.users:
                    self._unpin(registration.view)
                    self._registrations.remove(registration)
            self._last_copy.pop(result_id, None)
            self._results.pop(result_id, None)
            ready.add(result_id)
        return ready

    def close(self) -> None:
        self._copy_stream.synchronize()
        while self._registrations:
            self._unpin(self._registrations[-1].view)
            self._registrations.pop()
        if self._config.pool.registration == "pool":
            self._unpin(self._host_rows)
        del self._host_rows
        self._memory.close()
