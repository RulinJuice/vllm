# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import mmap
import os
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from uuid import uuid4

import torch

from vllm.v1.metrics.ple import PLEFetchObservation

from .types import PLEChunk, PoolDescriptor, PreparedRows


@dataclass
class _Preparation:
    ready: Future[PreparedRows] = field(default_factory=Future)
    task: Future[None] = field(init=False)
    chunks: tuple[PLEChunk, ...] = ()
    released: Future[None] | None = None


class LocalPLEProvider:
    """Gather an immutable CPU byte table into a provider-owned shared pool."""

    def __init__(
        self, table: torch.Tensor, *, pool_bytes: int, directory: str = "/dev/shm"
    ) -> None:
        if (
            table.device.type != "cpu"
            or table.dtype != torch.uint8
            or table.ndim != 2
            or not table.is_contiguous()
            or not table.numel()
        ):
            raise ValueError("PLE table must be a nonempty contiguous CPU byte matrix")
        self._table = table
        self.row_bytes = table.shape[1]
        self._capacity = pool_bytes // self.row_bytes * self.row_bytes
        if self._capacity <= 0:
            raise ValueError("PLE pool must hold at least one row")
        self._file = tempfile.NamedTemporaryFile(  # noqa: SIM115
            prefix="vllm-ple-", dir=directory
        )
        try:
            if hasattr(os, "posix_fallocate"):
                os.posix_fallocate(self._file.fileno(), 0, pool_bytes)
            else:
                self._file.truncate(pool_bytes)
            self._memory = mmap.mmap(self._file.fileno(), pool_bytes)
        except BaseException:
            self._file.close()
            raise
        self._pool = PoolDescriptor(uuid4().hex, pool_bytes, path=self._file.name)
        self._available = [(0, self._capacity)]
        self._condition = threading.Condition()
        self._operations: dict[str, _Preparation] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ple-local"
        )
        self._closed = False

    def export_pool(self, num_workers: int) -> PoolDescriptor:
        if num_workers < 1 or self._closed:
            raise ValueError("Cannot export PLE pool")
        return self._pool

    def prepare(
        self,
        result_id: str,
        row_ids: torch.Tensor,
        *,
        observation: PLEFetchObservation | None = None,
    ) -> Future[PreparedRows]:
        with self._condition:
            if self._closed or result_id in self._operations:
                raise ValueError(
                    "PLE provider is closed or result ID is already in use"
                )
            operation = _Preparation()
            self._operations[result_id] = operation
            operation.task = self._executor.submit(
                self._gather, result_id, row_ids, operation, observation
            )
            return operation.ready

    def _allocate(self, nbytes: int, operation: _Preparation) -> tuple[PLEChunk, ...]:
        with self._condition:
            self._condition.wait_for(
                lambda: self._closed
                or operation.released is not None
                or sum(end - start for start, end in self._available) >= nbytes
            )
            if self._closed:
                raise RuntimeError("PLE provider closed during preparation")
            if operation.released is not None:
                raise RuntimeError("PLE preparation withdrawn")
            chunks = []
            while nbytes:
                start, end = self._available[0]
                length = min(nbytes, end - start)
                chunks.append(PLEChunk(start, length))
                if start + length == end:
                    self._available.pop(0)
                else:
                    self._available[0] = (start + length, end)
                nbytes -= length
            return tuple(chunks)

    def _gather(
        self,
        result_id: str,
        row_ids: torch.Tensor,
        operation: _Preparation,
        observation: PLEFetchObservation | None,
    ) -> None:
        error = None
        try:
            with observation.measure("prepare") if observation else nullcontext():
                if (
                    row_ids.device.type != "cpu"
                    or row_ids.dtype != torch.int64
                    or row_ids.ndim != 1
                    or not row_ids.is_contiguous()
                    or not row_ids.numel()
                ):
                    raise ValueError(
                        "PLE row IDs must be a nonempty contiguous CPU int64 vector"
                    )
                lo, hi = torch.aminmax(row_ids)
                if lo.item() < 0 or hi.item() >= self._table.shape[0]:
                    raise IndexError("PLE row ID is outside the table")
                size = row_ids.numel() * self.row_bytes
                if size > self._capacity:
                    raise ValueError("PLE rows exceed the local provider pool capacity")
                operation.chunks = self._allocate(size, operation)
            with observation.measure("fetch") if observation else nullcontext():
                position = 0
                for chunk in operation.chunks:
                    count = chunk.nbytes // self.row_bytes
                    out = torch.frombuffer(
                        self._memory,
                        dtype=torch.uint8,
                        offset=chunk.offset,
                        count=chunk.nbytes,
                    ).view(count, self.row_bytes)
                    try:
                        torch.index_select(
                            self._table,
                            0,
                            row_ids[position : position + count],
                            out=out,
                        )
                    finally:
                        del out
                    position += count
        except Exception as exc:
            # A traceback could retain a view of the pool after release.
            error = RuntimeError(str(exc) or type(exc).__name__)
        if observation is not None:
            observation.finish(error)
        if error is not None:
            operation.ready.set_exception(error)
        else:
            operation.ready.set_result(
                PreparedRows(result_id, self._pool.pool_id, operation.chunks)
            )

    def release(self, result_id: str) -> Future[None]:
        with self._condition:
            operation = self._operations.get(result_id)
            if operation is None:
                done: Future[None] = Future()
                done.set_result(None)
                return done
            if operation.released is None:
                operation.released = Future()
                operation.ready.add_done_callback(
                    lambda _: self._free(result_id, operation)
                )
                if operation.task.cancel():
                    operation.ready.set_exception(
                        RuntimeError("PLE preparation withdrawn")
                    )
                else:
                    self._condition.notify_all()
            return operation.released

    def _free(self, result_id: str, operation: _Preparation) -> None:
        with self._condition:
            ranges = sorted(
                [
                    *self._available,
                    *(
                        (chunk.offset, chunk.offset + chunk.nbytes)
                        for chunk in operation.chunks
                    ),
                ]
            )
            self._available = []
            for start, end in ranges:
                if self._available and self._available[-1][1] == start:
                    self._available[-1] = (self._available[-1][0], end)
                else:
                    self._available.append((start, end))
            del self._operations[result_id]
            self._condition.notify_all()
        assert operation.released is not None
        operation.released.set_result(None)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            releases = [self.release(key) for key in list(self._operations)]
        for release in releases:
            release.result()
        self._executor.shutdown(wait=True)
        self._memory.close()
        self._file.close()
