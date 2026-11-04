# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import mmap
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import count

import torch

from vllm.logger import init_logger

from .provider import PLEProvider
from .types import PLEPool, PLEResult

logger = init_logger(__name__)


@dataclass
class _Lookup:
    result: PLEResult
    future: Future[str | None]
    dispatched: bool = False


class PLEConnector:
    """Own PLE result storage and admission state on the EngineCore thread."""

    def __init__(
        self,
        provider: PLEProvider,
        row_ids: Callable[[Sequence[int], int], torch.Tensor],
        *,
        num_heads: int,
        row_bytes: int,
        dtype: torch.dtype,
        pool_bytes: int,
        timeout: float,
        directory: str = "/dev/shm",
    ) -> None:
        self._provider = provider
        self._row_ids = row_ids
        self._row_bytes = row_bytes
        self._timeout = timeout
        # Keep the file named until shutdown so ranks can map it at startup.
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
        self.pool = PLEPool(
            self._file.name, pool_bytes, num_heads * row_bytes, str(dtype)
        )
        self._available = [(0, pool_bytes)]
        self._result_ids = count()
        self._lookups: dict[str, _Lookup] = {}
        self._retiring: dict[int, _Lookup] = {}
        self._deadlines: dict[str, float] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ple")

    def _gather(self, tokens: Sequence[int], result: PLEResult) -> str | None:
        try:
            size = (result.end - result.start) * self.pool.token_bytes
            out = torch.frombuffer(
                self._memory, dtype=torch.uint8, count=size, offset=result.offset
            ).view(-1, self._row_bytes)
            self._provider.lookup_into(self._row_ids(tokens, result.start), out)
        except Exception as exc:
            # Keep only the message, since a traceback can retain the mmap view.
            return str(exc) or type(exc).__name__
        return None

    def _allocate(self, size: int) -> int | None:
        for index, (start, end) in enumerate(self._available):
            if end - start >= size:
                if end - start == size:
                    self._available.pop(index)
                else:
                    self._available[index] = (start + size, end)
                return start
        return None

    def _free(self, result: PLEResult) -> None:
        end = result.offset + (result.end - result.start) * self.pool.token_bytes
        free_ranges = sorted([*self._available, (result.offset, end)])
        self._available = []
        for start, end in free_ranges:
            if self._available and self._available[-1][1] == start:
                self._available[-1] = (self._available[-1][0], end)
            else:
                self._available.append((start, end))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "PLE release result=%d pool_used_bytes=%d",
                result.result_id,
                self.pool.size - sum(end - start for start, end in self._available),
            )

    def ensure_ready(self, req_id: str, tokens: Sequence[int], start: int) -> bool:
        if not 0 <= start < len(tokens):
            raise ValueError("PLE requires a nonempty prompt suffix")
        deadline = self._deadlines.setdefault(req_id, time.monotonic() + self._timeout)
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for PLE rows")
        lookup = self._lookups.get(req_id)
        if lookup is not None:
            if lookup.result.start <= start and lookup.result.end == len(tokens):
                if not lookup.future.done():
                    return False
                if error := lookup.future.result():
                    raise RuntimeError(error)
                return True
            # Recompute the suffix when the valid KV prefix moves backwards.
            self._retire(self._lookups.pop(req_id))
        size = (len(tokens) - start) * self.pool.token_bytes
        if size > self.pool.size:
            raise ValueError("PLE prompt suffix exceeds the shared result pool")
        if (offset := self._allocate(size)) is None:
            return False
        result = PLEResult(next(self._result_ids), start, len(tokens), offset)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "PLE lookup request=%s result=%d range=[%d,%d) bytes=%d "
                "pool_used_bytes=%d",
                req_id,
                result.result_id,
                result.start,
                result.end,
                size,
                self.pool.size - sum(end - start for start, end in self._available),
            )
        self._lookups[req_id] = _Lookup(
            result, self._executor.submit(self._gather, tokens, result)
        )
        return False

    def dispatch(self, req_id: str) -> PLEResult:
        lookup = self._lookups[req_id]
        lookup.dispatched = True
        self._deadlines.pop(req_id, None)
        logger.debug(
            "PLE dispatch request=%s result=%d", req_id, lookup.result.result_id
        )
        return lookup.result

    def _retire(self, lookup: _Lookup) -> None:
        if not lookup.dispatched and (lookup.future.cancel() or lookup.future.done()):
            self._free(lookup.result)
        else:
            self._retiring[lookup.result.result_id] = lookup

    def finish(self, req_id: str) -> None:
        self._deadlines.pop(req_id, None)
        if (lookup := self._lookups.pop(req_id, None)) is not None:
            self._retire(lookup)

    def cleanup_ids(self) -> list[int]:
        dispatched = []
        for result_id, lookup in list(self._retiring.items()):
            if lookup.dispatched:
                dispatched.append(result_id)
            elif lookup.future.done():
                self.reclaim([result_id])
        return dispatched

    @property
    def has_cleanup(self) -> bool:
        return bool(self._retiring)

    def reclaim(self, result_ids: Sequence[int]) -> None:
        for result_id in result_ids:
            self._free(self._retiring.pop(result_id).result)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._memory.close()
        self._file.close()
