# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from itertools import count
from uuid import uuid4

import torch

from vllm.logger import init_logger

from .provider import PLEProvider
from .types import PLEReaderConfig, PLEResult, PreparedRows

logger = init_logger(__name__)


@dataclass
class _Preparation:
    result_id: str
    start: int
    end: int
    deadline: float
    future: Future[PreparedRows] = field(default_factory=Future)
    completed_at: float = 0
    provider_started: bool = False
    dispatched: bool = False
    release: Future[None] | None = None
    retry_at: float = 0
    retry_delay: float = 0.1


class PLEConnector:
    """Bind asynchronous row preparation to requests on the EngineCore thread.

    Background callbacks only complete their own preparation, never mutate
    scheduler state. Providers own storage and all capacity decisions.
    """

    def __init__(
        self,
        provider: PLEProvider,
        row_ids: Callable[[Sequence[int], int], torch.Tensor],
        *,
        num_heads: int,
        row_bytes: int,
        dtype: torch.dtype,
        max_preparations: int,
        timeout: float,
        num_workers: int = 1,
    ) -> None:
        if min(num_heads, row_bytes, max_preparations, timeout) <= 0:
            raise ValueError("PLE dimensions and preparation limits must be positive")
        self._provider = provider
        self._row_ids = row_ids
        self._num_heads = num_heads
        self._timeout = timeout
        self._max_preparations = max_preparations
        self.reader_config = PLEReaderConfig(
            provider.export_pool(num_workers), num_heads * row_bytes, str(dtype)
        )
        self._session = uuid4().hex
        self._ids = count()
        self._requests: dict[str, _Preparation] = {}
        self._retiring: dict[str, _Preparation] = {}
        self._deadlines: dict[str, float] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ple-hash"
        )
        self._closed = False

    def _complete(
        self,
        preparation: _Preparation,
        result: PreparedRows | None = None,
        error: str | None = None,
    ) -> None:
        preparation.completed_at = time.monotonic()
        if error is not None:
            preparation.future.set_exception(RuntimeError(error))
        else:
            assert result is not None
            preparation.future.set_result(result)

    def _prepare(self, preparation: _Preparation, tokens: Sequence[int]) -> None:
        error = None
        try:
            ids = self._row_ids(tokens, preparation.start)
            if (
                ids.device.type != "cpu"
                or ids.dtype != torch.int64
                or ids.ndim != 1
                or not ids.is_contiguous()
                or ids.numel()
                != (preparation.end - preparation.start) * self._num_heads
            ):
                raise ValueError("Model PLE hashing returned invalid row IDs")
            preparation.provider_started = True
            future = self._provider.prepare(preparation.result_id, ids)
            # Continue immediately after hashing; never wait for provider I/O
            # on the hashing thread.
            future.add_done_callback(lambda done: self._prepared(preparation, done))
        except Exception as exc:
            error = str(exc) or type(exc).__name__
        if error is not None:
            self._complete(preparation, error=error)

    def _prepared(
        self, preparation: _Preparation, future: Future[PreparedRows]
    ) -> None:
        error = None
        result = None
        try:
            result = future.result()
            pool = self.reader_config.pool
            if (
                result.result_id != preparation.result_id
                or result.pool_id != pool.pool_id
                or not result.chunks
                or any(
                    c.offset < 0 or c.nbytes <= 0 or c.offset + c.nbytes > pool.size
                    for c in result.chunks
                )
                or sum(c.nbytes for c in result.chunks)
                != (preparation.end - preparation.start)
                * self.reader_config.token_bytes
            ):
                raise ValueError("Provider returned an invalid PLE result descriptor")
        except Exception as exc:
            error = str(exc) or type(exc).__name__
        self._complete(preparation, result, error)

    def ensure_ready(self, req_id: str, tokens: Sequence[int], start: int) -> bool:
        if self._closed or not 0 <= start < len(tokens):
            raise ValueError(
                "PLE requires an open connector and nonempty prompt suffix"
            )
        self.poll()
        deadline = self._deadlines.setdefault(req_id, time.monotonic() + self._timeout)
        preparation = self._requests.get(req_id)
        if preparation is not None and not (
            preparation.start <= start and preparation.end == len(tokens)
        ):
            self._retiring[preparation.result_id] = self._requests.pop(req_id)
            preparation = None
        if preparation is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for PLE preparation admission")
            if len(self._requests) + len(self._retiring) >= self._max_preparations:
                return False
            preparation = _Preparation(
                f"{self._session}-{next(self._ids)}", start, len(tokens), deadline
            )
            self._requests[req_id] = preparation
            self._executor.submit(self._prepare, preparation, tuple(tokens))
        if (preparation.completed_at or time.monotonic()) >= preparation.deadline:
            raise TimeoutError("Timed out waiting for PLE rows")
        if not preparation.future.done():
            return False
        preparation.future.result()
        return True

    def dispatch(self, req_id: str) -> PLEResult:
        preparation = self._requests[req_id]
        rows = preparation.future.result()
        preparation.dispatched = True
        self._deadlines.pop(req_id, None)
        return PLEResult(preparation.start, preparation.end, rows)

    def finish(self, req_id: str) -> None:
        self._deadlines.pop(req_id, None)
        if (preparation := self._requests.pop(req_id, None)) is not None:
            self._retiring[preparation.result_id] = preparation
        self.poll()

    def retire_ids(self) -> set[str]:
        self.poll()
        return {key for key, value in self._retiring.items() if value.dispatched}

    def update_from_output(self, consumed: set[str]) -> None:
        for result_id in consumed:
            if (preparation := self._retiring.get(result_id)) is not None:
                preparation.dispatched = False
        self.poll()

    def poll(self) -> None:
        for key, preparation in list(self._retiring.items()):
            if preparation.dispatched or not preparation.future.done():
                continue
            if not preparation.provider_started:
                del self._retiring[key]
                continue
            if preparation.release is None:
                if time.monotonic() < preparation.retry_at:
                    continue
                try:
                    preparation.release = self._provider.release(key)
                except Exception as exc:
                    self._retry_release(preparation, exc)
                    continue
            if not preparation.release.done():
                continue
            try:
                preparation.release.result()
            except Exception as exc:
                self._retry_release(preparation, exc)
            else:
                del self._retiring[key]

    def _retry_release(self, preparation: _Preparation, error: Exception) -> None:
        logger.warning(
            "PLE release %s failed; will retry: %s", preparation.result_id, error
        )
        preparation.release = None
        preparation.retry_at = time.monotonic() + preparation.retry_delay
        preparation.retry_delay = min(5.0, preparation.retry_delay * 2)

    @property
    def has_cleanup(self) -> bool:
        return bool(self._retiring)

    def close(self, *, consumers_stopped: bool = False) -> None:
        preparations = [*self._requests.values(), *self._retiring.values()]
        if not consumers_stopped and any(p.dispatched for p in preparations):
            raise RuntimeError("Stop PLE consumers before closing dispatched results")
        self._closed = True
        for req_id in list(self._requests):
            self.finish(req_id)
        if consumers_stopped:
            for preparation in self._retiring.values():
                preparation.dispatched = False
        while self._retiring:
            self.poll()
            pending = [p.future for p in self._retiring.values() if not p.future.done()]
            if pending:
                wait(pending, timeout=0.001)
            elif self._retiring:
                time.sleep(0.001)
        self._executor.shutdown(wait=True)
        self._provider.close()
