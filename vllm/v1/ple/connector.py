# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from itertools import count
from threading import Event, Lock
from time import monotonic, sleep
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.utils.func_utils import supports_kw
from vllm.v1.metrics.ple import PLEFetchObservation, PLEMetrics

from .provider import ObservedPLEProvider, PLEProvider
from .types import PLEReaderConfig, PLEResult, PreparedRows

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


@dataclass
class _RequestState:
    result_id: str
    start: int
    end: int
    deadline: float
    observation: PLEFetchObservation | None
    ready: Future[PreparedRows] = field(default_factory=Future)
    retired: Event = field(default_factory=Event)
    # The hash task returns after prepare() registers the ID, without waiting
    # for I/O. Only this private task can be cancelled before it starts.
    submission: Future[bool] = field(init=False)
    dispatched: bool = False
    release: Future[None] | None = None
    retry_at: float = 0.0
    retry_delay: float = 0.1


class PLEConnector:
    """Bind provider results to requests and retire them after all consumers."""

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
        self.metrics = PLEMetrics()
        self._provider = provider
        try:
            observed = supports_kw(
                provider.prepare,
                "observation",
                requires_kw_only=True,
                allow_var_kwargs=False,
            )
        except (TypeError, ValueError):
            # A provider can expose a native callable without a Python signature.
            observed = False
        self._observed_prepare = (
            cast(ObservedPLEProvider, provider).prepare if observed else None
        )
        self._row_ids = row_ids
        self._num_heads = num_heads
        self._timeout = timeout
        self._max_preparations = max_preparations
        self.reader_config = PLEReaderConfig(
            provider.export_pool(num_workers), num_heads * row_bytes, str(dtype)
        )
        self._session = uuid4().hex
        self._ids = count()
        self._requests: dict[str, _RequestState] = {}
        self._retiring: dict[str, _RequestState] = {}
        self._admission_deadlines: dict[str, float] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ple-hash"
        )
        self._lock = Lock()
        self._closed = False
        self._draining = False
        self._provider_closed = False

    def _complete(
        self,
        state: _RequestState,
        *,
        rows: PreparedRows | None = None,
        error: BaseException | None = None,
    ) -> None:
        # Publish a single deadline decision, including when admission races
        # the provider callback. Never retain an SDK traceback or pool views.
        if state.observation is not None:
            state.observation.finish(error)
        with self._lock:
            if state.ready.done():
                return
            if monotonic() >= state.deadline:
                state.ready.set_exception(
                    TimeoutError("Timed out waiting for PLE rows")
                )
            elif error is not None:
                kind = TimeoutError if isinstance(error, TimeoutError) else RuntimeError
                state.ready.set_exception(kind(str(error) or type(error).__name__))
            else:
                assert rows is not None
                state.ready.set_result(rows)

    def _prepare(self, state: _RequestState, tokens: Sequence[int]) -> bool:
        try:
            ids = self._row_ids(tokens, state.start)
            if (
                ids.device.type != "cpu"
                or ids.dtype != torch.int64
                or ids.ndim != 1
                or not ids.is_contiguous()
                or ids.numel() != (state.end - state.start) * self._num_heads
            ):
                raise ValueError("Model PLE hashing returned invalid row IDs")
        except (Exception, asyncio.CancelledError) as exc:
            self._complete(state, error=exc)
            return False
        if state.retired.is_set():
            if state.observation is not None:
                state.observation.finish()
            return False
        try:
            if self._observed_prepare is None:
                future = self._provider.prepare(state.result_id, ids)
            else:
                future = self._observed_prepare(
                    state.result_id, ids, observation=state.observation
                )
            future.add_done_callback(lambda done: self._prepared(state, done))
        except (Exception, asyncio.CancelledError) as exc:
            self._complete(state, error=exc)
        # Even a synchronous prepare failure can have provider-owned resources.
        return True

    def _prepared(self, state: _RequestState, future: Future[PreparedRows]) -> None:
        try:
            rows = future.result()
            pool = self.reader_config.pool
            if (
                rows.result_id != state.result_id
                or rows.pool_id != pool.pool_id
                or not rows.chunks
                or any(
                    c.offset < 0 or c.nbytes <= 0 or c.offset + c.nbytes > pool.size
                    for c in rows.chunks
                )
                or sum(c.nbytes for c in rows.chunks)
                != (state.end - state.start) * self.reader_config.token_bytes
            ):
                raise ValueError("Provider returned an invalid PLE result descriptor")
        except (Exception, asyncio.CancelledError) as exc:
            self._complete(state, error=exc)
        else:
            self._complete(state, rows=rows)

    def _submit(
        self, req_id: str, tokens: Sequence[int], start: int, deadline: float
    ) -> _RequestState | None:
        if len(self._requests) + len(self._retiring) >= self._max_preparations:
            return None
        state = _RequestState(
            f"{self._session}-{next(self._ids)}",
            start,
            len(tokens),
            deadline,
            self.metrics.begin(deadline)
            if self._observed_prepare is not None
            else None,
        )
        state.submission = self._executor.submit(self._prepare, state, tuple(tokens))
        self._requests[req_id] = state
        self._admission_deadlines.pop(req_id, None)
        return state

    def prefetch(
        self,
        req_id: str,
        tokens: Sequence[int],
        start: int,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        """Best-effort submission at KV lookup; compute rechecks actual coverage."""
        with self._lock:
            if self._closed or req_id in self._requests or not 0 <= start < len(tokens):
                return
            if cancelled is not None and cancelled():
                return
            deadline = self._admission_deadlines.get(
                req_id, monotonic() + self._timeout
            )
            if monotonic() < deadline:
                self._submit(req_id, tokens, start, deadline)

    def _retire(self, req_id: str) -> None:
        self._admission_deadlines.pop(req_id, None)
        if (state := self._requests.pop(req_id, None)) is not None:
            if state.observation is not None:
                state.observation.abandon()
            state.retired.set()
            if state.submission.cancel() and state.observation is not None:
                state.observation.finish()
            self._retiring[state.result_id] = state

    def ensure_ready(self, req_id: str, tokens: Sequence[int], start: int) -> bool:
        self.poll()
        with self._lock:
            if self._closed or not 0 <= start < len(tokens):
                raise ValueError(
                    "PLE requires an open connector and nonempty prompt suffix"
                )
            state = self._requests.get(req_id)
            if state is not None and (start < state.start or state.end != len(tokens)):
                self._retire(req_id)
                state = None
            now = monotonic()
            if state is None:
                deadline = self._admission_deadlines.setdefault(
                    req_id, now + self._timeout
                )
                if now >= deadline:
                    raise TimeoutError(
                        "Timed out waiting for PLE preparation admission"
                    )
                state = self._submit(req_id, tokens, start, deadline)
                if state is None:
                    return False
            if not state.ready.done():
                if now < state.deadline:
                    return False
                state.ready.set_exception(
                    TimeoutError("Timed out waiting for PLE rows")
                )
            state.ready.result()
            return True

    def dispatch(self, req_id: str) -> PLEResult:
        state = self._requests[req_id]
        if not state.ready.done():
            raise RuntimeError("Cannot dispatch PLE before preparation completes")
        rows = state.ready.result()
        state.dispatched = True
        return PLEResult(state.start, state.end, rows)

    def update_scheduler_output(self, output: "SchedulerOutput") -> None:
        output.ple_results = {
            key: self.dispatch(key) for key in output.num_scheduled_tokens
        }
        output.ple_retire_ids = self.retire_ids()

    def finish(self, req_id: str) -> None:
        with self._lock:
            self._retire(req_id)
        self.poll()

    def retire_ids(self) -> set[str]:
        self.poll()
        return {key for key, state in self._retiring.items() if state.dispatched}

    def update_from_output(self, consumed: set[str]) -> None:
        for key in consumed:
            if (state := self._retiring.get(key)) is not None:
                state.dispatched = False
        self.poll()

    def poll(self) -> None:
        for key, state in list(self._retiring.items()):
            if state.dispatched or not state.submission.done():
                continue
            # Once registered, release can withdraw a capacity wait or collect
            # a late result, even while the prepare Future is pending.
            registered = not state.submission.cancelled() and state.submission.result()
            if registered and not self._release(state):
                continue
            with self._lock:
                del self._retiring[key]

    def _release(self, state: _RequestState) -> bool:
        try:
            if state.release is None:
                if monotonic() < state.retry_at:
                    return False
                state.release = self._provider.release(state.result_id)
            if not state.release.done():
                return False
            state.release.result()
        except (Exception, asyncio.CancelledError) as exc:
            logger.warning(
                "PLE release %s failed; will retry: %s", state.result_id, exc
            )
            state.release = None
            state.retry_at = monotonic() + state.retry_delay
            state.retry_delay = min(5.0, state.retry_delay * 2)
            return False
        return True

    @property
    def has_cleanup(self) -> bool:
        return bool(self._retiring)

    def begin_shutdown(self) -> None:
        # Live requests still run. Once only PLE cleanup is left, stop workers
        # and let close() drain within the shutdown deadline.
        self._draining = True

    @property
    def needs_engine_step(self) -> bool:
        return self.has_cleanup and not self._draining

    def close(
        self, *, consumers_stopped: bool = False, timeout: float | None = None
    ) -> None:
        if self._provider_closed:
            return
        if timeout is None:
            timeout = envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS
        deadline = monotonic() + timeout
        with self._lock:
            if not consumers_stopped and any(
                state.dispatched
                for state in (*self._requests.values(), *self._retiring.values())
            ):
                raise RuntimeError(
                    "Stop PLE consumers before closing dispatched results"
                )
            self._closed = True
            for req_id in list(self._requests):
                self._retire(req_id)
            self._admission_deadlines.clear()
        for state in self._retiring.values():
            state.dispatched = False
        while self._retiring:
            self.poll()
            if not self._retiring:
                break
            now = monotonic()
            if now >= deadline:
                outstanding = ", ".join(self._retiring)
                raise TimeoutError(
                    f"PLE shutdown timed out with unreleased results: {outstanding}"
                )
            pending: list[Future] = []
            wake_at = deadline
            for state in self._retiring.values():
                if state.release is not None:
                    pending.append(state.release)
                elif state.retry_at:
                    wake_at = min(wake_at, state.retry_at)
                else:
                    pending.append(state.submission)
            delay = max(0.0, wake_at - now)
            if pending:
                wait(pending, timeout=delay, return_when=FIRST_COMPLETED)
            else:
                sleep(delay)
        self._executor.shutdown(wait=True)
        self._provider.close()
        self._provider_closed = True
