# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-engine PLE transport measurements, bucketed by completion time."""

import asyncio
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import CancelledError
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Literal

PLE_GAUGES = {
    "ple_inflight_requests": "PLE transports still in progress.",
    "ple_success_requests": "Successful PLE transports over 60s.",
    "ple_failed_requests": "Failed PLE transports over 60s.",
    "ple_latency_ms": "Mean successful PLE transport latency over 60s.",
    "ple_latency_max_ms": "Maximum successful PLE transport latency over 60s.",
    "ple_prepare_latency_ms": "Mean successful PLE prepare stage latency over 60s.",
    "ple_fetch_latency_ms": "Mean successful PLE fetch stage latency over 60s.",
}

_Stage = Literal["prepare", "fetch", "success", "failed"]


@dataclass
class _Samples:
    count: int = 0
    total_ms: float = 0
    max_ms: float = 0


class PLEMetrics:
    def __init__(self, clock: Callable[[], float] = monotonic):
        self._clock = clock
        self._lock = Lock()
        self._inflight = 0
        self._buckets: deque[tuple[int, dict[_Stage, _Samples]]] = deque()

    def begin(self, deadline: float) -> "PLEFetchObservation":
        with self._lock:
            self._inflight += 1
            return PLEFetchObservation(self, self._clock(), deadline)

    def _expire(self, now: float) -> None:
        while self._buckets and self._buckets[0][0] <= int(now) - 60:
            self._buckets.popleft()

    def _record(self, stage: _Stage, start: float, now: float) -> None:
        # Caller holds the lock; completion seconds arrive in order.
        self._expire(now)
        if not self._buckets or self._buckets[-1][0] != int(now):
            self._buckets.append((int(now), {}))
        sample = self._buckets[-1][1].setdefault(stage, _Samples())
        ms = (now - start) * 1000
        sample.count += 1
        sample.total_ms += ms
        sample.max_ms = max(sample.max_ms, ms)

    def snapshot(self) -> dict[str, float]:
        totals = {
            stage: _Samples() for stage in ("prepare", "fetch", "success", "failed")
        }
        with self._lock:
            self._expire(self._clock())
            inflight = self._inflight
            for _, bucket in self._buckets:
                for stage, sample in bucket.items():
                    total = totals[stage]
                    total.count += sample.count
                    total.total_ms += sample.total_ms
                    total.max_ms = max(total.max_ms, sample.max_ms)
        values = dict.fromkeys(PLE_GAUGES, 0.0)
        values["ple_inflight_requests"] = float(inflight)
        values["ple_success_requests"] = float(totals["success"].count)
        values["ple_failed_requests"] = float(totals["failed"].count)
        values["ple_latency_max_ms"] = totals["success"].max_ms
        for stage, metric in (
            ("success", "ple_latency_ms"),
            ("prepare", "ple_prepare_latency_ms"),
            ("fetch", "ple_fetch_latency_ms"),
        ):
            sample = totals[stage]
            if sample.count:
                values[metric] = sample.total_ms / sample.count
        return values


class PLEFetchObservation:
    """One transport, from hash submission through validation of fetched bytes."""

    def __init__(self, metrics: PLEMetrics, start: float, deadline: float):
        self._metrics = metrics
        self._start = start
        self.deadline = deadline
        self._abandoned = False
        self._finished = False

    @contextmanager
    def measure(self, stage: Literal["prepare", "fetch"]) -> Iterator[None]:
        """Record a successful provider stage, including on a later request failure."""
        if stage not in ("prepare", "fetch"):
            raise ValueError(f"Unknown PLE transport stage: {stage}")
        metrics = self._metrics
        start = metrics._clock()
        yield
        with metrics._lock:
            metrics._record(stage, start, metrics._clock())

    def abandon(self) -> None:
        metrics = self._metrics
        with metrics._lock:
            if metrics._clock() < self.deadline:
                self._abandoned = True

    def finish(self, error: BaseException | None = None) -> None:
        """Settle once. Providers call this before acquiring local READ references."""
        metrics = self._metrics
        with metrics._lock:
            if self._finished:
                return
            self._finished = True
            metrics._inflight -= 1
            now = metrics._clock()
            if self._abandoned:
                return
            if now >= self.deadline:
                outcome: _Stage = "failed"
            elif isinstance(error, (CancelledError, asyncio.CancelledError)):
                return
            else:
                outcome = "success" if error is None else "failed"
            metrics._record(outcome, self._start, now)
