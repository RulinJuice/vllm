# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Future
from typing import Protocol

import torch

from vllm.v1.metrics.ple import PLEFetchObservation

from .types import PoolDescriptor, PreparedRows


class PLEProvider(Protocol):
    """Own storage and production for one immutable table and stable local pool.

    Calls must return promptly; I/O and capacity waits belong in the provider's
    background work. Futures must not be cancelled by the caller.
    """

    def export_pool(self, num_workers: int) -> PoolDescriptor:
        """Export once, before workers start consuming results."""
        ...

    def prepare(self, result_id: str, row_ids: torch.Tensor) -> Future[PreparedRows]:
        """Prepare CPU int64 IDs in order, including duplicates.

        Retain row_ids until production finishes. Success means every returned
        byte is locally readable and held until release. Keep SDK objects and
        READ references here; only offsets and valid lengths cross processes.
        A failed SDK operation cleans its own internal resources independently.
        After acquiring a result, retain a cleanup entry even on later failure.
        """
        ...

    def release(self, result_id: str) -> Future[None]:
        """Idempotently finish production and release provider-owned resources.

        May be called as soon as prepare() returns, while its Future is still
        pending. Withdraw queued work or dispose of a late result before this
        Future succeeds. Unknown IDs succeed. A failed release remains retryable.
        SDK-internal cleanup after failed production may continue independently.
        """
        ...

    def close(self) -> None:
        """Close the pool after production and all worker use have ended."""
        ...


class ObservedPLEProvider(PLEProvider, Protocol):
    """Optional transport instrumentation for providers with a fetch boundary."""

    def prepare(
        self,
        result_id: str,
        row_ids: torch.Tensor,
        *,
        observation: PLEFetchObservation | None = None,
    ) -> Future[PreparedRows]: ...
