# Integrating a PLE provider

Use the shared Connector, Reader and scheduler/output plumbing. An extension
implements `PLEProvider` and an importable factory; it does not duplicate the
request state machine or add worker-side SDK calls.

## Factory and configuration

```python
from collections.abc import Callable
from typing import Any

import torch

from vllm.v1.ple.provider import PLEProvider
from vllm.v1.ple.types import PLETableSpec


def create_remote_provider(
    *,
    table: PLETableSpec,
    load_table: Callable[[], torch.Tensor],
    options: dict[str, Any],
) -> PLEProvider:
    # Bind and validate the service's immutable table version and byte layout.
    # Do not call load_table unless this backend needs the local checkpoint rows.
    return RemoteProvider(table=table, options=options)
```

Configure an installed extension through `EngramConfig`:

```json
{
  "prefill_ple_provider": "my_extension.ple.create_remote_provider",
  "prefill_ple_provider_options": {},
  "prefill_ple_max_preparations": 16,
  "prefill_ple_timeout": 30
}
```

The default backend remains unchanged unless explicitly enabled. The built-in
`"local"` backend accepts `pool_bytes` and `directory` options and defaults to a
512 MiB `/dev/shm` result pool. The compatibility option `prefill_ple_pool_bytes`
also enables LocalProvider; it must not be passed with an external backend.
There is no Connector byte reservation or `reservation_bytes()` API.

`PLETableSpec` identifies `model`, `revision`, `name`, `num_rows`, `row_bytes`
and `dtype` (a string such as `"torch.bfloat16"`). Model code reads checkpoint
layout/hash metadata without materializing the table for external backends.
The factory must bind the corresponding immutable remote table and validate
its version and layout. `load_table()` is a lazy CPU uint8 matrix loader.

## Shared types

Import from `vllm.v1.ple.types`:

| Type | Fields and contract |
| --- | --- |
| `PoolDescriptor` | `pool_id: str`, `size: int`, either `path: str` or `handles: tuple`, and `registration: "pool" \| "region"` |
| `PLEChunk` | `offset: int`, `nbytes: int`; valid bytes relative to pool start |
| `PreparedRows` | `result_id: str`, `pool_id: str`, `chunks: tuple[PLEChunk, ...]` in input row order |
| `PLEReaderConfig` | Pool descriptor, common `token_bytes` and dtype; built by Connector |
| `PLEResult` | Common token binding `start`, `end`, `rows: PreparedRows`; never constructed by the provider |

`pool_id` includes generation identity: recreating storage must not silently
reuse it. Export one fixed, writable shared mapping for the engine lifetime.
No resizing, remapping or process addresses in descriptors. A path must identify
the same file for all workers. FD handles must implement `detach()` in the
receiver, for example `multiprocessing.reduction.DupFd`; supply one independently
consumable handle per rank in rank order.

Chunks can be noncontiguous and split a row. Concatenating their valid bytes
must produce exactly `row_ids.numel() * table.row_bytes` bytes in input order,
including repeated rows. Each chunk must lie within the exported pool. Only
valid bytes are described, not allocation padding or SDK metadata. Hold their
storage and read references until release succeeds.

## Implementing preparation and release

```python
class PLEProvider(Protocol):
    def export_pool(self, num_workers: int) -> PoolDescriptor: ...
    def prepare(self, result_id: str, row_ids: torch.Tensor) -> Future[PreparedRows]: ...
    def release(self, result_id: str) -> Future[None]: ...
    def close(self) -> None: ...
```

`export_pool()` runs at initialization. Common hashing calls `prepare()` from its
background thread. `release()` runs on the engine thread; callbacks can run on
provider threads. Protect provider-owned state accordingly. Both asynchronous
methods must return promptly: capacity waits, fetch, acquire and cleanup belong
in provider/SDK background execution, not on the caller's thread.

Connector allocates a unique string `result_id` **before** hashing and production.
Retain the input CPU int64 vector until production stops reading it. The provider
must not reinterpret it as tokens, deduplicate/reorder the output, compute model
hashes, or decide which KV prefix is reusable.

The preparation Future succeeds only after fetch, local ownership acquisition
and validation all complete. Keep SDK Result objects and read references in a
provider-owned map keyed by `result_id`. Return offsets and lengths, not those
objects. No worker attach RPC or readiness ACK is needed.

On an SDK exception before result delivery, surface the failure and allow new
requests immediately. SDK owns its internal cleanup; upper layers do not track
isolated memory or await a drain notification. Once a result or read reference
has been obtained, retain a cleanup entry even if subsequent operations fail.

`release(result_id)` is idempotent, including unknown/already-released IDs.
It resolves after the production operation and provider-owned cleanup finish.
Cancellation or deadline expiry can trigger release as soon as `prepare()`
returns, while its Future is still pending. Withdraw queued work when possible,
and release any result acquired by work already in flight. Never
implement cancellation by cancelling a Future and forgetting the actual SDK
operation. A preparation `TimeoutError` reaches the client as a retryable request
error. A provider cancellation fails only the affected request. A release exception must preserve enough ownership state to retry.
For SDK-managed internal cleanup after a failed operation, release need not wait
for physical memory reclamation. Do not add byte accounting or cleanup callbacks
to Connector.

`close()` runs after workers stop and result releases finish. Close SDK clients
and pool exports there. Worker failure requires terminating consumers before
releasing their results. If draining exceeds the worker shutdown timeout, the
Connector reports unresolved result IDs and retains resources for another
close attempt. Each DP engine exports its pool to its own executor ranks.

## Transport metrics

An adapter can declare a keyword-only `observation: PLEFetchObservation | None = None`
on `prepare(result_id, row_ids)`. The Connector recognizes this at startup.
Existing two-argument adapters remain usable. Only instrumented adapters
publish transport samples.

The Connector starts the total timer when it admits a preparation, before
background hashing. Wrap each SDK stage in `with observation.measure("prepare")`
or `with observation.measure("fetch")`. After fetching and validating the byte
count, call `observation.finish()` **before** acquiring READ references.
On a transport exception, call `observation.finish(exc)` and propagate the
exception. Completion is idempotent. Once transport succeeds, a later READ
failure still fails the request and follows normal cleanup. The seven Prometheus
gauges and their window are described in the [metrics guide](../usage/metrics.md#ple-transport).

## Adapter migration checklist

1. Replace any `prepare(tokens, start)` method with ordered row-ID input and the
   caller-assigned result ID. Remove provider hashing and token-range ownership.
2. Export the provider's stable pool once. Keep fetch, read acquisition, memory
   assembly and validation entirely inside the provider.
3. Translate the acquired local result into ordered pool-relative valid byte
   chunks. Bind token ranges only in the common Connector.
4. Remove worker SDK clients and per-request attach notifications. Use common
   Reader registration/H2D and `SchedulerOutput.ple_results`.
5. Use `ple_retire_ids` and all-rank `ple_consumed_result_ids` for reclamation;
   retain common cancellation, timeout and release-retry behavior unchanged.
6. Validate SDK failure without a drain event, acquisition failure after a result
   exists, cancellation with late success, release retry, and delayed/missing
   consumer ranks. Then run real SDK/GPU tests; an in-memory adapter is insufficient
   to establish remote transport correctness.

The common lifecycle and CUDA tests live in
`tests/v1/core/test_ple_connector.py`. Provider-specific tests should exercise
these same ownership boundaries using the actual SDK and deployment topology.
