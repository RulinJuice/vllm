# Scheduler-side PLE preparation

Qwen4Exp prefill prepares the uncached prompt's PLE rows in the EngineCore
process while workers compute other requests. A provider owns the table access
and result storage. The common connector handles hashing, request lifetimes and
scheduling; the common reader maps local storage and performs H2D.

This is opt-in for Linux CUDA, ModelRunner V2 and one BF16 or global FP8
Qwen4Exp PLE table. Each single-node engine (PP=1) owns its provider and pool.
Its TP and context-parallel workers consume rank-local token segments. The
model and runner's normal topology restrictions apply. Text prefill requests
use `max_tokens=1`. A KV producer can also configure Qwen4 MTP. See the
[configuration guide](../features/engram.md#scheduler-side-prefill-qwen4exp).

## Responsibilities

| Component | Owns |
| --- | --- |
| Model adapter | Table layout and snapshot, checkpoint hash parameters, CPU n-gram hashing |
| Scheduler | Confirmed KV frontier, token budgets, compute admission |
| `PLEConnector` | Result IDs, background hashing, request/token binding, deadlines, preparation count, consumption and release tracking |
| `PLEProvider` | Local or remote access, allocation/capacity, local result ownership, asynchronous preparation and release |
| `PLEReader` | Stable pool mapping, CUDA registration, packed byte copies and H2D completion |
| Executor | Normal scheduling transport and intersection of all consuming ranks' output acknowledgments |

Connector has no byte budget, allocator or storage-specific SDK references.
Provider is an interface, not an additional process. LocalProvider uses a
background thread; an external provider may use its SDK's executors.

## Sequence

```mermaid
sequenceDiagram
    participant K as KV connector
    participant S as Scheduler
    participant C as PLEConnector
    participant P as Provider
    participant W as TP workers / Reader

    Note over C,W: Startup: export stable pool, map once in each worker
    K-->>S: Expected reusable prefix (async KV hit)
    S->>C: prefetch(request, tokens, expected_start)
    Note over K,P: KV transfer and PLE lookup may proceed concurrently
    C->>C: Background CPU hash, preserving order and duplicates
    C->>P: prepare(result_id, row_ids)
    P-->>C: Pending Future
    Note over S,W: Other requests may compute during preparation
    P->>P: Fetch, acquire local ownership and validate
    P-->>C: Future succeeds with PreparedRows
    K-->>S: KV load completes with actual coverage
    S->>C: ensure_ready(actual_start), then dispatch
    C-->>S: Token range + PreparedRows
    S->>W: SchedulerOutput.ple_results
    W->>W: Register if needed, H2D on copy stream
    Note over W: GPU waits at PLE consumer; later chunks reuse prepared bytes
    S->>C: finish on completion / cancellation
    S->>W: SchedulerOutput.ple_retire_ids
    W-->>C: ModelRunnerOutput, aggregated across every rank
    C->>P: release(result_id)
    P-->>C: Release Future completes
```

Hash completion immediately calls `prepare()` in the common background task.
It does not wait for a scheduler step or occupy the hash thread waiting for
provider I/O. Provider completion updates the preparation Future. Admission and
descriptor broadcast happen in normal scheduling; there is no attach RPC,
worker bypass notification or new EngineCore event loop.

## KV coverage and admission

When a native lookup selects asynchronous KV loading, the scheduler calls
`prefetch()` with the expected first compute position before allocating KV
blocks. Local KV hits start preparation when the scheduler first considers the
request, even with an exhausted token budget or no free running slots. At GPU
admission, `ensure_ready()` checks the actual KV frontier and the result. A
synchronous external hit returns a request error.

The prepared range is the complete suffix `[first_compute_token, prompt_len)`.
For an 8,192-token prompt with 6,144 reusable tokens, prepare the remaining
2,048 tokens. Earlier tokens supply n-gram context only. A full cache hit that
requires last-token replay prepares that last token.

Before admission, the scheduler revalidates the KV frontier. A later frontier
can use a subset of an existing result. A backwards move retires that result
and prepares a new generation with a fresh deadline. Chunked prefill and preemption retain a valid
result; each scheduled step carries its descriptor and the current token range.

`prefill_ple_max_preparations` bounds retained results by count, including
pending, ready and retiring results. `prefill_ple_timeout` bounds admission and
production. The completion callback checks the deadline before publishing the
result. A timely result can be dispatched on a later scheduling pass.
LocalProvider handles pool capacity and fragmented free space internally.

## Provider contract

The complete API and extension factory are in the
[provider integration guide](ple_provider_integration.md). Its core is:

```python
class PLEProvider(Protocol):
    def export_pool(self, num_workers: int) -> PoolDescriptor: ...
    def prepare(self, result_id: str, row_ids: torch.Tensor) -> Future[PreparedRows]: ...
    def release(self, result_id: str) -> Future[None]: ...
    def close(self) -> None: ...
```

The model adapter binds a fixed table version and passes a contiguous CPU int64
vector in token/head order, preserving duplicate IDs. Providers receive neither
prompt tokens nor KV information. Output is raw checkpoint bytes, so BF16/FP8
encoding and the model's dequantization remain unchanged.

A successful preparation means data is fully local and all provider-side
ownership/acquire and validation have succeeded. `PreparedRows` contains only
the result ID, pool generation ID and ordered `(offset, nbytes)` chunks. Token
ranges belong to Connector. Process pointers and SDK objects never reach workers.

Failure ownership is explicit:

- If SDK production fails before delivering a result, SDK owns its internal
  cleanup. Connector reports the request failure and continues other requests;
  it does not wait for SDK drain or physical reclamation.
- After a provider obtains a result, it retains the result and acquired read
  references under `result_id`, including when later acquisition or validation
  fails. `release()` cleans them up and can be retried.
- Cancellation withdraws queued hashing or a local pool wait. If submission has
  reached the provider, `release()` can run while preparation is pending. The
  Connector retains cleanup state for a late success.
- Once dispatched, release requires every consuming rank to confirm its final
  H2D and drop local use. A missing rank never counts as acknowledgment.
- Release failures retain the cleanup entry and retry with bounded backoff.
  Successful release has no byte-credit return and need not await SDK-owned
  cleanup after failed production.

## Worker mapping, registration and copies

`export_pool()` is called once per engine. Workers map the stable backing file
using a shared path or one transferable FD handle per rank. Pool size and
identity stay fixed for the lifetime of the engine. LocalProvider supplies a
named file; external providers may export their existing shared pool.

The descriptor selects `registration="pool"` (pin once at startup) or
`registration="region"` (pin pages needed by results before H2D). Reader tracks
shared registration pages so retiring one result cannot unregister a page still
used by another result. Registration failures remain worker/runtime errors;
provider readiness eliminates per-request SDK interactions, not hardware errors.

Reader copies the current `InputBatch` ranges into fixed device staging rows,
using packed query offsets and preserving ordered fragments even at non-row byte
boundaries. Padding is zeroed. The copy stream waits for the previous GPU
consumer before overwriting the staging buffer. The model waits on a readiness
event at the PLE layer, permitting earlier layers to overlap H2D. CUDA Graphs
use the same stable staging buffer and external event dependency.

Retirement IDs travel through `SchedulerOutput`, including zero-token steps.
Reader reports an ID only after its final H2D event completes and local use is
released. `ModelRunnerOutput.ple_consumed_result_ids` is intersected across all
ranks alongside existing KV/EC output aggregation. Repeated retirement IDs are
safe. The engine continues cleanup after the last request finishes. Shutdown
stops workers before releasing provider storage, including a partial Reader startup
failure. After worker teardown, cleanup uses the worker shutdown timeout. On expiry
the Connector reports outstanding result IDs and retains ownership for a retry.

Transport timestamps and the per-engine Prometheus gauges are documented in
[PLE transport metrics](../usage/metrics.md#ple-transport).

## Validation

`tests/v1/core/test_ple_connector.py` covers raw row bytes, checkpoint metadata,
hashing, KV frontier and early preparation, async failure/cancellation/timeout,
release retry, fragmented local allocation, external factory loading, all-rank
consumption, and CUDA Reader copies using path/FD and pool/region registration
in eager and graph execution. The V1 Core CPU and GPU CI jobs select these tests.

Changes affecting serving also require a real model run, comparing generated
tokens and log probabilities with the prior backend at matched inputs and
batching settings. Unit tests do not establish performance or remote SDK health.
