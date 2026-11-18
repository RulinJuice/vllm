# Engram: conditional memory via n-gram lookups

Engram is a *conditional memory* module attached to specific layers of a
transformer backbone. At each position, the preceding few tokens are hashed
into large static embedding tables, and the retrieved rows are fused into the
hidden state, gated by how well they match it. Where Mixture-of-Experts scales
capacity through conditional computation, Engram scales it through conditional
memory: parameters that are never computed over, only looked up, so capacity
grows at O(1) per-token cost. The mechanism and its design rationale are
described in [Conditional Memory via Scalable Lookup: A New Axis of Sparsity
for Large Language Models](https://arxiv.org/abs/2601.07372).

vLLM enables Engram automatically whenever the checkpoint declares Engram
layers, so basic serving needs no extra flags. This page explains what Engram
does and how to control its memory footprint and multi-GPU topology.

## Supported models

Engram layers are declared by the checkpoint, not chosen at serve time:

| Architecture | HF config field | Notes |
| --- | --- | --- |
| `DeepseekV41ForCausalLM` | `engram_layer_ids` | e.g. `deepseek-ai/DeepSeek-V4.1-Flash` |
| `Qwen4ExpForCausalLM` / `Qwen4ExpForConditionalGeneration` | `ple_layer_ids` | same mechanism, called PLE |

Passing `--engram-config` for a model without n-gram embedding layers fails at
startup, as does running on a non-CUDA-alike platform.

## How it works

Engram processes each position in two phases: *retrieval* and *fusion*.

**Retrieval** maps the local token context to static table rows:

1. **Tokenizer compression.** At startup, vLLM builds a many-to-one map from
   the tokenizer that collapses tokens which normalize alike (NFKC, accent
   stripping, lowercasing, whitespace collapsing), so `" The"`, `"the"` and
   `"THE"` hash identically. This concentrates semantic density into a much
   smaller id space; its size must match `engram_compressed_vocab_size` from
   the checkpoint, or startup fails — every hash multiplier derives from it.
2. **Multi-head hashing.** For each n-gram order `n` (up to
   `engram_max_ngram_size - 1`), several hash heads each apply an independent
   multiplicative-XOR hash over the compressed suffix n-gram. Every (n-gram
   order, head) pair indexes its own prime-sized bucket range in the layer's
   table — disjoint across pairs, so collisions cannot collide twice — and
   the retrieved rows are concatenated into one memory vector per position.

**Fusion** turns the retrieved rows into a context-aware contribution to the
hidden state. The rows are context-independent priors and may be noisy (hash
collisions, polysemy), so they are gated: a `wkv` projection produces one key
per hyper-connection branch plus a shared value, and each branch computes a
sigmoid gate over the normalized dot product between its hidden state —
which has already aggregated global context through preceding attention —
and its key. If the retrieved memory contradicts the current context, the
gate tends toward zero and suppresses it. The gated value is added
residually into the branch's stream (`hidden + gate * value`); positions that
take no part in an n-gram (e.g. image spans) are masked so they pass through
untouched.

Engram sits only on the layers listed in the checkpoint. Placement is a
deliberate hardware-algorithm trade-off: early enough that the backbone is
relieved from reconstructing static patterns in its first layers, deep enough
that vLLM can hide table lookups behind the compute of the preceding layers
(see below).

In vLLM, hashing, lookup, and gating run as fused Triton kernels over FP8
rows with ue8m0 block scales. The layer layout — table sizes, n-gram size,
head count — is fixed by the checkpoint and is not user-configurable. Only the
storage and sharding described below are configurable, via
`--engram-config`, which takes a JSON object and can also be set field by
field as `--engram-config.<field>`
([CLI reference](../cli/serve.md#-engram-config)).

## CPU offload

Engram tables dwarf typical embedding weights: [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
carries two Engram layers of roughly 384M table rows each, with 256-byte FP8
entries plus their ue8m0 block scales — about 200 GB of table weights alone.
Because lookups are deterministic — the indices are known from the token ids
alone, before the forward pass — the tables need not sit in GPU memory at
all. By default vLLM stores them in **pinned host memory** and serves lookups
directly from the Triton kernel through unified virtual addressing (UVA),
prefetching on a side CUDA stream so transfers overlap with the compute of
preceding layers. The freed GPU memory goes to the KV cache.

`cpu_offload` defaults to `true` (on)

To keep the tables resident on GPU instead (e.g. when host memory is
scarce):

```bash
vllm serve deepseek-ai/DeepSeek-V4.1-Flash \
  --engram-config.cpu_offload false
```

CPU offload requires a GPU with UVA support;
vLLM fails fast when it is unavailable.

## Disk offload (Qwen4Exp)

`disk_offload` keeps Qwen4Exp PLE tables in the checkpoint files: no table
is allocated in GPU or pinned host memory, and nothing is copied to an offload
directory. Each step reads that step's rows from the safetensors files on the
host. Use it where neither `cpu_offload` nor a resident table fits, for example
on unified-memory GPUs such as DGX Spark, where pinned host memory comes out of
the same pool as the weights and the KV cache:

```bash
vllm serve <qwen4exp-checkpoint> --engram-config '{"disk_offload": true}'
```

It applies only to Qwen4Exp on CUDA and takes precedence over `cpu_offload`.

Rows are read with `preadv` and copied to fixed device buffers before the
forward, so CUDA graphs replay unchanged. They stay in the checkpoint format
(BF16, FP8 or NVFP4) and are decoded on the GPU, so an unquantized table must
match `--dtype`. Rows that were read recently are served from the OS page
cache, which is shared by co-located ranks, not counted in the worker's RSS,
and reclaimed by the kernel under memory pressure.

Costs and limits:

- The read needs the step's token ids, so it waits for the previous step on
  the GPU, and every TP rank reads all of its rows.
- A row that is not in the page cache is read from disk. For a fast first
  request, read the PLE shard files once after the server has started.
- The checkpoint files must not change while the server runs. Truncation stops
  the engine with an error naming the file; deleting or renaming is safe.
- Weight reload and elastic EP are unsupported. Loading fails when tensors are
  not file-backed safetensors views (`.bin`, runai, `fastsafetensors`,
  `instanttensor`, or `--safetensors-load-strategy eager`);
  `--load-format dummy` serves zeros.

## Scheduler-side prefill (Qwen4Exp)

Set `prefill_ple_provider` to `"local"` on the prefill instance to enable
scheduler-side PLE with a local DRAM table. The compatibility option
`prefill_ple_pool_bytes` also enables LocalProvider.

```bash
vllm serve <qwen4exp-checkpoint> \
  --distributed-executor-backend mp \
  --engram-config '{"prefill_ple_provider": "local", "prefill_ple_provider_options": {"pool_bytes": 268435456}, "prefill_ple_timeout": 30}'
```

Each EngineCore loads one local table. Its workers map the shared result pool
and copy the current batch's rows to their GPUs. The PLE layer waits for the
copy when it consumes the rows.

A native asynchronous KV hit starts PLE lookup for the expected prompt suffix
before KV block allocation. Local hits can also start preparation while token
budget or running slots are exhausted. Compute waits for both the usable KV
and the PLE rows. If fewer KV tokens arrive than predicted, the scheduler
prepares the larger suffix with a fresh timeout. A full hit prepares the one
token needed for replay. Synchronous external KV hits return a request error.

This mode requires Linux CUDA, ModelRunner V2, one BF16 or global FP8
Qwen4Exp PLE table, and single-node `mp` or `uni` workers with PP=1. Each DP
engine owns its provider and table. TP and context-parallel workers use the
token segments supplied by the model runner, subject to its supported
topologies. Submit text prompts with `max_tokens=1`. Qwen4 MTP is supported on
a KV producer. The decode instance uses its configured decode backend.

LocalProvider owns the result pool; Connector does not maintain a byte budget.
The local pool defaults to 512 MiB. Size it for prompt suffixes that can run at the
same time. Each prepared token takes `ple_embed_dim` elements in the
checkpoint dtype. The pool uses `/dev/shm` and is registered for CUDA DMA
once per worker. A custom pool directory must support CUDA host registration.
A suffix larger than the pool fails immediately. The
`prefill_ple_timeout` value bounds the wait for rows or pool space in
seconds and defaults to 30. `prefill_ple_max_preparations` (default 16) bounds
the number of retained preparations, including pending, ready and retiring results.

External backends use an importable factory in `prefill_ple_provider` and
backend-specific `prefill_ple_provider_options`. They own allocation and
capacity while reusing the common hashing, lifecycle and Reader. See the
[provider integration guide](../design/ple_provider_integration.md) and the
[per-engine transport metrics](../usage/metrics.md#ple-transport).

See the [scheduler-side PLE design](../design/ple_connector.md) for the
provider contract and buffer lifecycle.

## Data-parallel topologies (DeepSeek V4.1)

By default (`embedding_across_dp: false`), each DP replica keeps its own
TP-sharded copy of the tables — no cross-replica communication, but one host
copy per replica.

With `embedding_across_dp: true`, the hash heads are sharded across all
TP × DP ranks into a single table copy. Each step gathers the hash ids of
every co-located DP replica, each rank looks up the heads it owns, and the
rows are exchanged back. This trades per-step DP collectives for a much
smaller table footprint:

```bash
vllm serve deepseek-ai/DeepSeek-V4.1-Flash \
  --tensor-parallel-size 2 --data-parallel-size 4 \
  --engram-config '{"embedding_across_dp": true}'
```

`embedding_across_dp` is not supported with elastic expert parallelism yet.
Qwen4Exp PLE tables are ETP-sharded instead and honor only `cpu_offload` and
`disk_offload`.

## Sharing host tables across DP replicas

When the tables are CPU-offloaded and DP replicas are co-located on one node,
`dp_shared_memory` stores **one copy of each TP shard in `/dev/shm`**, shared
by all co-located replicas through `cudaHostRegister`. Each replica then
prefetches only its own tokens — no per-step Engram DP collectives — and host
memory drops from one table copy per replica to one per node.

`dp_shared_memory` defaults to enabled whenever the other settings allow it,
and falls back to per-replica (or DP-sharded) tables with a warning when:

- the DP replicas are not co-located on a single node,
- `/dev/shm` cannot hold the full tables — in containers, raise the limit
  with `--shm-size` or `--ipc=host`,
- expert parallelism is elastic, which is unsupported.

Requirements: `cpu_offload` enabled, `--data-parallel-size > 1`, and a shared
IPC namespace across the co-located replicas.

## Limitations

- CUDA-alike platforms only.
- DeepSeek V4.1 Engram does not support DBO or microbatching; disable
  `--enable-dbo` and set `--ubatch-size 0`.
- `dp_shared_memory` requires `cpu_offload`, `data_parallel_size > 1`, and is
  unsupported with elastic expert parallelism.
