# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open

from vllm.utils.math_utils import cdiv, round_up

from .connector import PLEConnector
from .factory import create_provider
from .types import PLE_TABLE_SHARD, PLETableSpec

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def prompt_row_ids(
    tokens: Sequence[int],
    start: int,
    *,
    eos: int,
    multipliers: torch.Tensor,
    sizes: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Compute suffix IDs in token and head order with left n-gram context."""
    ngram_size = multipliers.numel()
    heads_per_order = sizes.numel() // (ngram_size - 1)
    context_start = max(0, start - ngram_size + 1)
    context = torch.tensor(tokens[context_start:], dtype=torch.int64, device="cpu")
    positions = torch.arange(context.numel(), device="cpu")
    previous_eos = torch.where(context == eos, positions, -1).cummax(0).values
    previous_eos = torch.cat((positions.new_full((1,), -1), previous_eos[:-1]))
    positions = positions[start - context_start :]
    segment_offset = positions - previous_eos[positions] - 1
    mixed = context[positions] * multipliers[0]
    by_order = []
    for shift in range(1, ngram_size):
        previous = context[(positions - shift).clamp_min(0)]
        previous = torch.where(segment_offset >= shift, previous, eos)
        mixed = mixed ^ (previous * multipliers[shift])
        heads = slice((shift - 1) * heads_per_order, shift * heads_per_order)
        by_order.append(mixed[:, None] % sizes[heads] + offsets[heads])
    return torch.cat(by_order, dim=1).flatten()


def load_qwen4_connector(config: "VllmConfig") -> PLEConnector:
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    from vllm.models.qwen4_exp.nvidia.ngram_embedding import Qwen4ExpNGramEmbedding

    model = config.model_config
    assert model is not None
    engram = config.engram_config
    assert engram is not None and engram.prefill_ple_enabled
    text = model.hf_text_config
    ngram_size = int(text.ngram_size)
    num_heads = (ngram_size - 1) * int(text.heads_per_ngram)
    head_dim = int(text.ple_embed_dim) // num_heads
    default_sizes, default_offsets, total = Qwen4ExpNGramEmbedding._make_vocab_layout(
        ngram_vocab_size_base=int(text.ngram_vocab_size_base),
        ngram_heads=num_heads,
        ple_dense_layer_id=0,
    )
    rows = round_up(total, int(text.make_ngram_vocab_size_divisible_by))
    split_parts = int(getattr(text, "split_ngram_parts", 512))
    rows_per_shard = cdiv(rows, split_parts)
    multipliers = Qwen4ExpNGramEmbedding._make_layer_multipliers(
        ngram_size=ngram_size,
        unigram_vocab_size=int(text.vocab_size),
        seed=int(getattr(text, "seed", 1234)),
        ple_dense_layer_id=0,
    )
    metadata = {
        "layer_multipliers": torch.tensor(multipliers, device="cpu"),
        "ngram_heads_vocab_sizes": torch.tensor(default_sizes, device="cpu"),
        "ngram_heads_offsets": torch.tensor(default_offsets, device="cpu"),
    }

    loader = DefaultModelLoader(config.load_config)
    _, files, use_safetensors, _ = loader._prepare_weights(
        model.model, None, model.revision, False, None
    )
    if not use_safetensors:
        raise ValueError("Scheduler-side PLE requires a safetensors checkpoint")
    layer = int(text.ple_layer_ids[0]) - 1
    prefix = f"layers.{layer}.ple.ple_embedding."
    read_table = partial(
        _load_table,
        files,
        prefix=prefix,
        rows=rows,
        rows_per_shard=rows_per_shard,
        head_dim=head_dim,
        metadata=metadata,
    )
    layout, dtype = read_table(materialize=False)
    spec = PLETableSpec(
        model.model,
        getattr(model.hf_config, "_commit_hash", None) or model.revision,
        prefix,
        rows,
        layout.shape[1],
        str(dtype),
    )
    backend = engram.prefill_ple_provider or "local"
    options = dict(engram.prefill_ple_provider_options)
    if backend == "local":
        options.setdefault("pool_bytes", engram.prefill_ple_pool_bytes or 512 * 1024**2)
    provider = create_provider(
        backend,
        table=spec,
        load_table=lambda: read_table()[0],
        options=options,
    )
    try:
        return PLEConnector(
            provider,
            partial(
                prompt_row_ids,
                eos=int(text.eos_token_id),
                multipliers=metadata["layer_multipliers"],
                sizes=metadata["ngram_heads_vocab_sizes"],
                offsets=metadata["ngram_heads_offsets"],
            ),
            num_heads=num_heads,
            row_bytes=spec.row_bytes,
            dtype=dtype,
            max_preparations=engram.prefill_ple_max_preparations,
            num_workers=config.parallel_config.world_size,
            timeout=engram.prefill_ple_timeout,
        )
    except BaseException:
        provider.close()
        raise


def _load_table(
    files: Sequence[str],
    *,
    prefix: str,
    rows: int,
    rows_per_shard: int,
    head_dim: int,
    metadata: dict[str, torch.Tensor],
    materialize: bool = True,
) -> tuple[torch.Tensor, torch.dtype]:
    """Load table bytes in shard order and apply checkpoint hash overrides."""
    seen: set[int] = set()
    seen_metadata: set[str] = set()
    table = None
    dtype = None
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as reader:
            for name in reader.keys():  # noqa: SIM118
                marker, separator, suffix = name.rpartition(prefix)
                if not separator or (marker and not marker.endswith(".")):
                    continue
                if suffix in metadata:
                    tensor = reader.get_tensor(name)
                    if (
                        suffix in seen_metadata
                        or tensor.shape != metadata[suffix].shape
                    ):
                        raise ValueError(
                            f"Invalid or duplicate PLE hash metadata {name}"
                        )
                    seen_metadata.add(suffix)
                    metadata[suffix] = tensor.to(dtype=torch.int64, copy=True)
                    continue
                if not (match := PLE_TABLE_SHARD.fullmatch(suffix)):
                    continue
                index = int(match[1])
                start = index * rows_per_shard
                size = min(rows_per_shard, rows - start)
                if index in seen or size <= 0:
                    raise ValueError(f"Invalid or duplicate PLE shard {name}")
                shard = reader.get_slice(name)
                shard_dtype = {
                    "BF16": torch.bfloat16,
                    "F8_E4M3": torch.float8_e4m3fn,
                }.get(shard.get_dtype())
                if tuple(shard.get_shape()) != (size, head_dim) or shard_dtype not in (
                    torch.bfloat16,
                    torch.float8_e4m3fn,
                ):
                    raise ValueError(f"Unsupported PLE shard shape or dtype for {name}")
                if table is None:
                    dtype = shard_dtype
                    row_bytes = head_dim * torch.empty((), dtype=dtype).element_size()
                    table = torch.empty(
                        (rows, row_bytes),
                        dtype=torch.uint8,
                        device="cpu" if materialize else "meta",
                    )
                if shard_dtype != dtype:
                    raise ValueError(f"Inconsistent PLE shard dtype for {name}")
                if materialize:
                    table[start : start + size].copy_(
                        reader.get_tensor(name).view(torch.uint8)
                    )
                seen.add(index)

    if table is None or dtype is None or len(seen) != cdiv(rows, rows_per_shard):
        raise ValueError("Scheduler-side PLE checkpoint is missing table shards")
    return table, dtype
