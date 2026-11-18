# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any, Self

from pydantic import Field, model_validator

from vllm.config.utils import config, get_hash_factors, hash_factors

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig
    from vllm.config.parallel import ParallelConfig

# Architecture -> the hf_text_config field naming its n-gram layers. A model is
# only configurable here if it actually has such layers to store.
_NGRAM_LAYER_FIELDS = {
    "DeepseekV41ForCausalLM": "engram_layer_ids",
    "Qwen4ExpForCausalLM": "ple_layer_ids",
    "Qwen4ExpForConditionalGeneration": "ple_layer_ids",
}


def model_has_engram_layers(model_config: "ModelConfig | None") -> bool:
    """Whether the model carries n-gram embedding layers."""
    if model_config is None:
        return False
    field = _NGRAM_LAYER_FIELDS.get(model_config.architecture)
    if field is None:
        return False
    return bool(getattr(model_config.hf_text_config, field, None))


@config
class EngramConfig:
    """Configuration for Engram embedding storage and sharding."""

    cpu_offload: bool = True
    """Store embedding weights in pinned CPU memory for UVA lookup."""

    disk_offload: bool = False
    """Keep Qwen4Exp PLE tables in the checkpoint files and read each step's
    rows on the host instead of allocating the table. Takes precedence over
    cpu_offload."""

    prefill_ple_provider: str | None = None
    """Scheduler-side PLE backend: 'local' or an importable provider factory."""

    prefill_ple_provider_options: dict[str, Any] = Field(default_factory=dict)
    """Provider-specific constructor options; never interpreted by Connector."""

    prefill_ple_pool_bytes: int | None = Field(default=None, gt=0)
    """LocalProvider pool capacity; also enables the local backend for compatibility.
    With an explicit local backend, defaults to 512 MiB. Not a Connector budget."""

    prefill_ple_max_preparations: int = Field(default=16, gt=0)
    """Maximum retained preparations, including ready and retiring results."""

    prefill_ple_timeout: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    """Seconds a request can wait for scheduler-side PLE rows."""

    embedding_across_dp: bool = False
    """Shard embeddings across TP and all DP ranks when enabled.
    Otherwise, each DP rank has a separate TP-sharded embedding replica."""

    dp_shared_memory: bool | None = None
    """Share CPU-offloaded embedding weights between co-located
    DP replicas. Each node stores one copy of every TP shard, reducing host
    memory without per-step Engram DP collectives. Uses host-NUMA cuMem shared
    by FABRIC handle where supported, else /dev/shm (which then needs capacity
    and a shared IPC namespace). Defaults to enabled whenever the other settings
    allow it, falling back to per-replica tables when DP replicas are not
    co-located on one node or the host cannot hold them."""

    use_thp: bool = False
    """Back private CPU-offloaded tables with transparent huge pages (best
    effort, falls back to ordinary pinned pages). Prefaulting the tables at
    startup takes longer. Requires cpu_offload without dp_shared_memory.
    Always on for private tables on ROCm."""

    @property
    def prefill_ple_enabled(self) -> bool:
        return (
            self.prefill_ple_provider is not None
            or self.prefill_ple_pool_bytes is not None
        )

    @model_validator(mode="after")
    def _validate_shared_memory(self) -> Self:
        if (
            self.prefill_ple_provider not in (None, "local")
            and self.prefill_ple_pool_bytes is not None
        ):
            raise ValueError("prefill_ple_pool_bytes only configures LocalProvider")
        if self.prefill_ple_enabled and (
            self.disk_offload
            or self.embedding_across_dp
            or self.dp_shared_memory
            or self.use_thp
        ):
            raise ValueError("Scheduler-side PLE requires its own table backend")
        if self.dp_shared_memory and not self.cpu_offload:
            raise ValueError("dp_shared_memory requires cpu_offload=True")
        if self.use_thp and (not self.cpu_offload or self.dp_shared_memory):
            raise ValueError(
                "use_thp requires cpu_offload=True and dp_shared_memory=False"
            )
        return self

    def verify_model_config(self, model_config: "ModelConfig | None") -> None:
        """Reject Engram configuration for models without n-gram embeddings."""
        field = (
            _NGRAM_LAYER_FIELDS.get(model_config.architecture)
            if model_config is not None
            else None
        )
        if (
            model_config is None
            or field is None
            or not getattr(model_config.hf_text_config, field, None)
        ):
            raise ValueError(
                "EngramConfig requires a model with supported Engram "
                "embeddings and non-empty n-gram layer ids."
            )
        if self.disk_offload:
            from vllm.platforms import current_platform

            if field != "ple_layer_ids" or not current_platform.is_cuda():
                raise ValueError("disk_offload requires a Qwen4Exp model on CUDA")
        if self.prefill_ple_enabled:
            from vllm.platforms import current_platform

            if (
                field != "ple_layer_ids"
                or not current_platform.is_cuda()
                or len(model_config.hf_text_config.ple_layer_ids) != 1
            ):
                raise ValueError(
                    "Scheduler-side PLE requires CUDA Qwen4Exp with one table"
                )

    def resolve_dp_shared_memory(self, parallel_config: "ParallelConfig") -> None:
        """Share host tables by default wherever the configuration permits."""
        if self.dp_shared_memory is None:
            self.dp_shared_memory = (
                self.cpu_offload
                and not self.use_thp
                and parallel_config.data_parallel_size > 1
                and not parallel_config.enable_elastic_ep
            )

    def verify_parallel_config(self, parallel_config: "ParallelConfig") -> None:
        """Reject unsupported embedding parallel topologies."""
        if self.prefill_ple_enabled and (
            parallel_config.nnodes != 1
            or parallel_config.data_parallel_size != 1
            or parallel_config.pipeline_parallel_size != 1
            or parallel_config.prefill_context_parallel_size != 1
            or parallel_config.decode_context_parallel_size != 1
            or parallel_config.enable_elastic_ep
            or parallel_config.distributed_executor_backend not in ("mp", "uni")
        ):
            raise ValueError(
                "Scheduler-side PLE requires single-node TP with mp or uni"
            )
        if self.dp_shared_memory:
            if parallel_config.data_parallel_size <= 1:
                raise ValueError("dp_shared_memory requires data_parallel_size > 1.")
            if parallel_config.enable_elastic_ep:
                raise ValueError("dp_shared_memory is not supported with elastic EP.")
        if (
            self.embedding_across_dp
            and parallel_config.data_parallel_size > 1
            and parallel_config.enable_elastic_ep
        ):
            raise ValueError(
                "Engram embedding_across_dp is not supported with elastic EP yet."
            )
        if self.disk_offload and parallel_config.enable_elastic_ep:
            raise ValueError("disk_offload is not supported with elastic EP.")

    def get_parallel_size(self, parallel_config: "ParallelConfig") -> int:
        """Derive the embedding group size from the parallel configuration."""
        size = parallel_config.tensor_parallel_size
        if self.embedding_across_dp and parallel_config.data_parallel_size > 1:
            size *= parallel_config.data_parallel_size
        return size

    def compute_hash(self) -> str:
        """Hash settings that affect embedding execution and graph structure."""
        return hash_factors(get_hash_factors(self, set()))
