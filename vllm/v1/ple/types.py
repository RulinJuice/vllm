# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any, Literal

import regex as re

PLE_TABLE_SHARD = re.compile(r"ngram_embedding\.shard_(\d+)\.weight$")


@dataclass(frozen=True)
class PLEChunk:
    """An ordered slice of valid bytes in a stable shared pool."""

    offset: int
    nbytes: int


@dataclass(frozen=True)
class PoolDescriptor:
    """One engine-lifetime pool, exported as a path or one FD handle per rank.

    pool_id identifies the pool generation. Handles must support detach() in
    the receiving process (e.g. multiprocessing.reduction.DupFd).
    """

    pool_id: str
    size: int
    path: str | None = None
    handles: tuple[Any, ...] = ()
    registration: Literal["pool", "region"] = "pool"

    def __post_init__(self) -> None:
        if not self.pool_id or self.size <= 0:
            raise ValueError("PLE pool requires an identity and positive size")
        if (self.path is not None) == bool(self.handles):
            raise ValueError("PLE pool requires either a path or per-rank FD handles")
        if self.registration not in ("pool", "region"):
            raise ValueError("Unsupported PLE registration policy")


@dataclass(frozen=True)
class PreparedRows:
    result_id: str
    pool_id: str
    chunks: tuple[PLEChunk, ...]


@dataclass(frozen=True)
class PLEResult:
    """Connector-owned token binding; providers only produce PreparedRows."""

    start: int
    end: int
    rows: PreparedRows


@dataclass(frozen=True)
class PLEReaderConfig:
    pool: PoolDescriptor
    token_bytes: int
    dtype: str


@dataclass(frozen=True)
class PLETableSpec:
    model: str
    revision: str | None
    name: str
    num_rows: int
    row_bytes: int
    dtype: str
