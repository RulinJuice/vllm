# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from dataclasses import dataclass

PLE_TABLE_SHARD = re.compile(r"ngram_embedding\.shard_(\d+)\.weight$")


@dataclass(frozen=True)
class PLEResult:
    result_id: int
    start: int
    end: int
    offset: int


@dataclass(frozen=True)
class PLEPool:
    path: str
    size: int
    token_bytes: int
    dtype: str
