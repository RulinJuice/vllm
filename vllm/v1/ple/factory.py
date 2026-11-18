# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any

import torch

from vllm.utils.import_utils import resolve_obj_by_qualname

from .local import LocalPLEProvider
from .provider import PLEProvider
from .types import PLETableSpec


def create_provider(
    backend: str,
    *,
    table: PLETableSpec,
    load_table: Callable[[], torch.Tensor],
    options: dict[str, Any],
) -> PLEProvider:
    """Construct a built-in provider or an importable provider factory.

    Factories accept the same three keyword arguments. Remote backends need
    not call load_table, so table bytes are never materialized by common code.
    """
    if backend == "local":
        return LocalPLEProvider(load_table(), **options)
    factory = resolve_obj_by_qualname(backend)
    return factory(table=table, load_table=load_table, options=dict(options))
