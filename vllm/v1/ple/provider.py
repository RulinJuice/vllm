# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Protocol

import torch


class PLEProvider(Protocol):
    """Read rows of a single bound table into caller-owned CPU storage."""

    def lookup_into(self, row_ids: torch.Tensor, out: torch.Tensor) -> None:
        """Fill out in ID order and finish all writes before returning or raising."""
        ...


class LocalPLEProvider:
    """Gather raw checkpoint rows from a CPU-resident byte table."""

    def __init__(self, table: torch.Tensor) -> None:
        if (
            table.device.type != "cpu"
            or table.dtype != torch.uint8
            or table.ndim != 2
            or not table.is_contiguous()
            or not table.numel()
        ):
            raise ValueError("PLE table must be a nonempty contiguous CPU byte matrix")
        self._table = table
        self.row_bytes = table.shape[1]

    def lookup_into(self, row_ids: torch.Tensor, out: torch.Tensor) -> None:
        if (
            row_ids.device.type != "cpu"
            or row_ids.dtype != torch.int64
            or row_ids.ndim != 1
            or not row_ids.is_contiguous()
        ):
            raise ValueError("PLE row IDs must be a contiguous CPU int64 vector")
        if (
            out.device.type != "cpu"
            or out.dtype != torch.uint8
            or out.shape != (row_ids.numel(), self.row_bytes)
            or not out.is_contiguous()
        ):
            raise ValueError("PLE output must be a contiguous CPU byte matrix of rows")
        if row_ids.numel():
            lo, hi = torch.aminmax(row_ids)
            if lo.item() < 0 or hi.item() >= self._table.shape[0]:
                raise IndexError("PLE row ID is outside the table")
        torch.index_select(self._table, 0, row_ids, out=out)
