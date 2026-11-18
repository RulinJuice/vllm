# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.outputs import ModelRunnerOutput


class PLEOutputAggregator:
    """A result can retire only after every consuming rank stops using it."""

    def __init__(self, num_workers: int) -> None:
        self.num_workers = num_workers

    def aggregate(
        self, outputs: list[ModelRunnerOutput | None], output_rank: int = 0
    ) -> ModelRunnerOutput | None:
        output = outputs[output_rank]
        if output is None:
            return None
        consumed = set()
        if len(outputs) == self.num_workers and all(o is not None for o in outputs):
            consumed = set.intersection(
                *(o.ple_consumed_result_ids for o in outputs if o is not None)
            )
        return ModelRunnerOutput.with_ple_consumed(output, consumed)
