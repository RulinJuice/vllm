# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from functools import partial
from pathlib import Path
from threading import Event

import pytest
import torch
from safetensors.torch import save_file

from vllm.v1.ple.connector import PLEConnector
from vllm.v1.ple.provider import LocalPLEProvider
from vllm.v1.ple.qwen4 import _load_table, prompt_row_ids


def _make_connector(tmp_path, provider, pool_bytes=16, row_ids=None):
    def default_ids(tokens, start):
        return torch.tensor(tokens[start:], dtype=torch.int64)

    return PLEConnector(
        provider,
        default_ids if row_ids is None else row_ids,
        num_heads=1,
        row_bytes=2,
        dtype=torch.bfloat16,
        pool_bytes=pool_bytes,
        timeout=5,
        directory=str(tmp_path),
    )


def _wait_until(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "Timed out waiting for PLE"
        time.sleep(0.001)


@pytest.mark.cpu_test
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_provider_preserves_row_bytes_and_order(dtype):
    table = torch.arange(24, dtype=torch.float32).view(6, 4).to(dtype).view(torch.uint8)
    provider = LocalPLEProvider(table)
    out = torch.empty((3, table.shape[1]), dtype=torch.uint8)
    provider.lookup_into(torch.tensor([4, 0, 4]), out)
    assert torch.equal(out, table[[4, 0, 4]])
    with pytest.raises(IndexError, match="outside the table"):
        provider.lookup_into(torch.tensor([0, -1, 6]), out)
    with pytest.raises(ValueError, match="CPU int64"):
        provider.lookup_into(torch.tensor([0, 1, 2], dtype=torch.int32), out)


@pytest.mark.cpu_test
def test_checkpoint_preserves_shard_order_and_hash_overrides(tmp_path):
    prefix = "layers.0.ple.ple_embedding."
    shard_0 = tmp_path / "0.safetensors"
    shard_1 = tmp_path / "1.safetensors"
    rows = torch.arange(10, dtype=torch.float32).view(5, 2).to(torch.bfloat16)
    override = torch.tensor([17, 19])
    save_file(
        {
            f"model.{prefix}ngram_embedding.shard_0.weight": rows[:3],
            f"model.{prefix}layer_multipliers": override,
        },
        shard_0,
    )
    save_file(
        {f"model.{prefix}ngram_embedding.shard_1.weight": rows[3:]},
        shard_1,
    )
    metadata = {"layer_multipliers": torch.tensor([3, 5])}
    read = partial(_load_table, prefix=prefix, rows=5, rows_per_shard=3, head_dim=2)
    table, dtype = read([str(shard_1), str(shard_0)], metadata=metadata)
    assert dtype == rows.dtype
    assert torch.equal(table, rows.view(torch.uint8))
    assert torch.equal(metadata["layer_multipliers"], override)
    with pytest.raises(ValueError, match="missing table shards"):
        read([str(shard_0)], metadata=metadata)


@pytest.mark.cpu_test
def test_hash_suffix_preserves_left_context_and_eos():
    eos = 3
    multipliers = [5, -7, 11, -(2**60) + 1]
    sizes = [7, 13, 17, 23, 29, 31]
    offsets = [0, 7, 20, 37, 60, 89]
    tokens = [18, 22, 75, eos, 31, 34, eos, eos, 69, 89, 44, 19]
    row_ids = partial(
        prompt_row_ids,
        eos=eos,
        multipliers=torch.tensor(multipliers),
        sizes=torch.tensor(sizes),
        offsets=torch.tensor(offsets),
    )

    expected = []
    for position, token in enumerate(tokens):
        context = [token]
        for shift in range(1, len(multipliers)):
            older = tokens[position - shift] if position >= shift else eos
            context.append(eos if eos in context[1:] else older)
        mixed = 0
        for i, (value, multiplier) in enumerate(zip(context, multipliers)):
            mixed ^= value * multiplier
            signed = ((mixed + 2**63) % 2**64) - 2**63
            if i:
                for head in range(2 * (i - 1), 2 * i):
                    expected.append(signed % sizes[head] + offsets[head])

    for start in range(len(tokens)):
        assert row_ids(tokens, start).tolist() == expected[start * len(sizes) :]


@pytest.mark.cpu_test
def test_connector_rechecks_coverage_and_waits_for_ack(tmp_path):
    table = torch.arange(16, dtype=torch.uint8).view(8, 2)
    connector = _make_connector(tmp_path, LocalPLEProvider(table), pool_bytes=8)
    tokens = [5, 6, 1, 7]
    try:
        assert not connector.ensure_ready("a", tokens, 2)
        _wait_until(lambda: connector.ensure_ready("a", tokens, 3))
        result = connector.dispatch("a")
        assert (result.start, result.end) == (2, 4)
        data = Path(connector.pool.path).read_bytes()
        assert data[result.offset : result.offset + 4] == bytes([2, 3, 14, 15])

        # APC moved back after preemption. The old result still occupies space.
        assert not connector.ensure_ready("a", tokens, 0)
        assert connector.cleanup_ids() == [result.result_id]
        connector.reclaim([result.result_id])
        assert not connector.ensure_ready("a", tokens, 0)
        _wait_until(lambda: connector.ensure_ready("a", tokens, 0))
        replacement = connector.dispatch("a")
        assert replacement.start == 0
        assert replacement.result_id != result.result_id
        assert not connector.ensure_ready("b", tokens, 0)
        connector.finish("a")
        assert connector.cleanup_ids() == [replacement.result_id]
        connector.reclaim([replacement.result_id])
        _wait_until(lambda: connector.ensure_ready("b", tokens, 0))
        assert connector.dispatch("b").offset == replacement.offset
    finally:
        connector.close()
    assert not Path(connector.pool.path).exists()


@pytest.mark.cpu_test
def test_abort_does_not_reuse_bytes_while_provider_is_writing(tmp_path):
    started = Event()
    complete = Event()

    class BlockingProvider:
        def lookup_into(self, row_ids, out):
            started.set()
            assert complete.wait(5)
            out.fill_(123)

    connector = _make_connector(tmp_path, BlockingProvider(), pool_bytes=4)
    try:
        assert not connector.ensure_ready("aborted", [1, 2], 0)
        assert started.wait(5)
        connector.finish("aborted")
        assert connector.cleanup_ids() == []
        assert connector.has_cleanup
        assert not connector.ensure_ready("next", [4, 5], 0)

        complete.set()

        def retired():
            assert connector.cleanup_ids() == []
            return not connector.has_cleanup

        _wait_until(retired)
        _wait_until(lambda: connector.ensure_ready("next", [4, 5], 0))
        assert connector.dispatch("next").offset == 0
    finally:
        complete.set()
        connector.close()


@pytest.mark.cpu_test
@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize(
    ("remote_hit", "prompt_len", "first_compute_token"),
    [(0, 27, 16), (16, 27, 16), (32, 32, 31)],
)
def test_ple_waits_for_kv_and_only_prepares_the_suffix(
    tmp_path, async_scheduling, remote_hit, prompt_len, first_compute_token
):
    from .test_scheduler import _step_until_kv_transfer_finished, make_output
    from .utils import create_requests, create_scheduler, mock_kv

    scheduler = create_scheduler(
        async_scheduling=async_scheduling,
        enable_prefix_caching=True,
        use_v2_model_runner=True,
        use_kv_connector=(
            mock_kv(matched_tokens=remote_hit, is_async=True) if remote_hit else None
        ),
    )
    warmup, request = create_requests(
        num_requests=2, num_tokens=prompt_len, same_prompt=True, max_tokens=1
    )
    if not remote_hit:
        scheduler.add_request(warmup)
        output = scheduler.schedule()
        scheduler.update_from_output(output, make_output(scheduler))
        assert warmup.is_finished()

    prepared = []

    def row_ids(tokens, start):
        prepared.append((start, len(tokens)))
        return torch.arange(start, len(tokens))

    table = torch.arange(2 * prompt_len, dtype=torch.uint8).view(prompt_len, 2)
    provider = LocalPLEProvider(table)
    connector = _make_connector(tmp_path, provider, table.nbytes, row_ids)
    scheduler.ple_connector = connector
    try:
        scheduler.add_request(request)
        if remote_hit:
            _step_until_kv_transfer_finished(scheduler, [request.request_id])
            assert prepared == []

        output = scheduler.schedule()
        assert not output.num_scheduled_tokens

        def admitted():
            nonlocal output
            output = scheduler.schedule()
            return bool(output.num_scheduled_tokens)

        _wait_until(admitted)
        expected = (first_compute_token, prompt_len)
        assert prepared == [expected]
        assert output.num_scheduled_tokens == {
            request.request_id: prompt_len - first_compute_token
        }
        result = output.scheduled_new_reqs[0].ple_result
        assert result is not None and (result.start, result.end) == expected
        scheduler.update_from_output(output, make_output(scheduler))
        assert connector.cleanup_ids() == [result.result_id]
    finally:
        scheduler.shutdown()


@pytest.mark.skipif(
    torch.version.cuda is None or not torch.cuda.is_available(), reason="CUDA required"
)
@pytest.mark.parametrize("use_graph", [False, True])
def test_reader_stages_packed_chunks_and_padding(tmp_path, use_graph):
    from vllm.v1.ple.reader import PLEReader
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

    table = torch.arange(16, dtype=torch.uint8).view(8, 2)
    connector = _make_connector(tmp_path, LocalPLEProvider(table))
    rows = torch.zeros((4, 1), device="cuda", dtype=torch.bfloat16)
    ready = torch.cuda.Event(external=True)
    ready.record()
    reader = None
    try:
        reader = PLEReader(connector.pool, rows, ready)
        results = []
        for req, tokens, start in (("a", [0, 1, 2, 3], 1), ("b", [4, 5, 6], 0)):
            _wait_until(partial(connector.ensure_ready, req, tokens, start))
            results.append(connector.dispatch(req))
            reader.attach(req, results[-1])

        output = torch.empty_like(rows)

        def consume():
            torch.cuda.current_stream().wait_event(ready)
            output.copy_(rows)

        if use_graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                consume()
            consume = graph.replay

        batch = InputBatch.make_dummy(2, 2, InputBuffers(2, 4, rows.device))
        batch.req_ids = ["b", "a"]
        batch.num_tokens_after_padding = 4
        for start_a, start_b in ((2, 1), (3, 2)):
            batch.num_computed_prefill_tokens_np[:] = [start_b, start_a]
            reader.stage(batch)
            consume()
            reader.finish_forward()
            expected = torch.zeros_like(table[:4])
            expected[:2] = table[[4 + start_b, start_a]]
            assert torch.equal(output.cpu().view(torch.uint8), expected)

        for req in ("a", "b"):
            reader.remove_request(req)
            connector.finish(req)
        ids = connector.cleanup_ids()
        assert ids == [result.result_id for result in results]
        connector.reclaim(reader.acknowledge(ids))
        assert not connector.has_cleanup
    finally:
        if reader is not None:
            reader.close()
        connector.close()
