# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from concurrent.futures import Future
from functools import partial
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Event

import pytest
import torch
from safetensors.torch import save_file

from vllm.v1.ple.connector import PLEConnector
from vllm.v1.ple.local import LocalPLEProvider
from vllm.v1.ple.qwen4 import _load_table, prompt_row_ids
from vllm.v1.ple.types import PreparedRows


def _make_connector(tmp_path, table, pool_bytes=16, row_ids=None, **kwargs):
    def default_ids(tokens, start):
        return torch.tensor(tokens[start:], dtype=torch.int64)

    return PLEConnector(
        LocalPLEProvider(table, pool_bytes=pool_bytes, directory=str(tmp_path))
        if isinstance(table, torch.Tensor)
        else table,
        default_ids if row_ids is None else row_ids,
        num_heads=1,
        row_bytes=2,
        dtype=torch.bfloat16,
        max_preparations=kwargs.pop("max_preparations", 16),
        timeout=kwargs.pop("timeout", 5),
        **kwargs,
    )


def _wait_until(predicate):
    deadline = time.monotonic() + 5
    while not (value := predicate()):
        assert time.monotonic() < deadline, "Timed out waiting for PLE"
        time.sleep(0.001)
    return value


@pytest.mark.cpu_test
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_provider_preserves_row_bytes_and_order(tmp_path, dtype):
    table = torch.arange(24, dtype=torch.float32).view(6, 4).to(dtype).view(torch.uint8)
    provider = LocalPLEProvider(table, pool_bytes=table.nbytes, directory=str(tmp_path))
    try:
        result = provider.prepare("ordered", torch.tensor([4, 0, 4])).result(5)
        data = Path(provider.export_pool(1).path).read_bytes()
        actual = b"".join(data[c.offset : c.offset + c.nbytes] for c in result.chunks)
        assert actual == bytes(table[[4, 0, 4]].flatten().tolist())
        with pytest.raises(RuntimeError, match="outside the table"):
            provider.prepare("bad-index", torch.tensor([0, -1, 6])).result(5)
        with pytest.raises(RuntimeError, match="CPU int64"):
            provider.prepare(
                "bad-dtype", torch.tensor([0, 1, 2], dtype=torch.int32)
            ).result(5)
        for key in ("ordered", "bad-index", "bad-dtype"):
            provider.release(key).result(5)
    finally:
        provider.close()


@pytest.mark.cpu_test
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_checkpoint_preserves_shard_order_and_hash_overrides(tmp_path, dtype):
    prefix = "layers.0.ple.ple_embedding."
    shard_0 = tmp_path / "0.safetensors"
    shard_1 = tmp_path / "1.safetensors"
    rows = torch.arange(10, dtype=torch.float32).view(5, 2).to(dtype)
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
    layout, layout_dtype = read(
        [str(shard_1), str(shard_0)], metadata=metadata, materialize=False
    )
    assert layout.device.type == "meta" and layout.shape == table.shape
    assert layout_dtype == dtype
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
    connector = _make_connector(tmp_path, table, pool_bytes=8)
    tokens = [5, 6, 1, 7]
    try:
        connector.ensure_ready("a", tokens, 2)
        _wait_until(lambda: connector.ensure_ready("a", tokens, 3))
        result = connector.dispatch("a")
        assert (result.start, result.end) == (2, 4)
        data = Path(connector.reader_config.pool.path).read_bytes()
        assert data[
            result.rows.chunks[0].offset : result.rows.chunks[0].offset + 4
        ] == bytes([2, 3, 14, 15])

        # APC moved back after preemption. The old result still occupies space.
        assert not connector.ensure_ready("a", tokens, 0)
        assert connector.retire_ids() == {result.rows.result_id}
        connector.update_from_output({result.rows.result_id})
        _wait_until(lambda: connector.ensure_ready("a", tokens, 0))
        replacement = connector.dispatch("a")
        assert replacement.start == 0
        assert replacement.rows.result_id != result.rows.result_id
        assert not connector.ensure_ready("b", tokens, 0)
        connector.finish("a")
        assert connector.retire_ids() == {replacement.rows.result_id}
        connector.update_from_output({replacement.rows.result_id})
        _wait_until(lambda: connector.ensure_ready("b", tokens, 0))
        assert (
            connector.dispatch("b").rows.chunks[0].offset
            == replacement.rows.chunks[0].offset
        )
    finally:
        connector.close(consumers_stopped=True)
    assert not Path(connector.reader_config.pool.path).exists()


@pytest.mark.cpu_test
@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("blocked_by", ["tokens", "slots"])
@pytest.mark.parametrize("remote_hit", [0, 16])
def test_ple_prepares_waiting_prompt_without_compute_capacity(
    tmp_path, async_scheduling, blocked_by, remote_hit
):
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import RequestStatus

    from .test_scheduler import _step_until_kv_transfer_finished, make_output
    from .utils import create_requests, create_scheduler, mock_kv

    scheduler = create_scheduler(
        async_scheduling=async_scheduling,
        max_num_seqs=1 if blocked_by == "slots" else 16,
        max_num_batched_tokens=16,
        max_model_len=128,
        long_prefill_token_threshold=8 if blocked_by == "slots" else 0,
        use_v2_model_runner=True,
        use_kv_connector=mock_kv(remote_hit, is_async=True) if remote_hit else None,
    )
    running, waiting = create_requests(num_requests=2, num_tokens=64, max_tokens=1)
    prepared = Event()

    def row_ids(tokens, start):
        if tokens[0] == waiting.prompt_token_ids[0]:
            prepared.set()
        return torch.arange(start, len(tokens))

    table = torch.arange(128, dtype=torch.uint8).view(64, 2)
    connector = _make_connector(tmp_path, table, 256, row_ids)
    scheduler.ple_connector = connector
    try:
        scheduler.add_request(running)
        if remote_hit:
            _step_until_kv_transfer_finished(scheduler, [running.request_id])
        output = None

        def admitted():
            nonlocal output
            output = scheduler.schedule()
            return bool(output.num_scheduled_tokens)

        _wait_until(admitted)
        partial = make_output(scheduler)
        partial.sampled_token_ids = [[]]
        scheduler.update_from_output(output, partial)

        scheduler.add_request(waiting)
        output = scheduler.schedule()
        if remote_hit:
            assert waiting.status == RequestStatus.WAITING_FOR_REMOTE_KVS
            assert prepared.wait(5), "PLE should start before the KV receive finishes"
            assert waiting.request_id not in output.ple_results
            partial.kv_connector_output = KVConnectorOutput(
                finished_recving={waiting.request_id}
            )
            scheduler.update_from_output(output, partial)
            output = scheduler.schedule()
            partial.kv_connector_output = None
        assert output.num_scheduled_tokens == {
            running.request_id: 8 if blocked_by == "slots" else 16
        }
        assert prepared.wait(5), "PLE should start while computation is full"
        assert waiting.num_computed_tokens == remote_hit
        assert all(
            data.req_id != waiting.request_id for data in output.scheduled_new_reqs
        )

        scheduler.update_from_output(output, partial)
        scheduler.finish_requests([running.request_id], RequestStatus.FINISHED_ABORTED)
        _wait_until(admitted)
        result = output.ple_results[output.scheduled_new_reqs[0].req_id]
        assert result is not None and (result.start, result.end) == (remote_hit, 64)
    finally:
        scheduler.shutdown()


@pytest.mark.cpu_test
@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize(
    ("remote_hit", "prompt_len", "first_compute_token"),
    [(0, 27, 16), (16, 27, 16), (32, 32, 31), (32, 40, 16)],
)
def test_ple_prefetches_only_the_suffix_and_gates_compute_on_kv(
    tmp_path, async_scheduling, remote_hit, prompt_len, first_compute_token
):
    from dataclasses import replace

    from vllm.config import KVTransferConfig
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput

    from .test_scheduler import make_output
    from .utils import create_requests, create_scheduler

    scheduler = create_scheduler(
        async_scheduling=async_scheduling,
        enable_prefix_caching=True,
        # A full hit replays one real token even when drafts are configured.
        num_speculative_tokens=2 if remote_hit == prompt_len else None,
        use_v2_model_runner=True,
        use_kv_connector=(
            KVTransferConfig(
                kv_connector="MockKVConnector",
                kv_role="kv_both",
                kv_load_failure_policy="recompute",
                kv_connector_extra_config={
                    "matched_tokens": remote_hit,
                    "is_async": True,
                },
            )
            if remote_hit
            else None
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
    connector = _make_connector(tmp_path, table, table.nbytes, row_ids)
    scheduler.ple_connector = connector
    try:
        scheduler.add_request(request)
        if remote_hit:
            initial = scheduler.schedule()
            assert initial.total_num_scheduled_tokens == 0 and not initial.ple_results
            _wait_until(lambda: bool(prepared))
            scheduler.update_from_output(initial, EMPTY_MODEL_RUNNER_OUTPUT)
            waiting = scheduler.schedule()
            assert waiting.total_num_scheduled_tokens == 0 and not waiting.ple_results
            predicted = min(remote_hit, prompt_len - 1)
            assert prepared == [(predicted, prompt_len)]
            invalid_blocks = set()
            if first_compute_token < predicted:
                blocks = scheduler.kv_cache_manager.get_block_ids(request.request_id)[0]
                invalid_blocks.add(blocks[first_compute_token // scheduler.block_size])
            scheduler.update_from_output(
                waiting,
                replace(
                    EMPTY_MODEL_RUNNER_OUTPUT,
                    kv_connector_output=KVConnectorOutput(
                        finished_recving={request.request_id},
                        invalid_block_ids=invalid_blocks,
                    ),
                ),
            )

        def admitted():
            output = scheduler.schedule()
            return output if output.num_scheduled_tokens else None

        output = _wait_until(admitted)
        expected = (first_compute_token, prompt_len)
        if remote_hit and first_compute_token < min(remote_hit, prompt_len - 1):
            assert prepared == [(min(remote_hit, prompt_len - 1), prompt_len), expected]
        else:
            assert prepared == [expected]
        assert not output.scheduled_spec_decode_tokens
        assert output.num_scheduled_tokens == {
            request.request_id: prompt_len - first_compute_token
        }
        result = output.ple_results[output.scheduled_new_reqs[0].req_id]
        assert result is not None and (result.start, result.end) == expected
        scheduler.update_from_output(output, make_output(scheduler))
        assert connector.retire_ids() == {result.rows.result_id}
    finally:
        scheduler.shutdown()


@pytest.fixture
def ple_shm_dir():
    # CUDA host registration of file-backed pages depends on the filesystem.
    with TemporaryDirectory(dir="/dev/shm") as directory:
        yield Path(directory)


@pytest.mark.skipif(
    torch.version.cuda is None or not torch.cuda.is_available(), reason="CUDA required"
)
@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("registration", ["pool", "region"])
@pytest.mark.parametrize("transport", ["path", "fd"])
def test_reader_stages_packed_chunks_and_padding(
    ple_shm_dir, use_graph, registration, transport
):
    import os
    from dataclasses import replace
    from multiprocessing.reduction import DupFd

    from vllm.v1.ple.reader import PLEReader
    from vllm.v1.ple.types import PLEChunk
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

    table = torch.arange(16, dtype=torch.uint8).view(8, 2)
    connector = _make_connector(ple_shm_dir, table)
    rows = torch.zeros((4, 1), device="cuda", dtype=torch.bfloat16)
    ready = torch.cuda.Event(external=True)
    ready.record()
    reader = None
    try:
        pool = replace(connector.reader_config.pool, registration=registration)
        if transport == "fd":
            fd = os.open(pool.path, os.O_RDWR)
            try:
                pool = replace(pool, path=None, handles=(DupFd(fd),))
            finally:
                os.close(fd)
        reader = PLEReader(replace(connector.reader_config, pool=pool), rows, ready)
        results = {}
        for req, tokens, start in (("a", [0, 1, 2, 3], 1), ("b", [4, 5, 6], 0)):
            _wait_until(partial(connector.ensure_ready, req, tokens, start))
            result = connector.dispatch(req)
            chunk = result.rows.chunks[0]
            # A provider may fragment at any byte, including inside a BF16 row.
            results[req] = replace(
                result,
                rows=replace(
                    result.rows,
                    chunks=(
                        PLEChunk(chunk.offset, 1),
                        PLEChunk(chunk.offset + 1, chunk.nbytes - 1),
                    ),
                ),
            )

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
            reader.stage(batch, results)
            consume()
            reader.finish_forward()
            expected = torch.zeros_like(table[:4])
            expected[:2] = table[[4 + start_b, start_a]]
            assert torch.equal(output.cpu().view(torch.uint8), expected)

        connector.finish("a")
        connector.update_from_output(reader.release(connector.retire_ids()))
        # Both results share one page; retiring a must not unregister b's bytes.
        batch.req_ids = ["b", "b"]
        batch.num_computed_prefill_tokens_np[:] = [0, 1]
        reader.stage(batch, results)
        consume()
        reader.finish_forward()
        expected[:2] = table[[4, 5]]
        assert torch.equal(output.cpu().view(torch.uint8), expected)
        connector.finish("b")
        ids = connector.retire_ids()
        assert ids == {results["b"].rows.result_id}
        connector.update_from_output(reader.release(ids))
        assert not connector.has_cleanup
    finally:
        if reader is not None:
            reader.close()
        connector.close(consumers_stopped=True)


class DeferredRemoteProvider:
    """Test adapter for a controllable external service using real local storage."""

    def __init__(self, tmp_path):
        self.storage = LocalPLEProvider(
            torch.arange(32, dtype=torch.uint8).view(16, 2),
            pool_bytes=32,
            directory=str(tmp_path),
        )
        self.submissions: Queue[tuple[str, torch.Tensor, Future[PreparedRows]]] = (
            Queue()
        )
        self.released = []
        self.producing = {}
        self.release_error = None

    def export_pool(self, num_workers):
        return self.storage.export_pool(num_workers)

    def prepare(self, result_id, row_ids):
        future: Future[PreparedRows] = Future()
        self.producing[result_id] = future
        self.submissions.put((result_id, row_ids, future))
        return future

    def succeed(self, submission):
        result_id, row_ids, future = submission
        future.set_result(self.storage.prepare(result_id, row_ids).result(5))

    def release(self, result_id):
        if self.release_error is not None:
            error, self.release_error = self.release_error, None
            failed: Future[None] = Future()
            failed.set_exception(error)
            return failed
        released: Future[None] = Future()

        def release_rows(_):
            self.storage.release(result_id).result()
            self.released.append(result_id)
            released.set_result(None)

        self.producing.pop(result_id).add_done_callback(release_rows)
        return released

    def close(self):
        self.storage.close()


@pytest.mark.cpu_test
@pytest.mark.parametrize("failure", ["cancel", "sdk", "descriptor"])
def test_background_prepare_and_cleanup_do_not_wait_for_scheduler(tmp_path, failure):
    from dataclasses import replace

    provider = DeferredRemoteProvider(tmp_path)
    connector = _make_connector(tmp_path, provider)
    first = second = None
    try:
        connector.ensure_ready("first", [1, 2], 0)
        first = provider.submissions.get(timeout=5)
        # Remote I/O for the first request must not occupy the hash thread.
        connector.ensure_ready("second", [3, 4], 0)
        second = provider.submissions.get(timeout=5)
        assert first[1].tolist() == [1, 2] and second[1].tolist() == [3, 4]
        if failure == "cancel":
            connector.finish("first")
            assert connector.has_cleanup and not first[2].cancelled()
            provider.succeed(first)
        elif failure == "sdk":
            # No result was delivered. SDK drain is intentionally never signalled.
            first[2].set_exception(RuntimeError("SDK fetch failed"))
            with pytest.raises(RuntimeError, match="SDK fetch failed"):
                _wait_until(lambda: connector.ensure_ready("first", [1, 2], 0))
            connector.finish("first")
        else:
            rows = provider.storage.prepare(first[0], first[1]).result(5)
            first[2].set_result(replace(rows, pool_id="stale-generation"))
            with pytest.raises(RuntimeError, match="invalid PLE result"):
                _wait_until(lambda: connector.ensure_ready("first", [1, 2], 0))
            provider.release_error = RuntimeError("temporary READ release failure")
            connector.finish("first")
            assert connector.has_cleanup
        provider.succeed(second)
        _wait_until(lambda: connector.ensure_ready("second", [3, 4], 0))
        _wait_until(lambda: (connector.poll(), not connector.has_cleanup)[1])
        assert first[0] in provider.released
    finally:
        for submission in (first, second):
            if submission is not None and not submission[2].done():
                submission[2].set_exception(RuntimeError("test ended"))
        connector.close()


@pytest.mark.cpu_test
def test_cancel_during_hash_skips_provider_submission(tmp_path):
    started, proceed = Event(), Event()
    provider = DeferredRemoteProvider(tmp_path)

    def hash_ids(tokens, start):
        started.set()
        assert proceed.wait(5)
        return torch.tensor(tokens[start:])

    connector = _make_connector(
        tmp_path, provider, row_ids=hash_ids, max_preparations=1
    )
    try:
        connector.ensure_ready("first", [1, 2], 0)
        assert started.wait(5)
        connector.finish("first")
        assert not connector.ensure_ready("next", [3, 4], 0)
        proceed.set()
        _wait_until(lambda: (connector.poll(), not connector.has_cleanup)[1])
        assert provider.submissions.empty()
        connector.ensure_ready("next", [3, 4], 0)
        next_lookup = provider.submissions.get(timeout=5)
        assert next_lookup[1].tolist() == [3, 4]
        provider.succeed(next_lookup)
        _wait_until(lambda: connector.ensure_ready("next", [3, 4], 0))
    finally:
        proceed.set()
        connector.close()


@pytest.mark.cpu_test
def test_cancel_pool_waiter_allows_a_smaller_request_to_run(tmp_path):
    # Keep A live and unacknowledged. Only C fits in the remaining half-pool.
    table = torch.arange(16, dtype=torch.uint8).view(8, 2)
    c_hashed = Event()

    def row_ids(tokens, start):
        if tokens[0] == 6:
            c_hashed.set()
        return torch.tensor(tokens[start:])

    connector = _make_connector(tmp_path, table, 8, row_ids, timeout=1)
    try:
        _wait_until(lambda: connector.ensure_ready("a", [0, 1], 0))
        connector.dispatch("a")
        connector.ensure_ready("b", [2, 3, 4, 5], 0)
        connector.ensure_ready("c", [6], 0)
        assert c_hashed.wait(5)  # B's provider submission has returned.
        connector.finish("b")
        _wait_until(lambda: connector.ensure_ready("c", [6], 0))
        assert connector.dispatch("c").rows.chunks[0].nbytes == 2
    finally:
        connector.close(consumers_stopped=True)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "error", [TimeoutError("read deadline"), asyncio.CancelledError()]
)
@pytest.mark.parametrize("synchronous", [False, True])
def test_sdk_error_reaches_client_on_zero_token_step(tmp_path, error, synchronous):
    from vllm.v1.engine import FinishReason
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

    from .utils import create_requests, create_scheduler

    scheduler = create_scheduler(max_num_seqs=1, use_v2_model_runner=True)

    class FailingProvider(DeferredRemoteProvider):
        fail_first = synchronous

        def prepare(self, result_id, row_ids):
            future = super().prepare(result_id, row_ids)
            if self.fail_first:
                self.fail_first = False
                future.set_exception(error)
                raise error
            return future

    provider = FailingProvider(tmp_path)
    connector = _make_connector(tmp_path, provider, max_preparations=1)
    scheduler.ple_connector = connector
    (request,) = create_requests(num_requests=1, num_tokens=4, max_tokens=1)
    try:
        scheduler.add_request(request)
        step = scheduler.schedule()
        lookup = provider.submissions.get(timeout=5)
        scheduler.update_from_output(step, EMPTY_MODEL_RUNNER_OUTPUT)
        if not synchronous:
            lookup[2].set_exception(error)

        def reply_without_forward():
            step = scheduler.schedule()
            assert step.total_num_scheduled_tokens == 0
            outputs = scheduler.update_from_output(step, EMPTY_MODEL_RUNNER_OUTPUT)
            return (
                outputs[request.client_index].outputs
                if request.client_index in outputs
                else None
            )

        (reply,) = _wait_until(reply_without_forward)
        assert reply.finish_reason == FinishReason.ERROR and reply.new_token_ids == []
        assert reply.error.code == "ple_lookup_failed"
        assert reply.error.retryable == isinstance(error, TimeoutError)
        _wait_until(lambda: (connector.poll(), not connector.has_cleanup)[1])
        connector.ensure_ready("next", [1], 0)
        next_lookup = provider.submissions.get(timeout=5)
        provider.succeed(next_lookup)
        _wait_until(lambda: connector.ensure_ready("next", [1], 0))
    finally:
        scheduler.shutdown()


@pytest.mark.cpu_test
def test_local_provider_handles_fragmentation_and_odd_pool_size(tmp_path):
    table = torch.arange(16, dtype=torch.uint8).view(8, 2)
    provider = LocalPLEProvider(table, pool_bytes=9, directory=str(tmp_path))
    try:
        a = provider.prepare("a", torch.tensor([0])).result(5)
        provider.prepare("b", torch.tensor([1])).result(5)
        provider.prepare("c", torch.tensor([2])).result(5)
        provider.release("a").result(5)
        provider.release("c").result(5)
        d = provider.prepare("d", torch.tensor([7, 3, 7])).result(5)
        assert len(d.chunks) == 2 and d.chunks[0] == a.chunks[0]
        data = Path(provider.export_pool(1).path).read_bytes()
        assert b"".join(
            data[c.offset : c.offset + c.nbytes] for c in d.chunks
        ) == bytes(table[[7, 3, 7]].flatten().tolist())
    finally:
        provider.close()


@pytest.mark.cpu_test
def test_consumption_requires_every_rank_and_does_not_mutate_empty_output():
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
    from vllm.v1.ple.output import PLEOutputAggregator

    empty = EMPTY_MODEL_RUNNER_OUTPUT
    rank0 = ModelRunnerOutput.with_ple_consumed(empty, {"old", "new"})
    rank1 = ModelRunnerOutput.with_ple_consumed(empty, {"old"})
    aggregator = PLEOutputAggregator(2)
    assert aggregator.aggregate([rank0, rank1]).ple_consumed_result_ids == {"old"}
    assert not aggregator.aggregate([rank0, empty]).ple_consumed_result_ids
    assert not aggregator.aggregate([rank0, None]).ple_consumed_result_ids
    assert not aggregator.aggregate([rank0]).ple_consumed_result_ids
    assert not empty.ple_consumed_result_ids
    assert rank0.ple_consumed_result_ids == {"old", "new"}


@pytest.mark.cpu_test
def test_timeout_retains_cleanup_until_late_success(tmp_path):
    provider = DeferredRemoteProvider(tmp_path)
    connector = _make_connector(tmp_path, provider, timeout=0.1)
    submission = None
    try:
        connector.ensure_ready("slow", [1, 2], 0)
        submission = provider.submissions.get(timeout=5)

        def timed_out():
            try:
                connector.ensure_ready("slow", [1, 2], 0)
            except TimeoutError:
                return True
            return False

        _wait_until(timed_out)
        connector.finish("slow")
        assert connector.has_cleanup and not submission[2].cancelled()
        assert not provider.released
        provider.succeed(submission)
        _wait_until(lambda: (connector.poll(), not connector.has_cleanup)[1])
        assert provider.released == [submission[0]]
    finally:
        if submission is not None and not submission[2].done():
            provider.succeed(submission)
        connector.close()


@pytest.mark.cpu_test
def test_deadline_applies_to_completion_and_restarts_on_coverage_rollback(
    tmp_path, monkeypatch
):
    from vllm.v1.ple import connector as module

    now = 0.0
    monkeypatch.setattr(module, "monotonic", lambda: now)
    provider = DeferredRemoteProvider(tmp_path)
    connector = _make_connector(tmp_path, provider, timeout=2)
    tokens = [1, 2, 3, 4]
    try:
        connector.prefetch("req", tokens, 3)
        first = provider.submissions.get(timeout=5)
        # Serialize with the hash executor so the callback is installed before
        # completing the future at the chosen clock value.
        connector.prefetch("late", tokens, 1)
        late = provider.submissions.get(timeout=5)
        now = 1.0
        provider.succeed(first)
        now = 3.0
        provider.succeed(late)
        assert connector.ensure_ready("req", tokens, 3)
        with pytest.raises(TimeoutError):
            connector.ensure_ready("late", tokens, 1)
        connector.finish("late")

        # The KV transfer supplied less than predicted. The old generation's
        # deadline has expired; the replacement starts with its own deadline.
        assert not connector.ensure_ready("req", tokens, 1)
        replacement = provider.submissions.get(timeout=5)
        assert replacement[0] != first[0] and replacement[1].tolist() == [2, 3, 4]
        provider.succeed(replacement)
        _wait_until(lambda: connector.ensure_ready("req", tokens, 2))
        assert connector.dispatch("req").start == 1
    finally:
        for future in list(provider.producing.values()):
            if not future.done():
                future.set_exception(RuntimeError("test ended"))
        connector.close(consumers_stopped=True)


@pytest.mark.cpu_test
def test_shutdown_reports_outstanding_result_and_can_retry(tmp_path):
    from .utils import create_scheduler

    provider = DeferredRemoteProvider(tmp_path)
    connector = _make_connector(tmp_path, provider)
    scheduler = create_scheduler(use_v2_model_runner=True)
    scheduler.ple_connector = connector
    lookup = None
    try:
        connector.prefetch("slow", [1], 0)
        lookup = provider.submissions.get(timeout=5)
        connector.finish("slow")
        assert scheduler.has_requests()
        connector.begin_shutdown()
        assert not scheduler.has_requests()
        with pytest.raises(TimeoutError, match=lookup[0]):
            connector.close(timeout=0)
        assert Path(connector.reader_config.pool.path).exists()
        assert not lookup[2].cancelled()
        provider.succeed(lookup)
        connector.close()
        assert provider.released == [lookup[0]]
        assert not Path(connector.reader_config.pool.path).exists()
    finally:
        if lookup is not None and not lookup[2].done():
            provider.succeed(lookup)
        scheduler.shutdown()


@pytest.mark.cpu_test
@pytest.mark.parametrize("synchronous", [True, False])
def test_cancelled_release_is_retried(tmp_path, monkeypatch, synchronous):
    provider = DeferredRemoteProvider(tmp_path)
    connector = _make_connector(tmp_path, provider, max_preparations=1)
    release = provider.release
    attempts = []

    def release_once(result_id):
        attempts.append(result_id)
        if len(attempts) == 1:
            if synchronous:
                raise asyncio.CancelledError()
            provider.release_error = asyncio.CancelledError()
        return release(result_id)

    monkeypatch.setattr(provider, "release", release_once)
    try:
        connector.prefetch("req", [1], 0)
        lookup = provider.submissions.get(timeout=5)
        provider.succeed(lookup)
        _wait_until(lambda: connector.ensure_ready("req", [1], 0))
        connector.finish("req")
        _wait_until(lambda: (connector.poll(), not connector.has_cleanup)[1])
        assert attempts == [lookup[0], lookup[0]]
        assert provider.released == [lookup[0]]
    finally:
        connector.close()


@pytest.mark.cpu_test
@pytest.mark.parametrize("stop_fails", [False, True])
def test_partial_reader_initialization_keeps_pool_until_workers_stop(
    tmp_path, stop_fails
):
    import mmap

    from vllm.config import VllmConfig
    from vllm.v1.executor.uniproc_executor import UniProcExecutor

    connector = _make_connector(tmp_path, torch.arange(8, dtype=torch.uint8).view(4, 2))
    path = Path(connector.reader_config.pool.path)

    class FailingWorkerGroup(UniProcExecutor):
        def _init_executor(self):
            self.mapping = None

        def collective_rpc(self, method, *, args):
            assert method == "initialize_ple_reader"
            (config,) = args
            with open(config.pool.path, "r+b") as pool:
                self.mapping = mmap.mmap(pool.fileno(), config.pool.size)
            raise RuntimeError("another worker failed")

        def shutdown(self):
            assert path.exists() and self.mapping is not None
            if stop_fails:
                raise RuntimeError("worker still alive")
            self.mapping.close()

    group = FailingWorkerGroup(VllmConfig())
    try:
        with pytest.raises(
            RuntimeError,
            match="worker still alive" if stop_fails else "another worker failed",
        ):
            group.initialize_ple_readers(connector)
        assert path.exists() == stop_fails
        assert group.mapping is not None and group.mapping.closed != stop_fails
    finally:
        if group.mapping is not None:
            group.mapping.close()
        connector.close(consumers_stopped=True)


@pytest.mark.cpu_test
def test_queued_local_work_can_be_withdrawn_while_pool_is_full(tmp_path):
    table = torch.arange(8, dtype=torch.uint8).view(4, 2)
    pool = LocalPLEProvider(table, pool_bytes=2, directory=str(tmp_path))
    try:
        pool.prepare("held", torch.tensor([0])).result(5)
        waiting = pool.prepare("waiting", torch.tensor([1]))
        queued = pool.prepare("queued", torch.tensor([2]))
        pool.release("queued").result(5)
        with pytest.raises(RuntimeError, match="withdrawn"):
            queued.result()
        assert not waiting.done()
        pool.release("held").result(5)
        assert waiting.result(5).result_id == "waiting"
    finally:
        pool.close()


@pytest.fixture(scope="module")
def qwen4_ple_checkpoint(tmp_path_factory):
    from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

    path = tmp_path_factory.mktemp("qwen4-ple")
    config = Qwen4ExpTextConfig(
        output_gate_type="sigmoid",
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        hc_count=4,
        hc_lowrank=16,
        ple_layer_ids=[1],
        ple_embed_dim=128,
        ngram_size=2,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=2,
        eos_token_id=1,
        max_position_embeddings=64,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        Qwen4ExpForCausalLM(config).to(torch.bfloat16).save_pretrained(path)
    return str(path)


@pytest.mark.skipif(
    torch.version.cuda is None or not torch.cuda.is_available(), reason="CUDA required"
)
@pytest.mark.parametrize(
    ("role", "mtp", "supported"),
    [("kv_producer", True, True), (None, True, False), ("kv_consumer", False, False)],
)
def test_qwen4_ple_config_enforces_prefill_role(
    qwen4_ple_checkpoint, monkeypatch, role, mtp, supported
):
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import EngineArgs

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    args = EngineArgs(
        model=qwen4_ple_checkpoint,
        skip_tokenizer_init=True,
        enforce_eager=True,
        max_model_len=32,
        hf_overrides={"mtp_num_hidden_layers": 1},
        engram_config={"prefill_ple_provider": "local"},
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleConnector", kv_role=role
        )
        if role
        else None,
        speculative_config={"method": "mtp", "num_speculative_tokens": 1}
        if mtp
        else None,
    )
    if supported:
        resolved = args.create_engine_config()
        assert (
            resolved.speculative_config.draft_model_config.architecture == "Qwen4ExpMTP"
        )
        assert resolved.load_config.safetensors_load_strategy == "lazy"
    else:
        with pytest.raises(ValueError, match="prefill producer"):
            args.create_engine_config()


@pytest.mark.skipif(
    torch.version.cuda is None or not torch.cuda.is_available(), reason="CUDA required"
)
def test_llm_generates_after_a_ple_lookup_error(
    qwen4_ple_checkpoint, vllm_runner, monkeypatch
):
    from vllm import SamplingParams

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    with vllm_runner(
        qwen4_ple_checkpoint,
        dtype="bfloat16",
        enforce_eager=True,
        skip_tokenizer_init=True,
        enable_prefix_caching=False,
        max_model_len=32,
        max_num_seqs=2,
        max_num_batched_tokens=32,
        disable_log_stats=False,
        gpu_memory_utilization=0.01,
        kv_cache_memory_bytes=32 << 20,
        engram_config={
            "prefill_ple_provider": "local",
            "prefill_ple_provider_options": {"pool_bytes": 512},
        },
    ) as runner:
        params = SamplingParams(max_tokens=1, temperature=0, logprobs=1)
        # One token needs 256 bytes of rows. Three exceed the pool; two fit.
        (failed,) = runner.llm.generate({"prompt_token_ids": [2, 3, 4]}, params)
        assert failed.finished and failed.error.code == "ple_lookup_failed"
        assert (
            failed.outputs[0].finish_reason == "error"
            and not failed.outputs[0].token_ids
        )
        (ok,) = runner.llm.generate({"prompt_token_ids": [2, 3]}, params)
        assert ok.finished and ok.error is None and len(ok.outputs[0].token_ids) == 1
        from vllm.v1.metrics.reader import Gauge

        gauges = {
            m.name: m.value for m in runner.llm.get_metrics() if isinstance(m, Gauge)
        }
        assert (
            gauges["vllm:ple_failed_requests"]
            == gauges["vllm:ple_success_requests"]
            == 1
        )
        assert gauges["vllm:ple_inflight_requests"] == 0
        assert gauges["vllm:ple_latency_ms"] == gauges["vllm:ple_latency_max_ms"] > 0


@pytest.mark.cpu_test
def test_remote_read_failure_after_transport_success_is_not_a_fetch_failure(tmp_path):
    class ObservedRemote(DeferredRemoteProvider):
        def prepare(self, result_id, row_ids, *, observation=None):
            assert observation is not None
            self.observation = observation
            return super().prepare(result_id, row_ids)

    provider = ObservedRemote(tmp_path)
    connector = _make_connector(tmp_path, provider)
    try:
        connector.prefetch("failed-read", [1], 0)
        lookup = provider.submissions.get(timeout=5)
        assert connector.metrics.snapshot()["ple_inflight_requests"] == 1
        provider.observation.finish()  # Valid local bytes, READ still outstanding.
        assert not connector.ensure_ready("failed-read", [1], 0)
        lookup[2].set_exception(RuntimeError("READ acquire failed"))
        with pytest.raises(RuntimeError, match="READ acquire failed"):
            _wait_until(lambda: connector.ensure_ready("failed-read", [1], 0))
        connector.finish("failed-read")
        _wait_until(lambda: (connector.poll(), not connector.has_cleanup)[1])
        stats = connector.metrics.snapshot()
        assert stats["ple_success_requests"] == 1
        assert stats["ple_failed_requests"] == stats["ple_inflight_requests"] == 0
        assert provider.released == [lookup[0]]
    finally:
        connector.close()


def remote_factory(*, table, load_table, options):
    assert table.name == "table-v1" and table.num_rows == 16
    return DeferredRemoteProvider(options["directory"])


@pytest.mark.cpu_test
def test_external_provider_factory_does_not_materialize_checkpoint(tmp_path):
    from vllm.config.engram import EngramConfig
    from vllm.v1.ple.factory import create_provider
    from vllm.v1.ple.types import PLETableSpec

    config = EngramConfig(
        prefill_ple_provider=f"{__name__}.remote_factory",
        prefill_ple_provider_options={"directory": str(tmp_path)},
    )
    assert config.prefill_ple_enabled
    assert config.prefill_ple_pool_bytes is None
    provider = create_provider(
        config.prefill_ple_provider,
        table=PLETableSpec("model", "revision", "table-v1", 16, 2, "torch.bfloat16"),
        load_table=lambda: pytest.fail("RemoteProvider must not need a local table"),
        options=config.prefill_ple_provider_options,
    )
    try:
        assert provider.export_pool(2).size == 32
    finally:
        provider.close()


def _read_exported_pool(pool, rank, chunks, output):
    import mmap
    import os

    fd = pool.handles[rank].detach()
    try:
        with mmap.mmap(fd, pool.size) as memory:
            output.send(
                b"".join(memory[c.offset : c.offset + c.nbytes] for c in chunks)
            )
    finally:
        os.close(fd)
        output.close()


@pytest.mark.cpu_test
def test_fd_pool_export_is_consumable_by_each_worker(tmp_path):
    import multiprocessing
    import os
    from dataclasses import replace
    from multiprocessing.reduction import DupFd

    provider = LocalPLEProvider(
        torch.arange(8, dtype=torch.uint8).view(4, 2),
        pool_bytes=8,
        directory=str(tmp_path),
    )
    context = multiprocessing.get_context("spawn")
    processes, receivers = [], []
    try:
        result = provider.prepare("r", torch.tensor([3, 0, 3])).result(5)
        pool = provider.export_pool(2)
        fd = os.open(pool.path, os.O_RDWR)
        try:
            pool = replace(pool, path=None, handles=(DupFd(fd), DupFd(fd)))
        finally:
            os.close(fd)
        for rank in range(2):
            receiver, sender = context.Pipe(duplex=False)
            process = context.Process(
                target=_read_exported_pool, args=(pool, rank, result.chunks, sender)
            )
            process.start()
            sender.close()
            processes.append(process)
            receivers.append(receiver)
        for receiver, process in zip(receivers, processes):
            assert receiver.poll(30)
            assert receiver.recv() == bytes([6, 7, 0, 1, 6, 7])
            process.join(30)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
            process.close()
        for receiver in receivers:
            receiver.close()
        provider.close()
