# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from concurrent.futures import Future
from functools import partial
from pathlib import Path
from queue import Queue
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
    while not predicate():
        assert time.monotonic() < deadline, "Timed out waiting for PLE"
        time.sleep(0.001)


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
            assert not prepared.is_set()
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
    connector = _make_connector(tmp_path, table, table.nbytes, row_ids)
    scheduler.ple_connector = connector
    try:
        scheduler.add_request(request)
        if remote_hit:
            _step_until_kv_transfer_finished(scheduler, [request.request_id])
            assert prepared == []

        output = scheduler.schedule()

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
        result = output.ple_results[output.scheduled_new_reqs[0].req_id]
        assert result is not None and (result.start, result.end) == expected
        scheduler.update_from_output(output, make_output(scheduler))
        assert connector.retire_ids() == {result.rows.result_id}
    finally:
        scheduler.shutdown()


@pytest.mark.skipif(
    torch.version.cuda is None or not torch.cuda.is_available(), reason="CUDA required"
)
@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("registration", ["pool", "region"])
@pytest.mark.parametrize("transport", ["path", "fd"])
def test_reader_stages_packed_chunks_and_padding(
    tmp_path, use_graph, registration, transport
):
    import os
    from dataclasses import replace
    from multiprocessing.reduction import DupFd

    from vllm.v1.ple.reader import PLEReader
    from vllm.v1.ple.types import PLEChunk
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

    table = torch.arange(16, dtype=torch.uint8).view(8, 2)
    connector = _make_connector(tmp_path, table)
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
        self.fail_release_once = False

    def export_pool(self, num_workers):
        return self.storage.export_pool(num_workers)

    def prepare(self, result_id, row_ids):
        future: Future[PreparedRows] = Future()
        self.submissions.put((result_id, row_ids, future))
        return future

    def succeed(self, submission):
        result_id, row_ids, future = submission
        future.set_result(self.storage.prepare(result_id, row_ids).result(5))

    def release(self, result_id):
        if self.fail_release_once:
            self.fail_release_once = False
            failed: Future[None] = Future()
            failed.set_exception(RuntimeError("temporary READ release failure"))
            return failed
        self.released.append(result_id)
        return self.storage.release(result_id)

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
                connector.ensure_ready("first", [1, 2], 0)
            connector.finish("first")
        else:
            rows = provider.storage.prepare(first[0], first[1]).result(5)
            first[2].set_result(replace(rows, pool_id="stale-generation"))
            with pytest.raises(RuntimeError, match="invalid PLE result"):
                connector.ensure_ready("first", [1, 2], 0)
            provider.fail_release_once = True
            connector.finish("first")
            assert connector.has_cleanup
        provider.succeed(second)
        assert connector.ensure_ready("second", [3, 4], 0)
        _wait_until(lambda: (connector.poll(), not connector.has_cleanup)[1])
        assert first[0] in provider.released
    finally:
        for submission in (first, second):
            if submission is not None and not submission[2].done():
                submission[2].set_exception(RuntimeError("test ended"))
        connector.close()


@pytest.mark.cpu_test
def test_cancel_during_hash_still_releases_late_production(tmp_path):
    started, proceed = Event(), Event()
    provider = DeferredRemoteProvider(tmp_path)

    def hash_ids(tokens, start):
        started.set()
        assert proceed.wait(5)
        return torch.tensor(tokens[start:])

    connector = _make_connector(
        tmp_path, provider, row_ids=hash_ids, max_preparations=1
    )
    submission = None
    try:
        connector.ensure_ready("first", [1, 2], 0)
        assert started.wait(5)
        connector.finish("first")
        assert not connector.ensure_ready("next", [3, 4], 0)
        proceed.set()
        submission = provider.submissions.get(timeout=5)
        provider.succeed(submission)
        _wait_until(lambda: (connector.poll(), not connector.has_cleanup)[1])
        assert provider.released == [submission[0]]
    finally:
        proceed.set()
        if submission is None:
            submission = provider.submissions.get(timeout=5)
            provider.succeed(submission)
        connector.close()


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
