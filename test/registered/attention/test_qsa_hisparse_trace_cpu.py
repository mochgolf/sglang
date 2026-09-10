"""Observer protocol only: CPU tensors and a substituted model, no CUDA."""

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from sglang.srt.mem_cache.qsa_hisparse_trace import QSAHiSparseTrace
from sglang.srt.mem_cache.qsa_hisparse_slots import QSAHiSparseSlots


class Layer(torch.nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
        setattr(self, "o_proj" if index % 4 == 3 else "linear_attn", torch.nn.Identity())

    def forward(self, hidden, residual):
        child = self.o_proj if self.index % 4 == 3 else self.linear_attn
        return child(hidden[:, :8] + 1).repeat(1, 4), None


class Body(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hc_count = 4
        self.embed_tokens = torch.nn.Embedding(4, 8, dtype=torch.bfloat16)
        self.layers = torch.nn.ModuleList([Layer(i) for i in range(48)])
        self.hyper_connection_mixer = torch.nn.Identity()
        self.trace = None

    def forward(self, input_ids, positions, forward_batch):
        if input_ids is None:
            input_ids = forward_batch.input_ids
        hidden = self.embed_tokens(input_ids)
        residual = None
        for index, layer in enumerate(self.layers):
            if forward_batch.forward_mode.is_decode() and index % 4 == 3:
                count = input_ids.numel()
                q = torch.full((count, 12, 256), float(index), dtype=torch.bfloat16)
                k = torch.full((count * 2051, 1, 256), float(index + 1), dtype=torch.bfloat16)
                v = k + 1
                cu = torch.arange(count + 1, dtype=torch.int32)
                self.trace.attention(SimpleNamespace(layer_id=index, scaling=0.0625),
                    q, k, v, torch.zeros((count, 2051), dtype=torch.int32), q + 1,
                    0.75, 1.25, torch.full((count,), 2051, dtype=torch.int32), cu, cu * 2051)
            hidden, residual = layer(hidden, residual)
        return hidden[:, :8]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=16)
        self.is_mrope_enabled = True
        self.model = Body()
        self.logits_processor = Logits()

    def forward(self, input_ids, positions, forward_batch):
        hidden = self.model(input_ids=None, positions=forward_batch.mrope_positions,
                            forward_batch=forward_batch)
        return self.logits_processor(hidden, forward_batch)


class Logits(torch.nn.Module):
    def forward(self, hidden, forward_batch):
        return SimpleNamespace(next_token_logits=hidden[-1:].float().repeat(1, 2)
                               if not forward_batch.forward_mode.is_decode()
                               else hidden.float().repeat(1, 2))


@dataclass(frozen=True)
class Lease:
    req_pool_idx: int = 2
    generation: int = 4
    rid: str = QSAHiSparseTrace.RID
    slot: int = 0


class TestTraceProtocol(unittest.TestCase):
    def test_order_freshness_and_immutable_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Model()
            state = SimpleNamespace(lease=Lease(), seq_len=QSAHiSparseTrace.PROMPT, decode_steps=0)
            adapter = SimpleNamespace(strict=True, mode="p2-offload", rank=0, device="cpu",
                layer_ids=list(range(3, 48, 4)), graph_enabled=True, graph_capture_size=None,
                workspace=[], graph_batch=None, slots=QSAHiSparseSlots(4096, 4, 2),
                forward_id=1, batch_requests=[state], runner=SimpleNamespace(model=model,
                model_config=SimpleNamespace(dtype=torch.bfloat16, hidden_size=8)))
            adapter.slots.active[state.lease.req_pool_idx] = state.lease
            adapter.slots.phases[state.lease.req_pool_idx] = "prefill"
            adapter._request = lambda index, rid: state
            trace = QSAHiSparseTrace(adapter, directory)
            model.model.trace = trace
            self.assertEqual(sum(t.nbytes for t in adapter.workspace),
                             sum(t.nbytes for t in trace.buffers.values()) + trace.epoch.nbytes + trace.stamps.nbytes)
            pointers = [t.data_ptr() for t in trace.buffers.values()]

            def forward(decode, count):
                batch = SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: decode),
                    input_ids=torch.ones(count, dtype=torch.int64),
                    positions=torch.ones(count, dtype=torch.int64),
                    mrope_positions=torch.ones((3, count), dtype=torch.int64))
                # Real runners bypass root Module hooks with a direct forward call.
                model.forward(batch.input_ids, batch.positions, batch)
                return batch

            # Simulated capture refreshes staging but must never emit a sample.
            for count in (2, 1):
                adapter.graph_capture_size = count
                forward(True, count)
            adapter.graph_capture_size = None
            self.assertEqual(list(Path(directory).glob("*.pt")), [])
            for step in (0, 1, 2):
                adapter.forward_id = step + 1
                state.seq_len, state.decode_steps = trace.PROMPT + step, step
                adapter.slots.phases[state.lease.req_pool_idx] = "decode" if step else "prefill"
                adapter.graph_batch = ((state.lease, state.seq_len),) if step else None
                batch = SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: step > 0))
                trace.begin(batch)
                # Capture-time/stale staging cannot pass as a real forward.
                with self.assertRaisesRegex(RuntimeError, "stale or missing"):
                    trace.finish(graph=step > 0)
                forward(step > 0, 1 if step else 4)
                if step == 1:
                    original = state.lease
                    state.lease = replace(original, generation=original.generation + 1)
                    with self.assertRaisesRegex(RuntimeError, "identity/step changed"):
                        trace.finish(graph=True)
                    state.lease = original
                    adapter.graph_batch = ((original, state.seq_len + 1),)
                    with self.assertRaisesRegex(RuntimeError, "identity/step changed"):
                        trace.finish(graph=True)
                    adapter.graph_batch = ((original, state.seq_len),)
                    index = trace.field_indices["qsa.3.k"]
                    trace.stamps[index] = -1
                    with self.assertRaisesRegex(RuntimeError, "stale or missing"):
                        trace.finish(graph=True)
                    trace.stamps[index] = adapter.forward_id
                trace.finish(graph=step > 0)
            saved = torch.load(Path(directory) / "step-001-rank-0.pt", weights_only=True)
            self.assertEqual(saved["tensors"]["qsa.3.k"].shape, (2051, 1, 256))
            trace.buffers["qsa.3.k"].zero_()
            self.assertTrue(torch.all(saved["tensors"]["qsa.3.k"] == 4))
            self.assertEqual(saved["scales"][3], (0.75, 1.25, 0.0625))
            self.assertEqual(saved["tensors"]["layer.47.hidden"].shape, (1, 32))
            self.assertEqual(saved["tensors"]["final_mix"].shape, (1, 8))
            self.assertEqual(saved["null_fields"], [f"layer.{i}.residual" for i in range(48)])
            self.assertEqual([t.data_ptr() for t in trace.buffers.values()], pointers)
            self.assertEqual(len(list(Path(directory).glob("*.pt"))), 3)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            with self.assertRaisesRegex(RuntimeError, "out-of-order"):
                trace.begin(batch)
            state.lease = replace(state.lease, generation=5)
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                trace.begin(batch)


if __name__ == "__main__":
    unittest.main()
