"""CPU-only check for the bounded B8/B1 graph observer protocol."""

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from sglang.srt.mem_cache.qsa_hisparse_slots import QSAHiSparseSlots
from sglang.srt.mem_cache.qsa_hisparse_trace import QSAHiSparseTrace


class Projection(torch.nn.Module):
    def forward(self, value):
        return value.reshape(value.shape[0], -1)[:, :8], None


class Gate(torch.nn.Module):
    def forward(self, value):
        logits = value[:, :1] + torch.arange(512, dtype=value.dtype)[None]
        return logits, None


class Experts(torch.nn.Module):
    def forward(self, hidden_states, _topk):
        return hidden_states + 2


class TopK(torch.nn.Module):
    def forward(self, _hidden_states, logits):
        return SimpleNamespace(
            topk_weights=torch.softmax(logits.float(), dim=-1)[:, -10:],
            topk_ids=torch.arange(502, 512, dtype=torch.int32)[None].repeat(
                logits.shape[0], 1
            ),
        )


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = Gate()
        self.topk = TopK()
        self.experts = Experts()
        self.shared_expert = None

    def forward(self, hidden_states, _batch):
        logits, _ = self.gate(hidden_states)
        return self.experts(hidden_states, self.topk(hidden_states, logits))


class Layer(torch.nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
        self.mlp = MLP()
        setattr(self, "o_proj" if index % 4 == 3 else "linear_attn", Projection())

    def forward(self, *, hidden_states, forward_batch, **_kwargs):
        count = hidden_states.shape[0]
        if self.index % 4 == 3:
            value = torch.full((count, 12 * 256), self.index, dtype=torch.bfloat16)
            hidden, _ = self.o_proj(value)
        else:
            hidden, _ = self.linear_attn(hidden_states[:, :8])
        hidden = self.mlp(hidden, forward_batch)
        return hidden.repeat(1, 4), None


class Body(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_experts=512)
        self.hc_count = 4
        self.embed_tokens = torch.nn.Embedding(256, 8, dtype=torch.bfloat16)
        self.layers = torch.nn.ModuleList([Layer(index) for index in range(48)])
        self.hyper_connection_mixer = torch.nn.Identity()
        self.last_hc_hidden_states = None
        self.trace = None

    def forward(self, input_ids, positions, forward_batch):
        hidden = self.embed_tokens(forward_batch.input_ids if input_ids is None else input_ids)
        for index, layer in enumerate(self.layers):
            if index % 4 == 3:
                count = hidden.shape[0]
                q = torch.full((count, 12, 256), index, dtype=torch.bfloat16)
                indices = torch.arange(2051, dtype=torch.int32)[None].repeat(count, 1)
                self.trace.attention(
                    SimpleNamespace(layer_id=index, scaling=0.0625),
                    q,
                    None,
                    None,
                    indices,
                    q + 1,
                    0.75,
                    1.25,
                    torch.full((count,), 2051, dtype=torch.int32),
                    None,
                    None,
                )
            hidden, _ = layer(
                hidden_states=hidden,
                forward_batch=forward_batch,
                positions=positions,
                residual=None,
            )
        self.last_hc_hidden_states = hidden
        return hidden[:, :8]


class Logits(torch.nn.Module):
    def forward(self, hidden, _batch):
        return SimpleNamespace(next_token_logits=hidden.float().repeat(1, 2))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=16)
        self.is_mrope_enabled = True
        self.model = Body()
        self.logits_processor = Logits()

    def forward(self, batch):
        hidden = self.model(None, batch.mrope_positions, batch)
        return self.logits_processor(hidden, batch)


@dataclass(frozen=True)
class Lease:
    req_pool_idx: int
    generation: int
    rid: str
    slot: int


class TestTraceProtocol(unittest.TestCase):
    def test_target_row_freshness_and_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Model()
            adapter = SimpleNamespace(
                strict=True,
                mode="p2-offload",
                rank=0,
                device="cpu",
                max_requests=8,
                layer_ids=list(range(3, 48, 4)),
                graph_enabled=True,
                graph_capture_size=None,
                workspace=[],
                graph_batch=None,
                forward_id=0,
                batch_requests=[],
                runner=SimpleNamespace(
                    model=model,
                    model_config=SimpleNamespace(dtype=torch.bfloat16, hidden_size=8),
                ),
                slots=QSAHiSparseSlots(4096, 4, 8),
            )
            states = {}
            adapter._request = lambda index, rid: states[index]
            trace = QSAHiSparseTrace(adapter, directory)
            model.model.trace = trace

            def forward(count, base):
                batch = SimpleNamespace(
                    forward_mode=SimpleNamespace(is_decode=lambda: True),
                    input_ids=torch.arange(base, base + count, dtype=torch.int64),
                    positions=torch.arange(count, dtype=torch.int64),
                    mrope_positions=torch.arange(3 * count, dtype=torch.int64).reshape(3, count),
                )
                model(batch)
                return batch

            for count in (8, 1):
                adapter.graph_capture_size = count
                forward(count, 0)
            adapter.graph_capture_size = None
            self.assertEqual(list(Path(directory).glob("*.pt")), [])

            def install(label, count, row, step):
                nonlocal states
                rid = next(rid for rid, value in trace.TARGETS.items() if value == label)
                leases = [Lease(index, 1, f"other-{label}-{index}", index)
                          for index in range(count)]
                leases[row] = Lease(row, 2, rid, row)
                batch_states = [SimpleNamespace(
                    lease=lease,
                    seq_len=trace.PROMPT + step,
                    decode_steps=step,
                ) for lease in leases]
                states = {state.lease.req_pool_idx: state for state in batch_states}
                adapter.batch_requests = batch_states
                adapter.graph_batch = tuple((state.lease, state.seq_len) for state in batch_states)
                adapter.slots.active.clear()
                adapter.slots.phases.clear()
                for state in batch_states:
                    adapter.slots.active[state.lease.req_pool_idx] = state.lease
                    adapter.slots.phases[state.lease.req_pool_idx] = "decode"
                return batch_states[row]

            samples = (("b8", 8, 3, 8, 100), ("b8", 8, 3, 9, 110),
                       ("b1", 1, 0, 8, 200), ("b1", 1, 0, 9, 210))
            for label, count, row, step, base in samples:
                state = install(label, count, row, step)
                adapter.forward_id += 1
                batch = SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: True))
                trace.begin(batch)
                if label == "b8" and step == 8:
                    with self.assertRaisesRegex(RuntimeError, "stale or missing"):
                        trace.finish(graph=True)
                forward(count, base)
                if label == "b8" and step == 9:
                    original = state.lease
                    state.lease = replace(original, generation=3)
                    with self.assertRaisesRegex(RuntimeError, "identity/step changed"):
                        trace.finish(graph=True)
                    state.lease = original
                trace.finish(graph=True)

            b8 = torch.load(Path(directory) / "b8-step-008-rank-0.pt", weights_only=True)
            b1 = torch.load(Path(directory) / "b1-step-008-rank-0.pt", weights_only=True)
            self.assertEqual(b8["schema"], "qsa-b8-shape-trace-v1")
            self.assertEqual((b8["batch_size"], b8["row"]), (8, 3))
            self.assertEqual(b8["tensors"]["input_ids"].item(), 103)
            self.assertEqual(b1["tensors"]["input_ids"].item(), 200)
            self.assertEqual(b8["tensors"]["router.0"].shape, (1, 512))
            self.assertEqual(b8["tensors"]["topk.0.ids"].shape, (1, 10))
            self.assertEqual(b8["tensors"]["qsa.3.indices"].shape, (1, 2051))
            trace.buffers["input_ids"].zero_()
            self.assertEqual(b8["tensors"]["input_ids"].item(), 103)
            self.assertEqual(len(list(Path(directory).glob("*.pt"))), 4)
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                trace.begin(batch)


if __name__ == "__main__":
    unittest.main()
