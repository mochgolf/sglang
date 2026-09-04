"""GDN ReplaySSM fold-every-commit: fused ring-write + commit fold.

The production kernel targets bitwise parity with the recurrent verify and
per-draft snapshot baseline. These tests allow an absolute error up to FP32_ATOL
for committed/tracked state and downstream outputs, and verify that bound
through 256 chained commits. Ring-write output and untouched/null slots remain
exact.
"""

import unittest

import torch
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold import (
    commit_gdn_replayssm_fold_all_layers,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")

B, T = 3, 4
H, HV = 4, 8
K = V = 64
NUM_SLOTS = 8
DEVICE = "cuda"
# Absolute allowance for this deterministic regression case, not a general
# numerical-error guarantee for ReplaySSM.
FP32_ATOL = 2 * torch.finfo(torch.float32).eps


def _make_window(step_seed: int):
    gen = torch.Generator(device=DEVICE).manual_seed(step_seed)

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device=DEVICE, dtype=dtype, generator=gen)

    return {
        "q": rand(1, B * T, H, K),
        "k": rand(1, B * T, H, K),
        "v": rand(1, B * T, HV, V),
        "a": rand(B * T, HV),
        "b": rand(B * T, HV),
    }


def _run_verify(inputs, gating, state, slots, *, snapshots=None, rings=None):
    kwargs = {}
    if snapshots is not None:
        kwargs.update(
            intermediate_states_buffer=snapshots,
            intermediate_state_indices=slots,
            cache_steps=T,
        )
    if rings is not None:
        # Per-layer views, matching the backend's mamba2_layer_cache slices.
        kwargs.update(
            cache_ring=True,
            replayssm_rawv=rings["rawv"][0],
            replayssm_rawk=rings["rawk"][0],
            replayssm_g=rings["g"][0],
            replayssm_beta=rings["beta"][0],
        )
    cu_seqlens = torch.arange(0, B * T + 1, step=T, dtype=torch.int32, device=DEVICE)
    return fused_sigmoid_gating_delta_rule_update(
        A_log=gating["A_log"],
        dt_bias=gating["dt_bias"],
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        b=inputs["b"],
        a=inputs["a"],
        initial_state_source=state,
        initial_state_indices=slots,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        is_kda=False,
        disable_state_update=True,
        **kwargs,
    )


def _make_rings(dtype=torch.bfloat16):
    return {
        "rawv": torch.zeros(1, NUM_SLOTS, HV, T, V, device=DEVICE, dtype=dtype),
        "rawk": torch.zeros(1, NUM_SLOTS, H, T, K, device=DEVICE, dtype=dtype),
        "g": torch.zeros(1, NUM_SLOTS, HV, T, device=DEVICE, dtype=torch.float32),
        "beta": torch.zeros(1, NUM_SLOTS, HV, T, device=DEVICE, dtype=torch.float32),
    }


def _fold(state, rings, slots, accept_lens, track_slots=None, track_steps=None):
    commit_gdn_replayssm_fold_all_layers(
        checkpoint_state=state,
        rawv_cache=rings["rawv"],
        rawk_cache=rings["rawk"],
        g_cache=rings["g"],
        beta_cache=rings["beta"],
        ssm_state_indices=slots,
        accept_lens=accept_lens,
        max_cache_len=T,
        num_k_heads=H,
        mamba_track_indices=track_slots,
        mamba_steps_to_track=track_steps,
        null_block_id=-1,
    )


def _literal_fp32(inputs, gating, state):
    """Small independent oracle for S <- decay*S + beta*(v-Sk) k^T."""
    outputs, snapshots = [], []
    q, k, v, a, b = (inputs[name].float() for name in ("q", "k", "v", "a", "b"))
    group = v.shape[2] // k.shape[2]
    for t in range(k.shape[1]):
        qt = q[:, t] / torch.sqrt((q[:, t].square().sum(-1, keepdim=True)) + 1e-6)
        kt = k[:, t] / torch.sqrt((k[:, t].square().sum(-1, keepdim=True)) + 1e-6)
        qt = qt.repeat_interleave(group, dim=1) * (k.shape[-1] ** -0.5)
        kt = kt.repeat_interleave(group, dim=1)
        x = a[:, t] + gating["dt_bias"]
        softplus = torch.where(x <= 20, torch.log1p(torch.exp(x)), x)
        decay = torch.exp(-torch.exp(gating["A_log"]) * softplus)
        beta = torch.sigmoid(b[:, t])
        state.mul_(decay[..., None, None])
        delta = v[:, t] - (state * kt[..., None, :]).sum(-1)
        delta.mul_(beta[..., None])
        state.add_(delta[..., None] * kt[..., None, :])
        outputs.append((state * qt[..., None, :]).sum(-1).bfloat16())
        snapshots.append(state.clone())
    return torch.stack(outputs, 1), torch.stack(snapshots, 1)


def _production_case(T, seed):
    B, H_, HV_, K_, V_ = 1, 8, 24, 128, 128
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device=DEVICE, dtype=dtype, generator=gen)

    inputs = {
        "q": rand(B, T, H_, K_),
        "k": rand(B, T, H_, K_),
        "v": rand(B, T, HV_, V_),
        "a": rand(B, T, HV_),
        "b": rand(B, T, HV_),
    }
    gating = {
        "A_log": (rand(HV_, dtype=torch.float32) * 0.1).contiguous(),
        "dt_bias": (rand(HV_, dtype=torch.float32) * 0.1).contiguous(),
    }
    return inputs, gating


def _generic(inputs, gating, state, slot, *, snapshots=None, rings=None):
    T_ = inputs["k"].shape[1]
    kwargs = {}
    if snapshots is not None:
        kwargs.update(
            disable_state_update=True,
            intermediate_states_buffer=snapshots,
            intermediate_state_indices=slot,
            cache_steps=T_,
        )
    if rings is not None:
        kwargs.update(
            disable_state_update=True,
            cache_ring=True,
            replayssm_rawv=rings["rawv"][0],
            replayssm_rawk=rings["rawk"][0],
            replayssm_g=rings["g"][0],
            replayssm_beta=rings["beta"][0],
        )
    return fused_sigmoid_gating_delta_rule_update(
        A_log=gating["A_log"],
        dt_bias=gating["dt_bias"],
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        a=inputs["a"].squeeze(0),
        b=inputs["b"],
        initial_state_source=state,
        initial_state_indices=slot,
        cu_seqlens=torch.tensor([0, T_], dtype=torch.int32, device=DEVICE),
        use_qk_l2norm_in_kernel=True,
        is_kda=False,
        **kwargs,
    )


class TestGdnReplayssmSpecFold(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.gating = {
            "A_log": torch.randn(HV, device=DEVICE) * 0.1,
            "dt_bias": torch.randn(HV, device=DEVICE) * 0.1,
        }
        self.slots = torch.tensor([5, 2, 7], dtype=torch.int32, device=DEVICE)
        self.accept_lens = torch.tensor([3, 1, 4], dtype=torch.int32, device=DEVICE)

    def _state(self, dtype):
        gen = torch.Generator(device=DEVICE).manual_seed(1)
        return torch.randn(
            NUM_SLOTS, HV, K, V, device=DEVICE, dtype=dtype, generator=gen
        )

    def test_tp_local_decode_paths_match_literal_fp32(self):
        """Cover the production TP-local shape across decode and spec commit."""
        slots, hv, dim = 3, 24, 128
        slot = torch.tensor([1], dtype=torch.int32, device=DEVICE)
        neighbors = torch.tensor([0, 2], device=DEVICE)
        gen = torch.Generator(device=DEVICE).manual_seed(123)
        base = (
            torch.randn(
                slots, hv, dim, dim, device=DEVICE, dtype=torch.float32, generator=gen
            )
            * 0.1
        )
        packed_state = base.clone()
        generic_state = base.clone()
        snapshot_state = base.clone()
        fold_state = base.clone()
        literal_state = base[slot.long()].clone()

        def rings(T_):
            result = {
                "rawv": torch.full(
                    (1, slots, hv, T_, dim), 37, device=DEVICE, dtype=torch.bfloat16
                ),
                "rawk": torch.full(
                    (1, slots, 8, T_, dim), 37, device=DEVICE, dtype=torch.bfloat16
                ),
                "g": torch.full(
                    (1, slots, hv, T_), 37, device=DEVICE, dtype=torch.float32
                ),
                "beta": torch.full(
                    (1, slots, hv, T_), 37, device=DEVICE, dtype=torch.float32
                ),
            }
            for value in result.values():
                value[:, 1].zero_()
            return result

        def fold(ring, state, accept):
            commit_gdn_replayssm_fold_all_layers(
                checkpoint_state=state.unsqueeze(0),
                rawv_cache=ring["rawv"],
                rawk_cache=ring["rawk"],
                g_cache=ring["g"],
                beta_cache=ring["beta"],
                ssm_state_indices=slot,
                accept_lens=torch.tensor([accept], dtype=torch.int32, device=DEVICE),
                max_cache_len=ring["rawv"].shape[-2],
                num_k_heads=8,
            )

        def assert_state(actual, expected, path):
            torch.testing.assert_close(
                actual[1], expected[0], rtol=2e-5, atol=3e-5, msg=path
            )
            self.assertTrue(torch.equal(actual[neighbors], base[neighbors]), path)

        # Several consecutive decode tokens catch small per-step drift that a
        # zero-state or isolated T=1 check cannot see.
        for step in range(4):
            inputs, gating = _production_case(1, 1000 + step)
            literal_out, literal_snapshots = _literal_fp32(
                inputs, gating, literal_state
            )
            packed_out = inputs["q"].new_empty(1, 1, hv, dim)
            mixed_qkv = torch.cat(
                [
                    inputs["q"][:, 0].flatten(1),
                    inputs["k"][:, 0].flatten(1),
                    inputs["v"][:, 0].flatten(1),
                ],
                dim=1,
            )
            fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=mixed_qkv,
                a=inputs["a"][:, 0],
                b=inputs["b"][:, 0],
                A_log=gating["A_log"],
                dt_bias=gating["dt_bias"],
                scale=dim**-0.5,
                initial_state=packed_state,
                out=packed_out,
                ssm_state_indices=slot,
                use_qk_l2norm_in_kernel=True,
            )
            generic_out = _generic(inputs, gating, generic_state, slot)

            snapshots = torch.full(
                (slots, 1, hv, dim, dim),
                37,
                device=DEVICE,
                dtype=torch.float32,
            )
            snapshot_before = snapshot_state.clone()
            snapshot_out = _generic(
                inputs, gating, snapshot_state, slot, snapshots=snapshots
            )
            self.assertTrue(torch.equal(snapshot_state, snapshot_before))
            self.assertTrue(torch.all(snapshots[neighbors] == 37))
            snapshot_state[1].copy_(snapshots[1, 0])

            ring = rings(1)
            fold_before = fold_state.clone()
            ring_out = _generic(inputs, gating, fold_state, slot, rings=ring)
            self.assertTrue(torch.equal(fold_state, fold_before))
            self.assertTrue(torch.all(ring["rawv"][:, neighbors] == 37))
            self.assertTrue(torch.all(ring["rawk"][:, neighbors] == 37))
            fold(ring, fold_state, 1)

            for path, out in (
                ("packed", packed_out),
                ("generic", generic_out),
                ("snapshot", snapshot_out),
                ("ring", ring_out),
            ):
                torch.testing.assert_close(
                    out, literal_out, rtol=1e-2, atol=2e-2, msg=f"{path} step={step}"
                )
                torch.testing.assert_close(
                    out, generic_out, rtol=1e-5, atol=2e-6, msg=f"{path} step={step}"
                )
            for path, state in (
                ("packed", packed_state),
                ("generic", generic_state),
                ("snapshot", snapshot_state),
                ("fold", fold_state),
            ):
                assert_state(state, literal_snapshots[:, -1], f"{path} step={step}")
                torch.testing.assert_close(
                    state[1], generic_state[1], rtol=1e-5, atol=2e-6, msg=path
                )

        # A multi-token verify and partial exact fold exercise ring indexing and
        # prove accept_len selects the committed prefix, not the final draft.
        inputs, gating = _production_case(4, 2000)
        literal_out, literal_snapshots = _literal_fp32(inputs, gating, literal_state)
        snapshots = torch.full(
            (slots, 4, hv, dim, dim), 37, device=DEVICE, dtype=torch.float32
        )
        snapshot_before = snapshot_state.clone()
        snapshot_out = _generic(
            inputs, gating, snapshot_state, slot, snapshots=snapshots
        )
        self.assertTrue(torch.equal(snapshot_state, snapshot_before))
        ring = rings(4)
        fold_before = fold_state.clone()
        ring_out = _generic(inputs, gating, fold_state, slot, rings=ring)
        self.assertTrue(torch.equal(fold_state, fold_before))
        fold(ring, fold_state, 3)

        torch.testing.assert_close(snapshot_out, literal_out, rtol=1e-2, atol=2e-2)
        torch.testing.assert_close(ring_out, snapshot_out, rtol=0, atol=FP32_ATOL)
        for t in range(4):
            torch.testing.assert_close(
                snapshots[1, t], literal_snapshots[0, t], rtol=2e-5, atol=3e-5
            )
        assert_state(fold_state, literal_snapshots[:, 2], "T=4 accept_len=3")
        self.assertTrue(torch.all(snapshots[neighbors] == 37))
        for value in ring.values():
            self.assertTrue(torch.all(value[:, neighbors] == 37))

    def test_ring_write_does_not_change_verify_output(self):
        for dtype in (torch.float32, torch.bfloat16):
            state = self._state(dtype)
            inputs = _make_window(11)
            out_plain = _run_verify(inputs, self.gating, state.clone(), self.slots)
            out_ring = _run_verify(
                inputs, self.gating, state.clone(), self.slots, rings=_make_rings()
            )
            self.assertTrue(torch.equal(out_plain, out_ring), f"{dtype=}")

    def test_fold_matches_snapshot_baseline(self):
        for dtype in (torch.float32, torch.bfloat16):
            state = self._state(dtype)
            inputs = _make_window(22)

            snapshots = torch.zeros(NUM_SLOTS, T, HV, K, V, device=DEVICE, dtype=dtype)
            _run_verify(
                inputs, self.gating, state.clone(), self.slots, snapshots=snapshots
            )

            fold_state = state.clone().unsqueeze(0).contiguous()
            rings = _make_rings()
            _run_verify(inputs, self.gating, fold_state[0], self.slots, rings=rings)
            _fold(fold_state, rings, self.slots, self.accept_lens)

            for s, n in zip(self.slots.tolist(), self.accept_lens.tolist()):
                torch.testing.assert_close(
                    snapshots[s, n - 1],
                    fold_state[0, s],
                    rtol=0,
                    atol=FP32_ATOL,
                    msg=f"{dtype=} slot={s} accept_len={n}",
                )
            untouched = set(range(NUM_SLOTS)) - set(self.slots.tolist())
            for s in untouched:
                self.assertTrue(torch.equal(fold_state[0, s], state[s]))

    def test_track_store_and_null_slots(self):
        dtype = torch.float32
        state = self._state(dtype)
        inputs = _make_window(33)

        snapshots = torch.zeros(NUM_SLOTS, T, HV, K, V, device=DEVICE, dtype=dtype)
        _run_verify(inputs, self.gating, state.clone(), self.slots, snapshots=snapshots)

        fold_state = state.clone().unsqueeze(0).contiguous()
        rings = _make_rings()
        _run_verify(inputs, self.gating, fold_state[0], self.slots, rings=rings)

        track_slots = torch.tensor([1, 0, 3], dtype=torch.int64, device=DEVICE)
        track_steps = torch.tensor([1, -1, 2], dtype=torch.int64, device=DEVICE)
        slots_with_null = self.slots.clone()
        slots_with_null[1] = -1
        _fold(
            fold_state,
            rings,
            slots_with_null,
            self.accept_lens,
            track_slots=track_slots,
            track_steps=track_steps,
        )

        torch.testing.assert_close(
            fold_state[0, 1], snapshots[5, 1], rtol=0, atol=FP32_ATOL
        )
        torch.testing.assert_close(
            fold_state[0, 3], snapshots[7, 2], rtol=0, atol=FP32_ATOL
        )
        # Row 1's state slot is replaced with -1, and its track step is -1, so
        # neither its original state slot 2 nor tracking slot 0 is written.
        self.assertTrue(torch.equal(fold_state[0, 2], state[2]))
        self.assertTrue(torch.equal(fold_state[0, 0], state[0]))

    def test_long_chain_error_stays_bounded(self):
        """This regression case remains within FP32_ATOL through 256 commits."""
        num_iters = 256
        for dtype in (torch.float32, torch.bfloat16):
            base_state = self._state(dtype)
            fold_state = base_state.clone().unsqueeze(0).contiguous()
            snapshots = torch.zeros(NUM_SLOTS, T, HV, K, V, device=DEVICE, dtype=dtype)
            gen = torch.Generator().manual_seed(7)
            for it in range(num_iters):
                inputs = _make_window(1000 + it)
                accept_lens = torch.randint(1, T + 1, (B,), generator=gen).to(
                    device=DEVICE, dtype=torch.int32
                )

                out_base = _run_verify(
                    inputs, self.gating, base_state, self.slots, snapshots=snapshots
                )
                for s, n in zip(self.slots.tolist(), accept_lens.tolist()):
                    base_state[s] = snapshots[s, n - 1]

                rings = _make_rings()
                out_fold = _run_verify(
                    inputs, self.gating, fold_state[0], self.slots, rings=rings
                )
                _fold(fold_state, rings, self.slots, accept_lens)

                torch.testing.assert_close(
                    out_base,
                    out_fold,
                    rtol=0,
                    atol=FP32_ATOL,
                    msg=f"{dtype=} {it=}",
                )
                torch.testing.assert_close(
                    base_state,
                    fold_state[0],
                    rtol=0,
                    atol=FP32_ATOL,
                    msg=f"{dtype=} {it=}",
                )


if __name__ == "__main__":
    unittest.main()
