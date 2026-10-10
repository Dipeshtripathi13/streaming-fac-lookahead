"""Guard the experimental invariants that the paper's validity rests on.

Runs without torch (the torch-dependent tests self-skip), so it can run on
a Raspberry Pi or a bare clone.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sfac.pipeline import (  # noqa: E402
    HarnessConfig, Mode, assert_only_L_varies, sweep_configs,
)


def test_fingerprint_excludes_only_lookahead():
    a = HarnessConfig(lookahead_ms=0)
    b = HarnessConfig(lookahead_ms=640)
    assert a.fingerprint() == b.fingerprint()
    assert_only_L_varies([a, b])
    print("ok  fingerprint isolates L")


def test_confound_is_caught():
    a = HarnessConfig(lookahead_ms=0, chunk_ms=20)
    b = HarnessConfig(lookahead_ms=80, chunk_ms=40)   # chunk changed too
    try:
        assert_only_L_varies([a, b])
    except ValueError as e:
        assert "chunk_ms" in str(e)
        print("ok  confounded sweep is rejected:", str(e)[:70] + "...")
        return
    raise AssertionError("a confounded sweep was accepted -- the guard is broken")


def test_mode_change_is_a_confound_within_a_group():
    a = HarnessConfig(lookahead_ms=0, mode=Mode.AC)
    b = HarnessConfig(lookahead_ms=80, mode=Mode.VC_ONLY)
    try:
        assert_only_L_varies([a, b])
    except ValueError as e:
        assert "mode" in str(e)
        print("ok  mode is held constant within an L-sweep group")
        return
    raise AssertionError("mode change was not flagged")


def test_duplicate_lookaheads_rejected():
    try:
        assert_only_L_varies([HarnessConfig(lookahead_ms=80),
                              HarnessConfig(lookahead_ms=80)])
    except ValueError as e:
        assert "duplicate" in str(e)
        print("ok  duplicate L rejected")
        return
    raise AssertionError("duplicate L accepted")


def test_vocoder_hop_frame_rate_consistency():
    """A hop/frame_ms mismatch relabels every condition. Must fail loudly."""
    try:
        HarnessConfig(frame_ms=20.0, vocoder_hop=256)
    except ValueError as e:
        assert "frame_ms" in str(e)
        print("ok  vocoder hop / frame rate mismatch is rejected")
        return
    raise AssertionError("inconsistent hop accepted")


def test_full_sweep_is_clean():
    cfgs = sweep_configs()
    assert len(cfgs) == 14
    ac = [c for c in cfgs if c.mode is Mode.AC]
    vc = [c for c in cfgs if c.mode is Mode.VC_ONLY]
    assert len(ac) == len(vc) == 7
    # the two arms must be identical except for mode
    for x, y in zip(ac, vc):
        assert x.lookahead_ms == y.lookahead_ms
        fx, fy = x.fingerprint(), y.fingerprint()
        fx.pop("mode"), fy.pop("mode")
        assert fx == fy, "AC and VC-only arms differ in more than mode"
    print("ok  full 7x2 sweep is unconfounded")


def test_geometry_matches_config():
    c = HarnessConfig(lookahead_ms=160, chunk_ms=40, lookback_ms=2000)
    g = c.geometry
    assert g.lookahead_frames == 8 and g.chunk_frames == 2
    assert g.lookback_frames == 100
    assert g.algorithmic_ms == 200
    print("ok  config -> geometry")


def test_torch_model_masks_are_L_dependent_only():
    try:
        import torch  # noqa
    except Exception as e:
        # Not just ImportError: a CUDA-linked wheel on a machine with no CUDA
        # libraries raises ValueError at import. That is an environment
        # problem, not a repo failure, and it must not fail the suite on a Pi
        # or in a CPU-only sandbox.
        print(f"skip torch model test ({type(e).__name__}: {str(e)[:60]})")
        return
    from sfac.pipeline import build_modules
    m0, i0 = build_modules(HarnessConfig(lookahead_ms=0))
    m6, i6 = build_modules(HarnessConfig(lookahead_ms=640))
    assert i0["params"] == i6["params"], "param count must not depend on L"
    sd0, sd6 = m0.state_dict(), m6.state_dict()
    assert set(sd0) == set(sd6)
    for k in sd0:
        assert torch.equal(sd0[k], sd6[k]), f"seeded init differs at {k}"
    T = 32
    a0 = m0.additive_mask(T, torch.device("cpu"), torch.float32)
    a6 = m6.additive_mask(T, torch.device("cpu"), torch.float32)
    assert (a6 == 0).sum() > (a0 == 0).sum(), "larger L must attend to more"
    print("ok  torch: identical weights, wider mask")


def test_probe_lookahead_does_not_compound():
    """The probe's future context must equal its label, not grow with depth.

    Change the input from frame F on and find the first output frame that
    moves. With the lookahead in the first block only, that frame is the one
    the mask predicts. With it in every block, the reach is several times the
    label, which is the error the first arXiv version carried.
    """
    try:
        import torch
    except Exception as e:
        print(f"skip probe lookahead test ({type(e).__name__}: {str(e)[:60]})")
        return
    from sfac.translator import TranslatorConfig, PhoneVocab, build_translator
    vocab = PhoneVocab([f"p{i}" for i in range(36)])
    T, F = 300, 200
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, T, 768, generator=g)
    y = x.clone()
    y[:, F:] = torch.randn(1, T - F, 768, generator=g)

    def first_moved(scope, L):
        cfg = TranslatorConfig(lookahead_ms=L, lookahead_scope=scope)
        m, _ = build_translator(cfg, vocab)
        m.eval()
        with torch.no_grad():
            d = (m(x) - m(y))[0].abs().amax(-1)
        return int(torch.nonzero(d > 1e-5)[0]), cfg.geometry

    for L in (0, 20, 40, 60, 160, 340):
        first, geo = first_moved("end_to_end", L)
        C, NL = geo.chunk_frames, geo.lookahead_frames
        want = next(i for i in range(F + 1) if (i // C + 1) * C - 1 + NL >= F)
        assert first == want, f"L={L}: first moved frame {first}, mask says {want}"
    e2e, _ = first_moved("end_to_end", 40)
    per, _ = first_moved("per_layer", 40)
    assert per < e2e, "per_layer should reach further ahead than its label"
    print(f"ok  probe lookahead exact end-to-end; per_layer at L=40 ms reaches "
          f"{(F - per) * 20} ms against {(F - e2e) * 20} ms")


if __name__ == "__main__":
    fns =[v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{len(fns)} tests passed")
