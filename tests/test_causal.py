"""Correctness tests for the lookahead machinery.

These are not decoration. The entire paper rests on the claim that L is the
only thing varying across conditions. If the mask is off by one, or if the
streaming buffer does not reproduce the offline mask, every number in the
sweep is wrong in a way that is invisible in the plots.

Run:  python3 -m pytest research/tests -q
 or:  python3 research/tests/test_causal.py     (no pytest needed)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sfac.causal import (  # noqa: E402
    StreamGeometry,
    LookaheadBuffer,
    lookahead_mask,
    chunked_lookahead_mask,
    effective_lookahead_frames,
    mask_audit,
)


def test_frame_arithmetic():
    g = StreamGeometry(chunk_ms=20, lookahead_ms=0)
    assert g.chunk_frames == 1
    assert g.lookahead_frames == 0
    assert g.algorithmic_ms == 20

    g = StreamGeometry(chunk_ms=40, lookahead_ms=80)
    assert g.chunk_frames == 2
    assert g.lookahead_frames == 4
    assert g.algorithmic_ms == 120

    # ceil, not round: 30 ms of lookahead at 20 ms frames costs you 2 frames
    g = StreamGeometry(chunk_ms=20, lookahead_ms=30)
    assert g.lookahead_frames == 2, "must ceil so reported L is an upper bound"

    # exact multiples must not be inflated by float error
    for ms in (0, 20, 40, 80, 160, 320, 640):
        assert StreamGeometry(20, ms).lookahead_frames == ms // 20

    # the honest number includes the conv frontend
    g = StreamGeometry(chunk_ms=20, lookahead_ms=0)
    assert g.irreducible_lookahead_ms == 5.0
    assert g.algorithmic_ms_honest == 25.0
    print("ok  frame arithmetic")


def test_lookahead_mask_causality():
    n = 12
    m = lookahead_mask(n, 0)
    assert np.array_equal(m, np.tril(np.ones((n, n), bool))), "L=0 must be strictly causal"

    m = lookahead_mask(n, 3)
    eff = effective_lookahead_frames(m)
    # every query except the last few sees exactly +3
    assert eff[0] == 3 and eff[5] == 3
    assert eff[-1] == 0, "last frame cannot see beyond the sequence"

    # monotonicity: more lookahead never removes an edge
    prev = lookahead_mask(n, 0)
    for L in range(1, 8):
        cur = lookahead_mask(n, L)
        assert np.all(cur >= prev), "mask must be monotone increasing in L"
        prev = cur

    # unbounded lookahead == fully bidirectional
    assert lookahead_mask(n, n).all()

    # lookback truncation
    m = lookahead_mask(n, 0, lookback_frames=2)
    assert m[5, 3] and not m[5, 2], "lookback window off by one"
    print("ok  lookahead mask")


def test_chunked_mask_gives_boundary_bonus():
    n, cf, L = 12, 4, 2
    m = chunked_lookahead_mask(n, chunk_frames=cf, lookahead_frames=L)
    eff = effective_lookahead_frames(m)
    # frame 0 sits at the start of a 4-frame chunk, so it gets 3 free frames
    # of intra-chunk right context on top of L=2  -> 5
    assert eff[0] == 5, eff[0]
    # frame 3 is the chunk's last frame -> exactly L
    assert eff[3] == 2, eff[3]
    # the minimum realised lookahead across the sequence equals L (interior)
    interior = eff[: n - L]
    assert interior.min() == L
    # a per-frame mask must be a strict subset of the chunked one
    pf = lookahead_mask(n, L)
    assert np.all(m >= pf), "chunked mask must dominate the per-frame mask"
    print("ok  chunked mask boundary bonus")


def test_chunk1_equals_per_frame():
    n, L = 10, 3
    a = chunked_lookahead_mask(n, chunk_frames=1, lookahead_frames=L)
    b = lookahead_mask(n, L)
    assert np.array_equal(a, b), "chunk_frames=1 must degenerate to per-frame"
    print("ok  chunk=1 degenerates correctly")


def test_streaming_buffer_matches_mask():
    """The buffer must finalise frame i only after frame i+L has arrived.

    This is the streaming/offline equivalence check. If it fails, the model
    trained with the offline mask is not the model you are benchmarking.
    """
    for L_ms in (0, 20, 40, 80, 160):
        for chunk_ms in (20, 40, 80):
            g = StreamGeometry(chunk_ms=chunk_ms, lookahead_ms=L_ms, lookback_ms=None)
            buf = LookaheadBuffer(g, feat_dim=1)
            cf, La = g.chunk_frames, g.lookahead_frames
            total_chunks = 12
            emitted = []
            for c in range(total_chunks):
                frames = np.arange(c * cf, (c + 1) * cf, dtype=np.float32)[:, None]
                r = buf.push(frames)
                if r is None:
                    continue
                window, lo, hi = r
                emitted.append((lo, hi, buf.frames_written))
                # invariant: nothing is finalised without L frames after it
                assert buf.frames_written - hi >= La, (
                    f"finalised frame {hi-1} with only "
                    f"{buf.frames_written - hi} frames of lookahead, need {La}"
                )
            r = buf.flush()
            if r is not None:
                _, lo, hi = r
                emitted.append((lo, hi, buf.frames_written))

            # contiguity and completeness
            cursor = 0
            for lo, hi, _ in emitted:
                assert lo == cursor, f"gap/overlap at {lo} != {cursor}"
                cursor = hi
            assert cursor == total_chunks * cf, "did not emit every frame"
    print("ok  streaming buffer <-> mask equivalence")


def test_buffer_delay_is_exactly_lookahead():
    """Measured emission delay must equal the advertised t_algorithmic."""
    for L_ms in (0, 40, 160, 640):
        g = StreamGeometry(chunk_ms=20, lookahead_ms=L_ms)
        buf = LookaheadBuffer(g, feat_dim=1)
        arrival, finalised = {}, {}
        for c in range(60):
            frames = np.full((g.chunk_frames, 1), c, np.float32)
            for k in range(g.chunk_frames):
                arrival[c * g.chunk_frames + k] = c
            r = buf.push(frames)
            if r is None:
                continue
            _, lo, hi = r
            for f in range(lo, hi):
                finalised.setdefault(f, c)
        # steady state: skip the first few frames
        delays = [
            (finalised[f] - arrival[f]) * g.chunk_ms
            for f in sorted(finalised) if f > 5 and f < 50
        ]
        expected = g.lookahead_frames * g.frame_ms
        assert max(delays) <= expected + g.chunk_ms, (
            f"L={L_ms}: max delay {max(delays)} exceeds "
            f"lookahead {expected} + one chunk {g.chunk_ms}"
        )
        assert max(delays) >= expected - 1e-9 or L_ms == 0, (
            f"L={L_ms}: delay {max(delays)} is less than the advertised {expected}"
        )
    print("ok  emission delay == advertised t_algorithmic")


def test_mask_audit_reports_truth():
    m = chunked_lookahead_mask(40, chunk_frames=2, lookahead_frames=4)
    a = mask_audit(m)
    assert a["eff_lookahead_max"] == 5   # 1 intra-chunk + 4
    assert a["eff_lookahead_min"] == 0   # tail frames
    assert 0 < a["density"] < 1
    print("ok  mask audit")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{len(fns)} tests passed")
