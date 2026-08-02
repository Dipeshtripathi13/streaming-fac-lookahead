"""Latency decomposition and instrumentation.

The central methodological claim of this project is that a single end-to-end
latency number is not actionable. We always report:

    t_algorithmic = chunk_size + lookahead          (hardware-independent)
    t_compute     = encoder + conversion + vocoder  (hardware-dependent)
    t_buffer      = I/O + jitter buffer             (system-dependent)
    t_end_to_end  = t_algorithmic + t_compute + t_buffer

t_algorithmic is *inherent*: it is the delay you would suffer on an infinitely
fast machine, because the model refuses to emit output frame n until it has
seen input frame n + L. No amount of hardware fixes it.

t_compute is *contingent*: quantise, prune, buy a faster chip.

Reporting them fused is what lets a paper say "241 ms" without the reader
knowing whether that is a modelling choice or an implementation accident.
"""

from __future__ import annotations

import json
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterator, List, Optional


# --------------------------------------------------------------------------
# Stage timing
# --------------------------------------------------------------------------

class StageTimer:
    """Accumulates per-stage wall-clock samples across a streaming run.

    Uses time.perf_counter_ns() -- monotonic, highest resolution the platform
    offers.  Do NOT use time.time() for this; it is subject to NTP steps and
    on some platforms has ~1 ms granularity, which is the same order as the
    quantities we are trying to measure.
    """

    def __init__(self) -> None:
        self._samples: Dict[str, List[float]] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            dt_ms = (time.perf_counter_ns() - t0) / 1e6
            self._samples.setdefault(name, []).append(dt_ms)

    def record(self, name: str, ms: float) -> None:
        self._samples.setdefault(name, []).append(ms)

    def reset(self) -> None:
        self._samples.clear()

    @property
    def stages(self) -> List[str]:
        return list(self._samples)

    def samples(self, name: str) -> List[float]:
        return list(self._samples.get(name, []))

    def summary(self, drop_warmup: int = 3) -> Dict[str, Dict[str, float]]:
        """Per-stage p50/p90/p95/p99/mean/max in ms.

        `drop_warmup` discards the first N samples of every stage.  The first
        call into ONNX Runtime or PyTorch pays lazy allocation, kernel
        selection and page-fault costs that are not representative of steady
        state.  Three is empirically enough for ORT; report the value you used.
        """
        out: Dict[str, Dict[str, float]] = {}
        for name, xs in self._samples.items():
            xs = xs[drop_warmup:] if len(xs) > drop_warmup else xs
            if not xs:
                continue
            out[name] = _dist(xs)
        return out


def _dist(xs: List[float]) -> Dict[str, float]:
    s = sorted(xs)
    n = len(s)

    def q(p: float) -> float:
        if n == 1:
            return s[0]
        # nearest-rank; explicit so the paper can state the estimator used
        k = min(n - 1, max(0, int(round(p * (n - 1)))))
        return s[k]

    return {
        "n": float(n),
        "mean": statistics.fmean(s),
        "p50": q(0.50),
        "p90": q(0.90),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": s[-1],
        "stdev": statistics.pstdev(s) if n > 1 else 0.0,
    }


# --------------------------------------------------------------------------
# The decomposition itself
# --------------------------------------------------------------------------

@dataclass
class LatencyBudget:
    """One row of every table in the paper."""

    chunk_ms: float
    lookahead_ms: float
    compute_ms_p50: float
    compute_ms_p95: float
    buffer_ms: float = 0.0
    label: str = ""
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def algorithmic_ms(self) -> float:
        return self.chunk_ms + self.lookahead_ms

    @property
    def end_to_end_p50(self) -> float:
        return self.algorithmic_ms + self.compute_ms_p50 + self.buffer_ms

    @property
    def end_to_end_p95(self) -> float:
        return self.algorithmic_ms + self.compute_ms_p95 + self.buffer_ms

    @property
    def rtf_p50(self) -> float:
        """Real-time factor per chunk. Must be < 1.0 or the stream falls behind.

        This is the hard feasibility gate: RTF >= 1 means the buffer grows
        without bound and end-to-end latency is unbounded regardless of the
        algorithmic budget.
        """
        return self.compute_ms_p50 / self.chunk_ms if self.chunk_ms else float("inf")

    @property
    def rtf_p95(self) -> float:
        return self.compute_ms_p95 / self.chunk_ms if self.chunk_ms else float("inf")

    @property
    def feasible(self) -> bool:
        """p95 RTF < 1 with 20% headroom -- the deployability criterion."""
        return self.rtf_p95 < 0.8

    def to_row(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "chunk_ms": self.chunk_ms,
            "lookahead_ms": self.lookahead_ms,
            "t_algorithmic_ms": round(self.algorithmic_ms, 2),
            "t_compute_p50_ms": round(self.compute_ms_p50, 2),
            "t_compute_p95_ms": round(self.compute_ms_p95, 2),
            "t_buffer_ms": round(self.buffer_ms, 2),
            "t_e2e_p50_ms": round(self.end_to_end_p50, 2),
            "t_e2e_p95_ms": round(self.end_to_end_p95, 2),
            "rtf_p50": round(self.rtf_p50, 4),
            "rtf_p95": round(self.rtf_p95, 4),
            "feasible": self.feasible,
            **{f"meta.{k}": v for k, v in self.meta.items()},
        }


def budget_from_timer(
    timer: StageTimer,
    chunk_ms: float,
    lookahead_ms: float,
    compute_stages: Optional[List[str]] = None,
    buffer_ms: float = 0.0,
    label: str = "",
    meta: Optional[Dict[str, object]] = None,
    drop_warmup: int = 3,
) -> LatencyBudget:
    """Fold a StageTimer into a LatencyBudget.

    `compute_stages` selects which stages count toward t_compute; anything
    else recorded (e.g. "io", "resample") is excluded so you can time
    diagnostics without polluting the headline number.

    Note on percentile addition: summing per-stage p95s is conservative (it
    assumes worst-case co-occurrence). We instead sum per-chunk totals and
    take the percentile of the *sum*, which is the quantity a user actually
    experiences. That requires equal sample counts across stages -- enforced
    below, with a fallback to the conservative sum if they differ.
    """
    summ = timer.summary(drop_warmup=drop_warmup)
    stages = compute_stages if compute_stages is not None else list(summ)
    present = [s for s in stages if s in summ]

    series = [timer.samples(s)[drop_warmup:] for s in present]
    if series and len({len(x) for x in series}) == 1 and series[0]:
        totals = [sum(vals) for vals in zip(*series)]
        d = _dist(totals)
        p50, p95 = d["p50"], d["p95"]
    else:  # ragged -- fall back to conservative per-stage sum
        p50 = sum(summ[s]["p50"] for s in present)
        p95 = sum(summ[s]["p95"] for s in present)

    m = dict(meta or {})
    m["per_stage"] = {s: {k: round(v, 3) for k, v in summ[s].items()} for s in present}
    return LatencyBudget(
        chunk_ms=chunk_ms,
        lookahead_ms=lookahead_ms,
        compute_ms_p50=p50,
        compute_ms_p95=p95,
        buffer_ms=buffer_ms,
        label=label,
        meta=m,
    )


# --------------------------------------------------------------------------
# Result serialisation
# --------------------------------------------------------------------------

def write_jsonl(path: str, rows: List[Dict[str, object]]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    import csv
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
