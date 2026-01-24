#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _stats(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    mid = n // 2
    median = xs_sorted[mid] if n % 2 == 1 else 0.5 * (xs_sorted[mid - 1] + xs_sorted[mid])
    p95_i = int(round(0.95 * (n - 1)))
    return {
        "mean": sum(xs_sorted) / n,
        "median": median,
        "p95": xs_sorted[p95_i],
    }


def _get(d: Dict[str, Any], k: str) -> Optional[Any]:
    return d.get(k, None)


@dataclass(frozen=True)
class RunSummary:
    name: str
    run_dir: Path
    n: int
    token_ratio_mean: float
    after_tokens_mean: float
    gen_before_s_mean: float
    gen_after_s_mean: float
    overhead_s_mean: float


def summarize_run(run_dir: Path, name: str) -> Dict[str, Any]:
    final_path = run_dir / "final_summary_eval.json"
    timing_path = run_dir / "timing_summary_eval.json"

    if not final_path.exists():
        raise FileNotFoundError(f"Missing: {final_path}")
    if not timing_path.exists():
        raise FileNotFoundError(f"Missing: {timing_path}")

    rows = _read_json(final_path)
    timing = _read_json(timing_path)

    # Prefer new token fields (from updated 3_unified_sponge.py); fall back to char ratios.
    token_ratios: List[float] = []
    after_tokens: List[float] = []
    gen_before_s: List[float] = []
    gen_after_s: List[float] = []
    overhead_s: List[float] = []

    for r in rows:
        if isinstance(r, dict):
            if "token_ratio" in r:
                token_ratios.append(_safe_float(r.get("token_ratio", 0.0)))
            elif "ratio" in r:
                token_ratios.append(_safe_float(r.get("ratio", 0.0)))
            if "after_new_tokens" in r:
                after_tokens.append(_safe_float(r.get("after_new_tokens", 0.0)))
            if "gen_before_s" in r:
                gen_before_s.append(_safe_float(r.get("gen_before_s", 0.0)))
            if "gen_after_s" in r:
                gen_after_s.append(_safe_float(r.get("gen_after_s", 0.0)))

    # Timing summary always has overhead stats
    overhead_mean = _safe_float(_get(timing.get("stats_overhead_s", {}), "mean"), 0.0)
    if not overhead_s:
        overhead_s = [overhead_mean]

    out = {
        "name": name,
        "run_dir": str(run_dir),
        "n_eval_processed": int(timing.get("n_eval_processed", len(rows))),
        "token_ratio": _stats(token_ratios),
        "after_new_tokens": _stats(after_tokens),
        "gen_before_s": _stats(gen_before_s),
        "gen_after_s": _stats(gen_after_s),
        "overhead_s": timing.get("stats_overhead_s", {}),
        "timing_summary_eval": timing,
    }
    return out


def format_table(summaries: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("| run | n | token_ratio_mean | after_tokens_mean | gen_before_s_mean | gen_after_s_mean | overhead_s_mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        tr = s.get("token_ratio", {}).get("mean", 0.0)
        at = s.get("after_new_tokens", {}).get("mean", 0.0)
        gb = s.get("gen_before_s", {}).get("mean", 0.0)
        ga = s.get("gen_after_s", {}).get("mean", 0.0)
        oh = s.get("overhead_s", {}).get("mean", 0.0)
        n = int(s.get("n_eval_processed", 0))
        lines.append(
            f"| {s.get('name','')} | {n} | {tr:.4f} | {at:.1f} | {gb:.3f} | {ga:.3f} | {oh:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", nargs=2, metavar=("NAME", "DIR"), required=True,
                        help="Add a run: --run bddx_to_dd results_cross_bddx_to_dd")
    parser.add_argument("--out", type=str, default=None, help="Write a JSON+MD report under this directory.")
    args = parser.parse_args()

    summaries: List[Dict[str, Any]] = []
    for name, run_dir in args.run:
        summaries.append(summarize_run(Path(run_dir), name=name))

    table = format_table(summaries)
    print(table)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "cross_dataset_eval_summary.json").write_text(
            json.dumps({"runs": summaries}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (out_dir / "cross_dataset_eval_summary.md").write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()

