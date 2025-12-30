import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median


@dataclass(frozen=True)
class Row:
    video: str
    before_len: int
    after_len: int
    ratio: float


def _safe_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


def load_rows(path: Path) -> list[Row]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[Row] = []
    for item in data:
        video = str(item.get("video", ""))
        before_len = int(item.get("before_len", 0))
        after_len = int(item.get("after_len", 0))
        ratio = _safe_float(item.get("ratio", (after_len / max(1, before_len))), default=float("nan"))
        rows.append(Row(video=video, before_len=before_len, after_len=after_len, ratio=ratio))
    return rows


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401

        return True
    except Exception:
        return False


def write_report(rows: list[Row], input_path: Path, out_dir: Path) -> Path:
    ratios = [r.ratio for r in rows if not math.isnan(r.ratio)]
    ratios_sorted = sorted(ratios)
    before = [r.before_len for r in rows]
    after = [r.after_len for r in rows]

    def fmt(x):
        if isinstance(x, float):
            if math.isnan(x):
                return "nan"
            return f"{x:.2f}"
        return str(x)

    top = sorted(rows, key=lambda r: (-(r.ratio if not math.isnan(r.ratio) else -1), -r.after_len))[:10]
    bottom = sorted(rows, key=lambda r: ((r.ratio if not math.isnan(r.ratio) else float("inf")), r.after_len))[:10]

    lines: list[str] = []
    lines.append("# Sponge Attack Summary\n")
    lines.append(f"- Input: `{input_path.as_posix()}`")
    lines.append(f"- Videos: {len(rows)}")
    lines.append(f"- Mean ratio (after/before): {fmt(mean(ratios)) if ratios else 'n/a'}")
    lines.append(f"- Median ratio: {fmt(median(ratios)) if ratios else 'n/a'}")
    lines.append(
        "- Ratio percentiles (P10/P25/P50/P75/P90): "
        + "/".join(fmt(percentile(ratios_sorted, p)) for p in (10, 25, 50, 75, 90))
    )
    lines.append(f"- Mean lengths (before→after): {fmt(mean(before))} → {fmt(mean(after))}\n")

    lines.append("## Visualizations\n")
    lines.append(f"- Histogram: `{(out_dir / 'ratio_hist.png').as_posix()}`")
    lines.append(f"- Scatter (before vs after): `{(out_dir / 'before_after_scatter.png').as_posix()}`\n")

    lines.append("## Top-10 (largest ratio)\n")
    lines.append("| video | before_len | after_len | ratio |")
    lines.append("|---|---:|---:|---:|")
    for r in top:
        lines.append(f"| {r.video} | {r.before_len} | {r.after_len} | {fmt(r.ratio)} |")
    lines.append("")

    lines.append("## Bottom-10 (smallest ratio)\n")
    lines.append("| video | before_len | after_len | ratio |")
    lines.append("|---|---:|---:|---:|")
    for r in bottom:
        lines.append(f"| {r.video} | {r.before_len} | {r.after_len} | {fmt(r.ratio)} |")
    lines.append("")

    out_path = out_dir / "sponge_summary_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def plot(rows: list[Row], out_dir: Path) -> None:
    if not try_import_matplotlib():
        print(
            "matplotlib not available; skipping plots. Install with: pip install matplotlib\n"
            "Report markdown will still be generated."
        )
        return

    import matplotlib.pyplot as plt

    ratios = [r.ratio for r in rows if not math.isnan(r.ratio)]
    before = [r.before_len for r in rows]
    after = [r.after_len for r in rows]

    # Histogram of ratios
    plt.figure(figsize=(8, 4.5))
    plt.hist(ratios, bins=30)
    plt.title("Sponge Attack: after_len / before_len")
    plt.xlabel("ratio")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "ratio_hist.png", dpi=180)
    plt.close()

    # Scatter before vs after
    plt.figure(figsize=(6, 6))
    plt.scatter(before, after, s=18, alpha=0.7)
    max_axis = max(max(before, default=0), max(after, default=0), 1)
    plt.plot([0, max_axis], [0, max_axis], linestyle="--", linewidth=1)
    plt.title("Sponge Attack: before_len vs after_len")
    plt.xlabel("before_len")
    plt.ylabel("after_len")
    plt.tight_layout()
    plt.savefig(out_dir / "before_after_scatter.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to final_summary.json (list of per-video logs)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Directory to write plots + report",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows = load_rows(input_path)
    report_path = write_report(rows, input_path, out_dir)
    plot(rows, out_dir)
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()

