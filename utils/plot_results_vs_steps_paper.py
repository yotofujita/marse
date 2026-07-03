from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple, Sequence

import matplotlib.pyplot as plt


# Keep one fixed style per method across all CSVs
STYLES = {
    "C-NAR":      dict(color="#1f77b4", marker="o", linestyle="-"),
    "C-AR":        dict(color="#ff7f0e", marker="s", linestyle="--"),
    #"MG-Disc":   dict(color="#2ca02c", marker="X", linestyle="-"),
    "MARSE-causal":     dict(color="#d62728", marker="D", linestyle="--"),
    "MARSE-NC-random": dict(color="#9467bd", marker="v", linestyle=":"),
    "MARSE-NC-oracle": dict(color="#8c564b", marker="^", linestyle="-."),
}


def load_csv_series(csv_path: Path) -> Dict[str, List[Tuple[int, float]]]:
    """
    Read one CSV and return:
        {
            "Disc": [(1, 3.3679), ...],
            "AR": [(50, 3.4155), ...],
            ...
        }
    using DNSMOS_P835_OVRL as the y value.
    """
    result: Dict[str, List[Tuple[int, float]]] = {}

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"].strip()
            step = int(row["step"])
            ovrl = float(row["DNSMOS_P835_OVRL"])

            result.setdefault(label, []).append((step, ovrl))

    for label in result:
        result[label].sort(key=lambda x: x[0])

    return result


def plot_ovrl_from_csvs(
    csv_paths: Sequence[Path],
    output_path: Path,
    method_order: Sequence[str] | None = None,
) -> None:
    """
    Plot OVRL curves from multiple CSV files.

    Each CSV is drawn in its own subplot.
    The same method always uses the same style across all CSVs.
    """
    if not csv_paths:
        raise ValueError("csv_paths is empty")

    all_data = [load_csv_series(path) for path in csv_paths]

    if method_order is None:
        seen = []
        for data in all_data:
            for method in data:
                if method not in seen:
                    seen.append(method)
        method_order = seen

    nrows = 1
    fig, axes = plt.subplots(nrows, 1, figsize=(6.0, 3.0 * nrows))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, csv_path, data in zip(axes, csv_paths, all_data):
        for method in method_order:
            if method not in data:
                continue

            series = data[method]
            xs = [step for step, _ in series]
            ys = [ovrl for _, ovrl in series]

            style = STYLES.get(method, {"marker": "o", "linestyle": "-"})

            if method in {"C-NAR", "C-AR"}:
                ax.plot(
                    xs,
                    ys,
                    alpha=0.9,
                    linewidth=2,
                    markersize=10,
                    label=method,
                    **style,
                )
            else:
                ax.plot(
                    xs,
                    ys,
                    alpha=0.7,
                    linewidth=2,
                    label=method,
                    **style,
                )

        ax.set_xlabel("Number of decoding steps")
        ax.set_ylabel("DNSMOS OVRL")
        ax.set_xticks([1, 5, 10, 20, 30, 40, 50])
        ax.grid(True, alpha=0.3)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc=(0.41, 0.20), ncol=2, fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    csv_files = [
        Path("/home/yfujita/research/iwaenc2026/outputs/results_vs_steps_librimix_paper_table.csv"),
        # Path("/home/yfujita/research/iwaenc2026/outputs/results_vs_steps_demand_paper_table.csv"),
    ]

    plot_ovrl_from_csvs(
        csv_paths=csv_files,
        output_path=Path("outputs/ovrl_comparison.pdf"),
        method_order=["C-NAR", "C-AR", "MARSE-causal", "MARSE-NC-random", "MARSE-NC-oracle"],  # optional but keeps legend/order fixed
    )