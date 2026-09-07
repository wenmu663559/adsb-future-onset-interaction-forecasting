"""Build manuscript figures from the frozen KBTP and KAGC result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "reports" / "references"
FIGURE_DIR = ROOT / "paper_rewriting_output" / "final_paper" / "figures"
HORIZONS = ("30", "120", "300")


def _load(name: str) -> dict:
    with (REFERENCE_DIR / name).open(encoding="utf-8") as stream:
        return json.load(stream)


def _values(result: dict, comparison: str, field: str) -> list[float]:
    return [result["horizons"][h][comparison][field] for h in HORIZONS]


def _finish(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _grouped_bars(ax, series, colors, ylabel, title, decimals=3) -> None:
    x = np.arange(len(HORIZONS), dtype=float)
    width = 0.22 if len(series) == 3 else 0.28
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * width
    for offset, (label, values), color in zip(offsets, series, colors):
        bars = ax.bar(x + offset, values, width, label=label, color=color, edgecolor="white", linewidth=0.7)
        ax.bar_label(bars, labels=[f"{v:.{decimals}f}" for v in values], padding=3, fontsize=7)
    ax.set_xticks(x, [f"{h} s" for h in HORIZONS])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="semibold")
    ax.grid(axis="y", color="#D9E2EC", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=len(series), loc="upper left")
    ax.margins(y=0.18)


def build_kbtp(result: dict) -> None:
    b_count = _values(result, "B_vs_A_count_MAE", "candidate_equal_day_mean")
    a_count = _values(result, "B_vs_A_count_MAE", "baseline_equal_day_mean")
    pair_count = _values(result, "B_vs_pair_exposure_count_MAE", "baseline_equal_day_mean")
    b_brier = _values(result, "B_logistic_vs_prevalence_Brier", "candidate_equal_day_mean")
    prevalence = _values(result, "B_logistic_vs_prevalence_Brier", "baseline_equal_day_mean")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.35), constrained_layout=True)
    _grouped_bars(
        axes[0],
        [("Pairwise B", b_count), ("Aggregate A", a_count), ("Pair-rate reference", pair_count)],
        ["#79A9DC", "#B8C7D9", "#F3C88B"],
        "Equal-group MAE (lower is better)",
        "(a) Future-onset count",
    )
    _grouped_bars(
        axes[1],
        [("Pairwise B", b_brier), ("Training prevalence", prevalence)],
        ["#79A9DC", "#F3C88B"],
        "Equal-group Brier score (lower is better)",
        "(b) High-interaction probability",
    )
    fig.suptitle("One-shot KBTP confirmation on eight unseen source groups", fontsize=12, fontweight="semibold")
    _finish(fig, "figure_1_kbtp_confirmation")


def build_kagc(result: dict) -> None:
    b_count = _values(result, "B_vs_A_count_MAE", "candidate_equal_day_mean")
    a_count = _values(result, "B_vs_A_count_MAE", "baseline_equal_day_mean")
    pair_count = _values(result, "B_vs_KBTP_pair_exposure_count_MAE", "baseline_equal_day_mean")
    b_brier = _values(result, "B_logistic_vs_KBTP_prevalence_Brier", "candidate_equal_day_mean")
    prevalence = _values(result, "B_logistic_vs_KBTP_prevalence_Brier", "baseline_equal_day_mean")
    passed = [result["horizons"][h]["passed_external_gate"] for h in HORIZONS]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.35), constrained_layout=True)
    _grouped_bars(
        axes[0],
        [("Frozen pairwise B", b_count), ("Aggregate A", a_count), ("Pair-rate reference", pair_count)],
        ["#79A9DC", "#B8C7D9", "#F3C88B"],
        "Equal-group MAE (lower is better)",
        "(a) Future-onset count",
    )
    _grouped_bars(
        axes[1],
        [("Frozen pairwise B", b_brier), ("KBTP prevalence", prevalence)],
        ["#79A9DC", "#F3C88B"],
        "Equal-group Brier score (lower is better)",
        "(b) High-interaction probability",
        decimals=4,
    )
    for idx, ok in enumerate(passed):
        axes[1].text(
            idx,
            0.92,
            "PASS" if ok else "FAIL",
            transform=axes[1].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            fontweight="semibold",
            color="#2F7D4A" if ok else "#8A4B4B",
        )
    fig.suptitle("Zero-shot KAGC transfer on ten unseen source groups", fontsize=12, fontweight="semibold")
    _finish(fig, "figure_2_kagc_transfer")


def main() -> None:
    build_kbtp(_load("r03_future_onset_confirmation_2026-09-03.json"))
    build_kagc(_load("r03_kagc_external_validation_2026-09-03.json"))


if __name__ == "__main__":
    main()
