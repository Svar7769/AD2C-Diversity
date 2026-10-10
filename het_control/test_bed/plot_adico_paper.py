#!/usr/bin/env python3
"""Create IEEE-ready ADiCo navigation figures from the downloaded CSV logs.

The curves are unsmoothed means and the bands are +/- one standard error.
Run from any directory; paths are resolved relative to this file.
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FRAME = "counters/total_frames"
SND = "eval/agents/snd"
RETURN = "eval/agents/reward/episode_reward_mean"

METHODS = {
    "ad2c": ("ADiCo", "#0072B2", "-"),
    "dico": ("Grid-searched DiCo", "#D55E00", "-."),
    "unconstrained": ("Unconstrained DiCo", "#009E73", "--"),
}


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.65,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "axes.grid.axis": "both",
        "grid.color": "#d8d8d8",
        "grid.linewidth": 0.45,
        "grid.alpha": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def finish_axis(ax: plt.Axes, panel: str, ylabel: str, xlabel: bool = True) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel("Training Frames (M)")
    ax.text(0.01, 0.98, f"({panel})", transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, fontweight="bold")
    ax.set_xlim(0, 12)
    ax.set_xticks([0, 4, 8, 12])
    ax.tick_params(direction="out", length=2.5, width=0.7)


def read_runs(pattern: Path) -> list[pd.DataFrame]:
    files = sorted(glob.glob(str(pattern)))
    return [pd.read_csv(path) for path in files]


def aggregate(runs: list[pd.DataFrame], metric: str, points: int = 200):
    clean = []
    for df in runs:
        if FRAME not in df or metric not in df:
            continue
        part = df[[FRAME, metric]].dropna().sort_values(FRAME)
        part = part.drop_duplicates(FRAME)
        if len(part) >= 2:
            clean.append(part)
    if not clean:
        raise ValueError(f"No usable runs for metric {metric}")
    lo = max(x[FRAME].min() for x in clean)
    hi = min(x[FRAME].max() for x in clean)
    frames = np.linspace(lo, hi, points)
    values = np.vstack([np.interp(frames, x[FRAME], x[metric]) for x in clean])
    mean = values.mean(axis=0)
    sem = values.std(axis=0, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.zeros(points)
    return frames / 1e6, mean, sem, len(values)


def curve(ax: plt.Axes, runs: list[pd.DataFrame], metric: str, label: str,
          color: str, linestyle: str):
    x, mean, sem, count = aggregate(runs, metric)
    line, = ax.plot(x, mean, label=label, color=color, linestyle=linestyle)
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.17, linewidth=0)
    return line, count


def fixed_target_landscape(ax: plt.Axes, data_dir: Path) -> None:
    groups: dict[float, list[pd.DataFrame]] = {}
    for path in sorted(data_dir.glob("dico_snd*_seed*.csv")):
        match = re.search(r"dico_snd([0-9]+)p([0-9]+)_seed", path.name)
        if match:
            target = float(f"{match.group(1)}.{match.group(2)}")
            groups.setdefault(target, []).append(pd.read_csv(path))
    if not groups:
        raise FileNotFoundError(f"No fixed-target DiCo runs in {data_dir}")

    targets, means, sems = [], [], []
    for target, runs in sorted(groups.items()):
        finals = []
        for df in runs:
            vals = df[[FRAME, RETURN]].dropna().sort_values(FRAME)[RETURN].to_numpy()
            if len(vals):
                finals.append(vals[-20:].mean())  # converged final 10% of 200 evaluations
        targets.append(target)
        means.append(np.mean(finals))
        sems.append(np.std(finals, ddof=1) / np.sqrt(len(finals)) if len(finals) > 1 else 0)
        print(f"Fixed target theta={target:g}: n={len(finals)}")
    ax.errorbar(targets, means, yerr=sems, color="#3B3B3B", marker="o", markersize=4.5,
                linewidth=1.2, capsize=2.2, label="Fixed-target DiCo")
    best = int(np.argmax(means))
    ax.axvspan(max(0, targets[best] - 0.08), targets[best] + 0.08,
               color="#F0E442", alpha=0.18, linewidth=0)
    ax.set_xlabel(r"SND target $\theta$")
    ax.set_ylabel(r"Final return $J(\theta)$")
    ax.set_xticks(targets)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.01, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, fontweight="bold")


def make_main_figure(out_dir: Path) -> None:
    nav = HERE / "navigation_data"
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45))
    fixed_target_landscape(axes[0], HERE / "convergence_data")
    for method, (label, color, style) in METHODS.items():
        runs = read_runs(nav / f"{method}_3A-3G_seed*.csv")
        _, n_snd = curve(axes[1], runs, SND, label, color, style)
        curve(axes[2], runs, RETURN, label, color, style)
        print(f"Main figure {label}: n={n_snd}")
    finish_axis(axes[1], "b", "System Neural Diversity (SND)")
    finish_axis(axes[2], "c", "Mean Episodic Return")
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.64, 1.01), handlelength=2.5, columnspacing=1.2)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.21, top=0.83, wspace=0.34)
    save(fig, out_dir / "figure1_main_results")


def initialization_runs(data_dir: Path, value: str) -> list[pd.DataFrame]:
    return read_runs(data_dir / f"adco_snd{value}_seed*.csv")


def stop_frame(runs: list[pd.DataFrame]) -> float:
    found = []
    for df in runs:
        if "esc/state" in df:
            stopped = df[pd.to_numeric(df["esc/state"], errors="coerce") == 1]
            if len(stopped):
                found.append(stopped[FRAME].iloc[0] / 1e6)
    return float(np.mean(found))


def make_support_figure(out_dir: Path) -> None:
    conv = HERE / "convergence_data"
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.25), sharex=True)
    colors = ["#0072B2", "#CC79A7", "#E69F00"]
    for value, shown, color in zip(["0", "0.5", "1"], ["0", "0.5", "1.0"], colors):
        runs = initialization_runs(conv, value)
        _, n = curve(axes[0], runs, SND, rf"$\theta_0={shown}$", color, "-")
        print(f"Initialization theta_0={shown}: n={n}")
    finish_axis(axes[0], "a", "System Neural Diversity (SND)", xlabel=False)
    axes[0].legend(frameon=False, ncol=3, loc="upper center", handlelength=1.5,
                   columnspacing=0.8, borderaxespad=0.35)

    persistent = read_runs(HERE / "navigation_data" / "ad2c_3A-3G_seed*.csv")
    stopped = read_runs(HERE / "stability_data" / "stable_3A-3G_seed*.csv")
    curve(axes[1], persistent, SND, "Persistent ESC", "#0072B2", "-")
    curve(axes[1], stopped, SND, "Stopped perturbation", "#D55E00", "--")
    stopped_at = stop_frame(stopped)
    axes[1].axvline(stopped_at, color="#555555", linestyle=":", linewidth=1.0)
    axes[1].annotate("Dither stopped", (stopped_at, 0.04), xycoords=("data", "axes fraction"),
                     xytext=(3, 0), textcoords="offset points", rotation=90,
                     ha="left", va="bottom", fontsize=7)
    finish_axis(axes[1], "b", "SND")
    axes[1].legend(frameon=False, loc="upper right", handlelength=2.2)
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.12, top=0.98, hspace=0.16)
    save(fig, out_dir / "figure2_convergence_perturbation")


def save(fig: plt.Figure, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {prefix}.pdf and {prefix}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "paper_figures")
    args = parser.parse_args()
    configure_style()
    make_main_figure(args.output_dir)
    make_support_figure(args.output_dir)


if __name__ == "__main__":
    main()
