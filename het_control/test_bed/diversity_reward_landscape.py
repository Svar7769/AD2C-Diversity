"""Download and plot the fixed-target DiCo diversity--reward landscape."""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

ENTITY_PROJECT = "svarp-university-of-massachusetts-lowell/benchmarl"
TARGETS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
GOALS = (1, 2, 3)
SEEDS = (0, 1, 2)
FRAME = "counters/total_frames"
RETURN = "eval/agents/reward/episode_reward_mean"
SND = "eval/agents/snd"
FIRST_RUN = "ippo_navigation_hetcontrolmlpempirical__5915d6f5_26_08_26-14_36_26"
LAST_RUN = "ippo_navigation_hetcontrolmlpempirical__fc88ef79_26_08_31-11_34_02"


def _configuration(run):
    """Extract (goals, theta, seed) from W&B config, never the run name."""
    config = run.config
    model = config.get("model_config", {}) or {}
    task = config.get("task_config", {}) or {}
    try:
        return (int(task["agents_with_same_goal"]),
                float(model["desired_snd"]), int(config["seed"]))
    except (KeyError, TypeError, ValueError):
        return None


def discover():
    """Find the 54-run series using endpoint times and configuration values."""
    api = wandb.Api()
    runs = list(api.runs(ENTITY_PROJECT))
    by_name = {run.name: run for run in runs}
    if FIRST_RUN not in by_name or LAST_RUN not in by_name:
        raise KeyError("Could not find the stated first/last run in the benchmarl project.")
    start, end = by_name[FIRST_RUN].created_at, by_name[LAST_RUN].created_at
    if start > end:
        start, end = end, start
    matched = {}
    for run in runs:
        config = _configuration(run)
        if not config or not (start <= run.created_at <= end):
            continue
        goals, theta, seed = config
        valid_theta = any(np.isclose(theta, target) for target in TARGETS)
        if goals in GOALS and seed in SEEDS and valid_theta:
            theta = min(TARGETS, key=lambda target: abs(target - theta))
            matched.setdefault((goals, theta), []).append((seed, run.name))
    print(f"Project: {ENTITY_PROJECT}")
    print(f"Series window: {start} through {end}")
    print("Expected: 3 goals x 6 targets x 3 seeds = 54 runs")
    for goals in GOALS:
        for theta in TARGETS:
            seeded = sorted(matched.get((goals, theta), []))
            matched[(goals, theta)] = [name for seed, name in seeded]
            print(f"  agents_with_same_goal={goals}, theta={theta:>4g}: seeds={[seed for seed, name in seeded]}")
    total = sum(len(names) for names in matched.values())
    problems = {key: names for key, names in matched.items() if len(names) != 3}
    print(f"Matched {total}/54 runs.")
    if problems:
        print("WARNING: combinations without exactly 3 seeds:", problems)
    return api, by_name, matched

def download(by_name, matched, data_dir="landscape_data"):
    """Download complete unsampled histories, reusing existing CSV files."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for goals in GOALS:
        for theta in TARGETS:
            for index, name in enumerate(matched.get((goals, theta), [])):
                if name not in by_name:
                    print("MISSING exact W&B name:", name)
                    continue
                tag = str(theta).replace(".", "p")
                path = data_dir / f"dico_3a{goals}g_theta{tag}_run{index}.csv"
                if not path.exists():
                    run = by_name[name]
                    keys = [FRAME, RETURN, SND, "_step"]
                    df = pd.DataFrame(list(run.scan_history(keys=keys, page_size=1000)))
                    if df.empty:
                        print(f"WARNING: no history rows for {name}; skipping write of {path.name}")
                        continue
                    df["wandb_run_name"], df["wandb_run_id"] = name, run.id
                    df["goals"], df["theta"] = goals, theta
                    df.to_csv(path, index=False)
                    print(f"Downloaded {name}: {len(df)} rows -> {path}")
                else:
                    print("Reusing", path)
                paths.append(path)
    return paths

def aggregate(data_dir="landscape_data", final_fraction=0.10):
    """Compute each run's final-window return, then mean +/- SEM across seeds."""
    records = []
    for path in sorted(Path(data_dir).glob("dico_3a*g_theta*_run*.csv")):
        df = pd.read_csv(path)
        cols = set(df.columns)
        if FRAME not in cols or RETURN not in cols:
            print(f"WARNING: missing expected columns in {path.name}: {sorted(list(cols))}")
            continue
        values = df[[FRAME, RETURN]].dropna().sort_values(FRAME)[RETURN]
        if values.empty:
            print("WARNING: no return values in", path.name)
            continue
        count = max(1, int(np.ceil(len(values) * final_fraction)))
        records.append({"goals": int(df.goals.dropna().iloc[0]),
                        "theta": float(df.theta.dropna().iloc[0]),
                        "run": df.wandb_run_name.dropna().iloc[0],
                        "final_return": values.iloc[-count:].mean()})
    per_run = pd.DataFrame(records)
    if per_run.empty:
        raise ValueError("No downloaded landscape data found.")
    result = per_run.groupby(["goals", "theta"], as_index=False).agg(
        mean_return=("final_return", "mean"), std_return=("final_return", "std"),
        seeds=("final_return", "size"))
    result["sem_return"] = result.std_return.fillna(0) / np.sqrt(result.seeds)
    return per_run, result

def plot(landscape, output_dir="paper_figures"):
    """Plot measured points, guide lines, SEM bars, and best-point outlines."""
    mpl.rcParams.update({"font.family":"serif", "font.serif":["Times New Roman","Times","Nimbus Roman","DejaVu Serif"],
        "font.size":8, "axes.labelsize":9, "xtick.labelsize":8, "ytick.labelsize":8,
        "legend.fontsize":7.5, "axes.linewidth":.7, "figure.facecolor":"white",
        "axes.facecolor":"white", "pdf.fonttype":42, "ps.fonttype":42})
    styles = {3:("1 goal","#009E73","o"), 2:("2 goals","#E69F00","s"), 1:("3 goals","#0072B2","^")}
    fig, ax = plt.subplots(figsize=(4.6, 3.35))
    for goals in reversed(GOALS):
        part = landscape[landscape.goals == goals].sort_values("theta")
        if part.empty: continue
        label, color, marker = styles[goals]
        ax.errorbar(part.theta, part.mean_return, yerr=part.sem_return, label=label,
                    color=color, marker=marker, markersize=4.5, linewidth=1.6, capsize=2.2)
        best = part.loc[part.mean_return.idxmax()]
        ax.scatter(best.theta, best.mean_return, s=80, facecolors="none",
                   edgecolors=color, linewidths=1.1, zorder=4)
    ax.set(xlabel=r"SND target $\theta$", ylabel=r"Final evaluation return $J(\theta)$")
    ax.set_xticks(TARGETS)
    ax.grid(True, color="#d8d8d8", linewidth=.45, alpha=.7)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(direction="out", length=2.5, width=.7)
    ax.legend(frameon=False, loc="best", handlelength=2.2)
    fig.tight_layout()
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    prefix = out / "diversity_reward_landscape"
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved {prefix}.pdf and {prefix}.png")
    return fig, ax
