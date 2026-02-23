"""Plot RMSE vs observation dt from a saved obs_dt_sweep run.

Usage
-----
  # most recent run (default):
  python plot_rmse.py

  # specific run:
  python plot_rmse.py --run runs/2026-02-22/18-30-00

  # save figure without displaying:
  python plot_rmse.py --save
"""

import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt


# ── helpers ───────────────────────────────────────────────────────────────────

def _latest_run(runs_dir: Path) -> Path:
    """Return the most recently created run subfolder (handles nested date/time dirs)."""
    candidates = sorted(runs_dir.rglob("run_config.pt"))
    if not candidates:
        raise FileNotFoundError(f"No run folders found under {runs_dir}")
    return candidates[-1].parent


def load_run(run_dir: Path):
    """
    Load run config and all obs_pred_results files.

    Returns
    -------
    config : dict
    records : list[dict]  — each entry is the metadata dict for one obs_dt,
                            sorted by obs_dt ascending
    """
    config  = torch.load(run_dir / "run_config.pt", weights_only=True)
    obs_dir = run_dir / "obs_pred_results"

    records = []
    for pt_file in sorted(obs_dir.glob("obs_dt_*.pt")):
        payload = torch.load(pt_file, weights_only=True)
        records.append(payload["metadata"])

    records.sort(key=lambda r: r["obs_dt"])
    return config, records


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_rmse(run_dir: Path, save: bool = False) -> None:
    config, records = load_run(run_dir)

    obs_dts    = [r["obs_dt"]          for r in records]
    rmse_tots  = [r["rmse_total"]      for r in records]
    rmse_comps = [r["rmse_components"] for r in records]   # list of lists
    n_fcs      = [r["n_forecasts"]     for r in records]
    n_comp     = len(rmse_comps[0])

    model_target = config.get("model", "unknown")
    model_name   = model_target.split(".")[-1] if "." in model_target else model_target
    data_name    = config.get("data", "data")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"{model_name} — Effect of Observation Frequency  [{data_name}]\n"
        f"N={config['N_ens']},  σ_obs={config['obs_std']},  σ_proc={config['proc_std']}",
        fontsize=11,
    )

    # left: total RMSE
    ax = axes[0]
    ax.plot(obs_dts, rmse_tots, "ko-", lw=2, ms=7)
    for n_fc, x, y in zip(n_fcs, obs_dts, rmse_tots):
        ax.annotate(f"n={n_fc}", (x, y),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("Observation $\\Delta t$", fontsize=11)
    ax.set_ylabel("RMSE", fontsize=11)
    ax.set_title("Total RMSE vs Observation $\\Delta t$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)

    # right: per-component RMSE
    ax = axes[1]
    colors = plt.cm.tab10.colors
    for i in range(n_comp):
        comp_rmse = [c[i] for c in rmse_comps]
        ax.plot(obs_dts, comp_rmse,
                marker="o", ls="--", lw=1.5, ms=6,
                color=colors[i % len(colors)],
                label=f"comp {i}")
    ax.set_xlabel("Observation $\\Delta t$", fontsize=11)
    ax.set_ylabel("RMSE", fontsize=11)
    ax.set_title("Per-Component RMSE vs Observation $\\Delta t$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save:
        out = run_dir / "rmse_vs_obs_dt.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    else:
        plt.show()


# ── entry point ───────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Plot RMSE sweep from a saved run.")
    p.add_argument(
        "--run", type=str, default=None,
        help="Path to a run folder (default: most recent run).",
    )
    p.add_argument(
        "--save", action="store_true",
        help="Save figure to the run folder instead of displaying.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args     = _parse_args()
    runs_dir = Path(__file__).parent / "runs"

    if args.run:
        run_dir = (Path(args.run) if Path(args.run).is_absolute()
                   else Path(__file__).parent / args.run)
    else:
        run_dir = _latest_run(runs_dir)

    print(f"Loading run: {run_dir}")
    plot_rmse(run_dir, save=args.save)
