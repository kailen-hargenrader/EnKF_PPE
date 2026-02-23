"""Plot trajectory comparisons from a saved obs_dt_sweep run.

For a given pair of obs_dt settings (default: best and worst RMSE), plots
the ensemble mean ± 1σ band against the truth for each state component.

Usage
-----
  # best vs worst (default):
  python plot_trajectories.py

  # specific obs_dt values:
  python plot_trajectories.py --obs_dts 0.01 0.50

  # specific run:
  python plot_trajectories.py --run runs/2026-02-22/18-30-00

  # save figure to run folder:
  python plot_trajectories.py --save
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt


# ── helpers ───────────────────────────────────────────────────────────────────

def _latest_run(runs_dir: Path) -> Path:
    """Return the most recently created run subfolder (handles nested date/time dirs)."""
    candidates = sorted(runs_dir.rglob("run_config.pt"))
    if not candidates:
        raise FileNotFoundError(f"No run folders found under {runs_dir}")
    return candidates[-1].parent


def load_obs_pred(run_dir: Path, obs_dt: float) -> dict:
    """Load a single obs_pred_results file by obs_dt value."""
    fname = run_dir / "obs_pred_results" / f"obs_dt_{obs_dt:.4f}.pt"
    if not fname.exists():
        available = sorted((run_dir / "obs_pred_results").glob("obs_dt_*.pt"))
        raise FileNotFoundError(
            f"No result file for obs_dt={obs_dt:.4f}.\n"
            f"Available: {[f.name for f in available]}"
        )
    return torch.load(fname, weights_only=True)


def load_truth(truth_file: str) -> torch.Tensor:
    payload = torch.load(truth_file, weights_only=True)
    return payload["data"]   # (T_full, n_state)


def _rmse_sorted_obs_dts(run_dir: Path) -> list[float]:
    """Return obs_dt values sorted by total RMSE (ascending = best first)."""
    obs_dir = run_dir / "obs_pred_results"
    records = []
    for pt_file in sorted(obs_dir.glob("obs_dt_*.pt")):
        meta = torch.load(pt_file, weights_only=True)["metadata"]
        records.append((meta["obs_dt"], meta["rmse_total"]))
    records.sort(key=lambda x: x[1])
    return [r[0] for r in records]


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_trajectories(
    run_dir:  Path,
    obs_dts:  list[float],
    T_show:   int  = 150,
    save:     bool = False,
) -> None:
    """
    Plot ensemble trajectories for each requested obs_dt.

    Args:
        run_dir:  path to the run folder
        obs_dts:  list of obs_dt values to plot (one column each)
        T_show:   max observation steps to show per panel
        save:     write PNG to run folder instead of displaying
    """
    config = torch.load(run_dir / "run_config.pt", weights_only=True)
    model_target = config.get("model", "unknown")
    model_name   = model_target.split(".")[-1] if "." in model_target else model_target

    n_cols = len(obs_dts)
    colors = plt.cm.tab10.colors

    # infer number of state components from the first result file
    first_payload = load_obs_pred(run_dir, obs_dts[0])
    n_comp = first_payload["data"].shape[-1]   # (T_obs, N_ens, n_state)

    fig, axes = plt.subplots(
        n_comp, n_cols,
        figsize=(6 * n_cols, 3 * n_comp),
        sharex="col",
        squeeze=False,
    )
    fig.suptitle(
        f"{model_name} — Trajectory Comparison\n"
        "(shaded = ensemble ±1σ  |  solid = truth  |  dashed = ensemble mean)",
        fontsize=11,
    )

    for col, obs_dt in enumerate(obs_dts):
        payload    = load_obs_pred(run_dir, obs_dt)
        meta       = payload["metadata"]
        X_hist     = payload["data"]                        # (T_obs, N_ens, n_state)
        truth_all  = load_truth(meta["truth_file"])
        n_fc       = meta["n_forecasts"]

        T_use      = X_hist.shape[0] * n_fc
        truth_sub  = truth_all[:T_use:n_fc]                # (T_obs, n_state)

        T_col      = min(T_show, X_hist.shape[0])
        X_show     = X_hist[:T_col]                         # (T_col, N_ens, n_state)
        truth_show = truth_sub[:T_col]                      # (T_col, n_state)
        mean_show  = X_show.mean(dim=1)                     # (T_col, n_state)
        std_show   = X_show.std(dim=1)                      # (T_col, n_state)
        t_ax       = np.arange(T_col) * obs_dt

        for row in range(n_comp):
            ax    = axes[row, col]
            color = colors[row % len(colors)]

            ax.fill_between(
                t_ax,
                (mean_show[:, row] - std_show[:, row]).numpy(),
                (mean_show[:, row] + std_show[:, row]).numpy(),
                color=color, alpha=0.25, label="Ens ±1σ",
            )
            ax.plot(t_ax, truth_show[:, row].numpy(),
                    "k-", lw=1.5, label="Truth")
            ax.plot(t_ax, mean_show[:, row].numpy(),
                    color=color, lw=1.5, ls="--", label="Ens mean")

            ax.set_ylabel(f"comp {row}", fontsize=11)
            ax.grid(alpha=0.3)

            if row == 0:
                ax.set_title(
                    f"obs_dt={obs_dt:.3f}  (n_fc={n_fc})\n"
                    f"RMSE={meta['rmse_total']:.4f}",
                    fontsize=10,
                )
            if row == n_comp - 1:
                ax.set_xlabel("Time", fontsize=10)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()

    if save:
        label = "_".join(f"{d:.4f}" for d in obs_dts)
        out   = run_dir / f"trajectories_{label}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    else:
        plt.show()


# ── entry point ───────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Plot trajectory comparisons from a saved run.",
    )
    p.add_argument(
        "--run", type=str, default=None,
        help="Path to a run folder (default: most recent run).",
    )
    p.add_argument(
        "--obs_dts", type=float, nargs="+", default=None,
        help=(
            "obs_dt values to plot (default: best and worst RMSE). "
            "E.g. --obs_dts 0.01 0.10 0.50"
        ),
    )
    p.add_argument(
        "--t_show", type=int, default=150,
        help="Max observation steps to display per column (default: 150).",
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

    if args.obs_dts:
        obs_dts = args.obs_dts
    else:
        sorted_dts = _rmse_sorted_obs_dts(run_dir)
        obs_dts    = [sorted_dts[0], sorted_dts[-1]]   # best and worst
        print(f"Plotting best (obs_dt={obs_dts[0]:.4f}) "
              f"and worst (obs_dt={obs_dts[1]:.4f}) by RMSE.")

    plot_trajectories(run_dir, obs_dts, T_show=args.t_show, save=args.save)
