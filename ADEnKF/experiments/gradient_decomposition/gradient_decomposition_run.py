"""
gradient_decomposition_run.py
==============================
Demonstrates Term A vs Term A+B gradient decomposition from §4.1.1,
using the existing torchEnKF infrastructure on the Lorenz-63 system.

The key distinction between EM-EnKF and AD-EnKF:
  - AD-EnKF: full backprop through the EnKF computation graph
              → gradient captures Term A + Term B (particle history)
  - EM-EnKF: particles detached before gradient step
              → gradient captures Term A only (direct dependence on theta)

Term B = AD grad - EM grad

Run from repo root with:
    PYTHONPATH=. python ADEnKF/experiments/gradient_decomposition/gradient_decomposition_run.py

Run from ADEnKF/experiments/gradient_decomposition with:
    python gradient_decomposition_run.py

If you haven't already, first generate the L63 observations with:
    PYTHONPATH=. python ADEnKF/experiments/lorenz63_data_gen.py

Outputs saved to ADEnKF/experiments/gradient_decomposition/figures/
"""

import sys
import os
from pathlib import Path

# -- path setup (mirrors l63_param_est_run.py) ----------------------------
_script_dir   = Path(__file__).resolve().parent
_ad_enkf_dir  = _script_dir.parent.parent          # -> ADEnKF/
_repo_root    = _ad_enkf_dir.parent                # -> repo root
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_ad_enkf_dir))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from torchEnKF import da_methods, nn_templates, noise
from paths import DATA_DIR

torch.manual_seed(42)
np.random.seed(42)

# ── figure output directory ───────────────────────────────────────────────
FIG_DIR = _script_dir / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


# ── shared system parameters ─────────────────────────────────────────────
TRUE_SIGMA = 10.0
TRUE_RHO   = 28.0
TRUE_BETA  = 8 / 3
x_dim      = 3
N_ENS      = 50
OBS_STD    = 1.0
PROC_STD   = 0.5

true_coeff    = torch.tensor([TRUE_SIGMA, TRUE_RHO, TRUE_BETA], device=device)
true_ode_func = nn_templates.Lorenz63(true_coeff).to(device)

H_true       = torch.eye(x_dim, device=device)
true_obs_func = nn_templates.Linear(x_dim, x_dim, H=H_true).to(device)
noise_R_true  = noise.AddGaussian(x_dim, torch.tensor(OBS_STD, device=device),
                                  param_type="scalar").to(device)

init_m      = torch.zeros(x_dim, device=device)
init_cov    = torch.diag(torch.tensor([25.0, 25.0, 50.0], device=device))
init_C_param = noise.AddGaussian(x_dim, init_cov, "full").to(device)

MODEL_Q_SCALE = PROC_STD
model_Q = noise.AddGaussian(x_dim,
                             MODEL_Q_SCALE * torch.ones(x_dim, device=device),
                             "diag").to(device)


def load_observations(n_obs: int = 200, n_forecasts: int = 5, dt: float = 0.01):
    """Load L63 observations from the pre-generated data file."""
    data_file = DATA_DIR / "Lorentz63/sigma10.0000_rho28.0000_beta2.6667_dt0.0100.pt"
    payload   = torch.load(data_file, weights_only=True)
    truth     = payload["data"]          # (T_full, 3)
    truth_sub = truth[:n_obs * n_forecasts:n_forecasts][:n_obs].to(device)  # (n_obs, 3)
    obs_dt    = n_forecasts * dt
    t_obs     = obs_dt * torch.arange(1, n_obs + 1, device=device)
    y_obs     = truth_sub + OBS_STD * torch.randn_like(truth_sub)
    # da_methods.EnKF expects (n_obs, *bs, y_dim) — add batch dim of size 1
    return y_obs.unsqueeze(1), t_obs, obs_dt

def compute_gradient(coeff: torch.Tensor, y_obs, t_obs, obs_dt, mode: str):
    """
    Compute d(log_likelihood)/d(coeff) using either AD or EM mode.

    mode="ad" → adjoint backprop, full computation graph → Term A + B
    mode="em" → torch.no_grad() on particles (detach after each step) → Term A only

    The EM approximation is implemented by running EnKF inside torch.no_grad()
    to get the filtered states, then re-running one forward pass through only
    the *current* step's likelihood term with grad enabled.  This mirrors the
    EM M-step: particles are treated as fixed ("frozen") constants.
    """
    assert mode in ("ad", "em")

    coeff_p = coeff.detach().clone().requires_grad_(True)
    learned = nn_templates.Lorenz63(coeff_p).to(device)

    bs = y_obs.shape[1:-1]

    if mode == "ad":
        _, _, log_lik = da_methods.EnKF(
            learned, true_obs_func, t_obs, y_obs, N_ENS,
            init_m, init_C_param, model_Q, noise_R_true, device,
            save_filter_step={},
            ode_method="rk4",
            ode_options=dict(step_size=0.01),
            adjoint=True,
            adjoint_method="rk4",
            adjoint_options=dict(step_size=0.05),
            tqdm=None,
        )

    else:
        import math
        from torchdiffeq import odeint
        from torchEnKF.da_methods import inv_logdet

        x_dim_     = init_m.shape[0]
        y_dim_     = y_obs.shape[-1]
        n_obs_     = y_obs.shape[0]
        step_size  = 0.01

        noise_R        = noise_R_true.full()
        noise_R_inv    = noise_R_true.inv()
        logdet_noise_R = noise_R_true.logdet()

        X       = init_C_param(init_m.expand(*bs, N_ENS, x_dim_))
        log_lik = torch.zeros(bs, device=device)
        t_cur   = 0.0

        for j in range(n_obs_):
            # forecast
            n_steps = round(((t_obs[j] - t_cur) / step_size).item())
            t_span  = torch.linspace(t_cur, t_obs[j].item(),
                                     n_steps + 1, device=device)
            X     = odeint(learned, X, t_span, method='rk4',
                           options=dict(step_size=step_size))[-1]
            t_cur = t_obs[j].item()

            if model_Q is not None:
                X = model_Q(X)

            X_m  = X.mean(dim=-2).unsqueeze(-2)
            X_ct = X - X_m

            # analysis
            H    = true_obs_func.H
            HX   = X @ H.T
            HX_m = X_m @ H.T

            y_obs_j     = y_obs[j].unsqueeze(-2)
            obs_perturb = noise_R_true(y_obs_j.expand(*bs, N_ENS, y_dim_))

            HX_ct   = HX - HX_m
            C_ww_sq = 1 / math.sqrt(N_ENS - 1) * HX_ct

            v = torch.cat([obs_perturb - HX, y_obs_j - HX_m], dim=-2)
            C_ww_R_invv, C_ww_R_logdet = inv_logdet(
                v, C_ww_sq, noise_R, noise_R_inv, logdet_noise_R)

            part1 = -0.5 * (y_dim_ * math.log(2 * math.pi) + C_ww_R_logdet)
            part2 = -0.5 * (C_ww_R_invv[..., N_ENS:, :]
                            @ (y_obs_j - HX_m).transpose(-1, -2))
            log_lik += (part1 + part2.squeeze(-1).squeeze(-1))

            pre = C_ww_R_invv[..., :N_ENS, :]
            X   = X + (1 / math.sqrt(N_ENS - 1)
                       * (pre @ C_ww_sq.transpose(-1, -2)) @ X_ct)

            # severs Term B, leaving only Term A
            X = X.detach()

        log_lik = log_lik.mean()

    log_lik.mean().backward()
    return learned.coeff.grad.clone()


# ═══════════════════════════════════════════════════════════════════════════
# Panel 1 — Gradient landscape: AD vs EM as a function of sigma
# ═══════════════════════════════════════════════════════════════════════════

def panel1_gradient_landscape(ax, n_obs: int = 100):
    print("Panel 1: gradient landscape...")
    y_obs, t_obs, _ = load_observations(n_obs=n_obs)

    sigma_vals = np.linspace(5.0, 16.0, 25)
    ad_grads, em_grads = [], []

    for sv in sigma_vals:
        coeff = torch.tensor([sv, TRUE_RHO, TRUE_BETA], dtype=torch.float32, device=device)
        g_ad = compute_gradient(coeff, y_obs, t_obs, None, mode="ad")
        g_em = compute_gradient(coeff, y_obs, t_obs, None, mode="em")
        ad_grads.append(g_ad[0].item())   # sigma component
        em_grads.append(g_em[0].item())

    ad_grads = np.array(ad_grads)
    em_grads = np.array(em_grads)

    ax.plot(sigma_vals, ad_grads, color="#2563eb", lw=2.0, label="AD-EnKF  (Term A + B)")
    ax.plot(sigma_vals, em_grads, color="#dc2626", lw=2.0, linestyle="--", label="EM-EnKF  (Term A only)")
    ax.fill_between(sigma_vals, em_grads, ad_grads,
                    alpha=0.15, color="#7c3aed", label="Term B  (missing from EM)")
    ax.axvline(TRUE_SIGMA, color="black", lw=1.2, linestyle=":",
               label=rf"True $\sigma^*={TRUE_SIGMA}$")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel(r"$\sigma$", fontsize=12)
    ax.set_ylabel(r"$\nabla_\sigma \, \ell^{\mathrm{EnKF}}$", fontsize=11)
    ax.set_title(f"Panel 1: Gradient Landscape  (T={n_obs})", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


# ═══════════════════════════════════════════════════════════════════════════
# Panel 2 — Optimisation trajectories from a shared off-true initialisation
# ═══════════════════════════════════════════════════════════════════════════

def panel2_optimisation(ax, n_obs: int = 150, n_steps: int = 40, lr: float = 0.3):
    print("Panel 2: optimisation trajectories...")
    y_obs, t_obs, _ = load_observations(n_obs=n_obs)

    sigma_ad = 5.0
    sigma_em = 5.0
    traj_ad  = [sigma_ad]
    traj_em  = [sigma_em]

    for _ in range(n_steps):
        coeff = torch.tensor([sigma_ad, TRUE_RHO, TRUE_BETA], dtype=torch.float32, device=device)
        g     = compute_gradient(coeff, y_obs, t_obs, None, mode="ad")
        sigma_ad = float(np.clip(sigma_ad + lr * g[0].item(), 1.0, 20.0))
        traj_ad.append(sigma_ad)

        coeff = torch.tensor([sigma_em, TRUE_RHO, TRUE_BETA], dtype=torch.float32, device=device)
        g     = compute_gradient(coeff, y_obs, t_obs, None, mode="em")
        sigma_em = float(np.clip(sigma_em + lr * g[0].item(), 1.0, 20.0))
        traj_em.append(sigma_em)

    steps = np.arange(n_steps + 1)
    ax.plot(steps, traj_ad, color="#2563eb", lw=2.0, label="AD-EnKF")
    ax.plot(steps, traj_em, color="#dc2626", lw=2.0, linestyle="--", label="EM-EnKF")
    ax.axhline(TRUE_SIGMA, color="black", lw=1.2, linestyle=":",
               label=rf"True $\sigma^*={TRUE_SIGMA}$")
    ax.set_xlabel("Gradient ascent iteration", fontsize=11)
    ax.set_ylabel(r"$\sigma$ estimate", fontsize=11)
    ax.set_title(f"Panel 2: Optimisation Trajectories  (T={n_obs})", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


# ═══════════════════════════════════════════════════════════════════════════
# Panel 3 — |Term B| / |Term A| vs sequence length T
# ═══════════════════════════════════════════════════════════════════════════

def panel3_termB_vs_T(ax, T_values=None):
    print("Panel 3: Term B magnitude vs T...")
    if T_values is None:
        T_values = [10, 20, 40, 80, 150]

    ratios = []
    coeff  = torch.tensor([TRUE_SIGMA, TRUE_RHO, TRUE_BETA],
                           dtype=torch.float32, device=device)

    for T in T_values:
        y_obs, t_obs, _ = load_observations(n_obs=T)
        g_ad = compute_gradient(coeff, y_obs, t_obs, None, mode="ad")
        g_em = compute_gradient(coeff, y_obs, t_obs, None, mode="em")

        term_B = (g_ad - g_em).norm().item()
        term_A = g_em.norm().item() + 1e-8
        ratios.append(term_B / term_A)

    ax.bar(range(len(T_values)), ratios, color="#7c3aed", alpha=0.75, edgecolor="white")
    ax.set_xticks(range(len(T_values)))
    ax.set_xticklabels([str(t) for t in T_values])
    ax.axhline(1.0, color="grey", lw=0.9, linestyle="--", label="Term B = Term A")
    ax.set_xlabel("Sequence length  T", fontsize=11)
    ax.set_ylabel(r"$\|\mathrm{Term\,B}\| \,/\, \|\mathrm{Term\,A}\|$", fontsize=11)
    ax.set_title("Panel 3: Relative Magnitude of Term B vs T", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    panel1_gradient_landscape(ax1)
    panel2_optimisation(ax2)
    panel3_termB_vs_T(ax3)

    fig.suptitle(
        r"Gradient Decomposition: EM-EnKF (Term A) vs AD-EnKF (Term A + B)  —  §4.1.1  (L63, $\sigma$-component)",
        fontsize=12, y=1.01
    )

    out_path = FIG_DIR / "gradient_decomposition.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {out_path}")


if __name__ == "__main__":
    main()
