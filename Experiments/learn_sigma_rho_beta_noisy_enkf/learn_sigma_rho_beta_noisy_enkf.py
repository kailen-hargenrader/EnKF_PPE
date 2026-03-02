"""
Experiment: Learn Sigma, Rho, and Beta Parameters with Noisy Observations using State Augmented EnKF

This experiment extends learn_sigma_beta_noisy_enkf by learning ALL THREE parameters:
- Sigma (shape parameter)
- Rho (Rayleigh number - controls chaotic behavior)
- Beta (geometric parameter)

This tests the State Augmented EnKF's ability to learn multiple parameters
from realistic noisy data with high-dimensional parameter space.

The EnKF approach:
- Treats parameters as part of the augmented state: z = [x; theta]
- Uses sequential Bayesian estimation with forecast and analysis steps
- Learns parameters through cross-correlations in the forecast covariance
- Handles observation noise through the obs_noise_cov matrix

Challenge: Learning 3 parameters from noisy observations is significantly harder than
learning 1-2 parameters. Requires careful tuning of ensemble size and noise parameters.

Goal: Assess:
- Full parameter identifiability with noisy observations
- How many observations are needed for convergence
- Parameter coupling and correlation effects
- Limitations of EnKF for high-dimensional parameter spaces
"""

import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple
from omegaconf import DictConfig
import hydra

from enkf_ppe.Dynamics.Lorentz63 import Lorentz63, Lorenz63derivs
from enkf_ppe.Models.ENKF.state_aug_enkf import StateAugEnKF
from enkf_ppe.Utils.observation_fns import FullObservation
from enkf_ppe.Utils.covariances import ScaledIdentity


def load_data(config: DictConfig) -> torch.Tensor:
    """
    Load the Lorentz63 dataset.
    
    Args:
        config: Experiment configuration (DictConfig)
        
    Returns:
        observations: (T, 3) tensor of observations, optionally with added noise
    """
    data_dir = Path(__file__).parent.parent.parent / "Data" / "Lorentz63"
    data_path = data_dir / config.data.file
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found at {data_path}\n"
            f"Please generate it with: python Data/Lorentz63/generate_data.py"
        )
    
    payload = torch.load(data_path, weights_only=True)
    data = payload['data']
    
    # Use first n_obs time steps
    observations = data[:int(config.data.n_obs)]
    
    # Add observation noise if specified
    obs_noise_std = config.data.noise_std
    if obs_noise_std > 0:
        observations = observations + torch.randn_like(observations) * obs_noise_std
        print(f"Added Gaussian noise (std={obs_noise_std}) to observations")
    
    print(f"Loaded data shape: {observations.shape}")
    print(f"Data file: {data_path}")
    
    return observations


def create_dynamics_wrapper(config: DictConfig) -> Tuple[nn.Module, nn.Module]:
    """
    Create the dynamics and observation functions.
    
    The dynamics function is the TRUE Lorentz63 system (not learned).
    This tests whether EnKF can recover parameters when the model is correct.
    
    Args:
        config: Experiment configuration
        
    Returns:
        trans_fn: transition function (X, theta) -> dX/dt
        obs_fn: observation function X -> Y
    """
    # Dynamics: true Lorentz63 system (provides derivatives)
    dynamics = Lorenz63derivs()
    
    # Observation: full state observation (identity function)
    obs_fn = FullObservation()
    
    return dynamics, obs_fn


def run_experiment(config: DictConfig) -> Dict:
    """
    Run the learn_sigma_rho_beta_noisy_enkf experiment.
    
    Args:
        config: Experiment configuration (DictConfig)
        
    Returns:
        results: Dictionary containing:
            - X_hist: state trajectory history (T, N, 3)
            - theta_hist: parameter history (T, N, 3)
            - true_params: true parameters
            - initial_params_mean: mean of initial parameter ensemble
            - observations: observation data
    """
    # Extract config values
    true_params = torch.tensor([[
        config.parameters.truth.sigma,
        config.parameters.truth.rho,
        config.parameters.truth.beta
    ]])
    
    initial_mean = torch.tensor([
        config.parameters.initial_guess.sigma,
        config.parameters.initial_guess.rho,
        config.parameters.initial_guess.beta
    ])
    
    x0_mean = torch.tensor([
        config.initial_condition.x,
        config.initial_condition.y,
        config.initial_condition.z
    ])
    
    ensemble_size = int(config.enkf.ensemble_size)
    dt_model = config.data.dt_model
    dt_obs = config.data.dt_obs
    process_noise_std = config.enkf.process_noise_std
    param_noise_std = config.enkf.param_noise_std
    obs_noise_std = config.enkf.obs_noise_std + 1e-6
    
    print("\n" + "="*70)
    print("EXPERIMENT: Learn Sigma, Rho, and Beta Parameters using State Augmented EnKF")
    print("="*70)
    
    # Load data
    print("\n[1/4] Loading data...")
    observations = load_data(config)
    
    # Create dynamics and observation functions
    print("[2/4] Setting up dynamics...")
    trans_fn, obs_fn = create_dynamics_wrapper(config)
    
    # Create ensemble of initial conditions
    print(f"[3/4] Setting up State Augmented EnKF (ensemble size: {ensemble_size})...")
    
    # Initial state ensemble: small perturbation around the true initial state
    X0_ensemble = x0_mean.unsqueeze(0).repeat(ensemble_size, 1)
    X0_ensemble = X0_ensemble + torch.randn_like(X0_ensemble) * 0.01
    
    # Initial parameter ensemble: spread around the initial guesses
    # Learning all three parameters: sigma (index 0), rho (index 1), beta (index 2)
    theta0_ensemble = initial_mean.unsqueeze(0).repeat(ensemble_size, 1)
    theta0_ensemble = theta0_ensemble + torch.randn_like(theta0_ensemble) * (
        torch.tensor([0.5, 1.0, 0.3])  # Spread for sigma, rho, beta
    )
    
    # Create covariance matrices for EnKF
    process_noise_cov = ScaledIdentity(dim=3, std=process_noise_std, track_grads=False)
    param_noise_cov = ScaledIdentity(dim=3, std=param_noise_std, track_grads=False)
    obs_noise_cov = ScaledIdentity(dim=3, std=obs_noise_std, track_grads=False)
    
    # Create EnKF model
    model = StateAugEnKF(
        process_noise_cov=process_noise_cov,
        param_noise_cov=param_noise_cov,
        obs_noise_cov=obs_noise_cov,
    )
    
    # Run the filter
    print("[4/4] Running State Augmented EnKF...")
    print(f"  - Initial sigma (mean):     {initial_mean[0].item():.4f}")
    print(f"  - Initial rho (mean):       {initial_mean[1].item():.4f}")
    print(f"  - Initial beta (mean):      {initial_mean[2].item():.4f}")
    print(f"  - True sigma:               {true_params[0, 0].item():.4f}")
    print(f"  - True rho:                 {true_params[0, 1].item():.4f}")
    print(f"  - True beta:                {true_params[0, 2].item():.4f}")
    print(f"  - Process noise std:        {process_noise_std}")
    print(f"  - Param noise std:          {param_noise_std}")
    print(f"  - Obs noise std:            {obs_noise_std}")
    
    X_hist, theta_hist = model.run(
        X0=X0_ensemble,
        theta0=theta0_ensemble,
        observations=observations,
        trans_dt=dt_model,
        obs_dt=dt_obs,
        trans_fn=trans_fn,
        obs_fn=obs_fn,
    )
    
    # Compute final metrics for all three parameters
    final_sigma_mean = theta_hist[-1, :, 0].mean().item()
    final_rho_mean = theta_hist[-1, :, 1].mean().item()
    final_beta_mean = theta_hist[-1, :, 2].mean().item()
    
    true_sigma = true_params[0, 0].item()
    true_rho = true_params[0, 1].item()
    true_beta = true_params[0, 2].item()
    
    sigma_error = abs(final_sigma_mean - true_sigma)
    rho_error = abs(final_rho_mean - true_rho)
    beta_error = abs(final_beta_mean - true_beta)
    
    sigma_relative_error = sigma_error / true_sigma * 100
    rho_relative_error = rho_error / true_rho * 100
    beta_relative_error = beta_error / true_beta * 100
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Final sigma (ensemble mean): {final_sigma_mean:.6f}")
    print(f"True sigma:                  {true_sigma:.6f}")
    print(f"Sigma absolute error:        {sigma_error:.6f}")
    print(f"Sigma relative error:        {sigma_relative_error:.2f}%")
    print(f"Sigma ensemble std (final):  {theta_hist[-1, :, 0].std().item():.6f}")
    print()
    print(f"Final rho (ensemble mean):   {final_rho_mean:.6f}")
    print(f"True rho:                    {true_rho:.6f}")
    print(f"Rho absolute error:          {rho_error:.6f}")
    print(f"Rho relative error:          {rho_relative_error:.2f}%")
    print(f"Rho ensemble std (final):    {theta_hist[-1, :, 1].std().item():.6f}")
    print()
    print(f"Final beta (ensemble mean):  {final_beta_mean:.6f}")
    print(f"True beta:                   {true_beta:.6f}")
    print(f"Beta absolute error:         {beta_error:.6f}")
    print(f"Beta relative error:         {beta_relative_error:.2f}%")
    print(f"Beta ensemble std (final):   {theta_hist[-1, :, 2].std().item():.6f}")
    
    results = {
        'X_hist': X_hist,
        'theta_hist': theta_hist,
        'true_params': true_params,
        'initial_params_mean': initial_mean,
        'observations': observations,
        'config': config,
    }
    
    return results


def plot_results(results: Dict, output_dir: Path) -> None:
    """
    Create visualizations of the experiment results.
    
    Args:
        results: Results dictionary from run_experiment
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    X_hist = results['X_hist']
    theta_hist = results['theta_hist']
    observations = results['observations']
    
    T, N, n = X_hist.shape
    
    # Figure 1: All three parameter convergence (ensemble statistics)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    sigma_hist = theta_hist[:, :, 0].numpy()  # (T, N)
    rho_hist = theta_hist[:, :, 1].numpy()    # (T, N)
    beta_hist = theta_hist[:, :, 2].numpy()   # (T, N)
    
    sigma_mean = sigma_hist.mean(axis=1)
    sigma_std = sigma_hist.std(axis=1)
    rho_mean = rho_hist.mean(axis=1)
    rho_std = rho_hist.std(axis=1)
    beta_mean = beta_hist.mean(axis=1)
    beta_std = beta_hist.std(axis=1)
    
    true_sigma = results['true_params'][0, 0].item()
    true_rho = results['true_params'][0, 1].item()
    true_beta = results['true_params'][0, 2].item()
    initial_sigma = results['initial_params_mean'][0].item()
    initial_rho = results['initial_params_mean'][1].item()
    initial_beta = results['initial_params_mean'][2].item()
    
    time_axis = np.arange(len(sigma_mean))
    
    # Sigma convergence
    ax = axes[0]
    ax.plot(time_axis, sigma_mean, 'b-', linewidth=2.5, label='Ensemble mean sigma')
    ax.fill_between(time_axis, 
                     sigma_mean - sigma_std, 
                     sigma_mean + sigma_std, 
                     alpha=0.3, color='blue', label='Ensemble ±1 std')
    ax.axhline(true_sigma, color='r', linestyle='--', linewidth=2, label='True sigma')
    ax.scatter([0], [initial_sigma], color='g', s=100, marker='o', 
               label='Initial guess (mean)', zorder=5, edgecolors='darkgreen', linewidth=2)
    ax.set_ylabel('Sigma value', fontsize=12)
    ax.set_title('Sigma Parameter Convergence', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Rho convergence
    ax = axes[1]
    ax.plot(time_axis, rho_mean, 'purple', linewidth=2.5, label='Ensemble mean rho')
    ax.fill_between(time_axis, 
                     rho_mean - rho_std, 
                     rho_mean + rho_std, 
                     alpha=0.3, color='purple', label='Ensemble ±1 std')
    ax.axhline(true_rho, color='r', linestyle='--', linewidth=2, label='True rho')
    ax.scatter([0], [initial_rho], color='orange', s=100, marker='o', 
               label='Initial guess (mean)', zorder=5, edgecolors='darkorange', linewidth=2)
    ax.set_ylabel('Rho value', fontsize=12)
    ax.set_title('Rho Parameter Convergence', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Beta convergence
    ax = axes[2]
    ax.plot(time_axis, beta_mean, 'g-', linewidth=2.5, label='Ensemble mean beta')
    ax.fill_between(time_axis, 
                     beta_mean - beta_std, 
                     beta_mean + beta_std, 
                     alpha=0.3, color='green', label='Ensemble ±1 std')
    ax.axhline(true_beta, color='r', linestyle='--', linewidth=2, label='True beta')
    ax.scatter([0], [initial_beta], color='cyan', s=100, marker='o', 
               label='Initial guess (mean)', zorder=5, edgecolors='darkcyan', linewidth=2)
    ax.set_xlabel('Time (observations)', fontsize=12)
    ax.set_ylabel('Beta value', fontsize=12)
    ax.set_title('Beta Parameter Convergence', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('All Parameters Convergence - State Augmented EnKF (with Noisy Observations)', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'sigma_rho_beta_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Final trajectory vs observations
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Compute ensemble mean trajectory over all time steps
    final_traj_mean = X_hist.mean(dim=1).numpy()  # (T, 3) - mean over ensemble
    obs_np = observations.numpy()
    
    for i, (ax, name) in enumerate(zip(axes, ['x', 'y', 'z'])):
        ax.plot(obs_np[:, i], 'r-', linewidth=1.5, alpha=0.7, label='Observations (noisy)')
        ax.plot(final_traj_mean[:, i], 'b--', linewidth=1.5, alpha=0.7, 
                label='Ensemble mean prediction')
        ax.set_ylabel(name.upper(), fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time step', fontsize=12)
    fig.suptitle('Final Ensemble Mean Fit vs Noisy Observations', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory_fit.png', dpi=150)
    plt.close()
    
    # Figure 3: Parameter history for all ensemble members (all three parameters)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    param_names = ['sigma', 'rho', 'beta']
    param_indices = [0, 1, 2]
    
    for ax, name, idx in zip(axes, param_names, param_indices):
        param_hist = theta_hist[:, :, idx].numpy()  # (T, N)
        true_val = results['true_params'][0, idx].item()
        initial_val = results['initial_params_mean'][idx].item()
        
        # Plot individual ensemble members (light)
        for j in range(N):
            ax.plot(param_hist[:, j], alpha=0.15, color='blue', linewidth=0.8)
        
        # Plot ensemble mean
        param_mean = param_hist.mean(axis=1)
        ax.plot(param_mean, 'b-', linewidth=2.5, label='Ensemble mean')
        
        # Plot true value
        ax.axhline(true_val, color='r', linestyle='--', linewidth=2, label='True')
        ax.axhline(initial_val, color='g', linestyle=':', linewidth=2, label='Initial guess (mean)')
        
        ax.set_ylabel(name, fontsize=11)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (observations)', fontsize=12)
    fig.suptitle('Parameter Evolution - Sigma, Rho, and Beta (All Ensemble Members)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_evolution.png', dpi=150)
    plt.close()
    
    print(f"\nPlots saved to: {output_dir}")


def plot_trajectory_comparison(results: Dict, output_dir: Path) -> None:
    """
    Plot trajectories for true and final parameters.
    
    Args:
        results: Results dictionary from run_experiment
        output_dir: Directory to save plot
    """
    from enkf_ppe.Dynamics.Lorentz63 import Lorentz63
    
    observations = results['observations']
    true_params = results['true_params']
    initial_params_mean = results['initial_params_mean']
    theta_hist = results['theta_hist']
    
    final_sigma = theta_hist[-1, :, 0].mean().item()
    final_rho = theta_hist[-1, :, 1].mean().item()
    final_beta = theta_hist[-1, :, 2].mean().item()
    
    initial_sigma = initial_params_mean[0].item()
    initial_rho = initial_params_mean[1].item()
    initial_beta = initial_params_mean[2].item()
    
    true_sigma = true_params[0, 0].item()
    true_rho = true_params[0, 1].item()
    true_beta = true_params[0, 2].item()
    
    # Get initial state from observations
    x0 = observations[0:1]
    dt_obs = results['config'].data.dt_obs
    dt_model = results['config'].data.dt_model
    
    # Initialize RK4 dynamics
    dynamics_rk4 = Lorentz63()
    
    # Generate trajectories for different parameter combinations
    trajectories = {}
    for label, sigma_val, rho_val, beta_val in [
        ('true', true_sigma, true_rho, true_beta),
        ('initial', initial_sigma, initial_rho, initial_beta),
        ('final', final_sigma, final_rho, final_beta)
    ]:
        traj = [x0.numpy()]
        X = x0.clone()
        params = torch.tensor([[sigma_val, rho_val, beta_val]])
        
        for _ in range(len(observations) - 1):
            n_steps = int(round(dt_obs / dt_model))
            for _ in range(n_steps):
                X = dynamics_rk4(X, params, dt=dt_model)
            traj.append(X.detach().numpy())
        
        trajectories[label] = torch.tensor(traj).squeeze()
    
    # Create comparison plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    obs_np = observations.numpy()
    time_steps = range(len(observations))
    
    for i, (ax, name) in enumerate(zip(axes, ['x', 'y', 'z'])):
        # Plot observations
        ax.plot(time_steps, obs_np[:, i], 'k-', linewidth=2.5, 
                label='Noisy observations', alpha=0.8, zorder=4)
        
        # Plot trajectories
        ax.plot(time_steps, trajectories['true'][:, i], 'r--', linewidth=2, 
                label=f'True (σ={true_sigma:.2f}, ρ={true_rho:.2f}, β={true_beta:.4f})', 
                alpha=0.7, zorder=3)
        ax.plot(time_steps, trajectories['initial'][:, i], 'orange', 
                linestyle=':', linewidth=2, 
                label=f'Initial (σ={initial_sigma:.2f}, ρ={initial_rho:.2f}, β={initial_beta:.4f})', 
                alpha=0.7, zorder=2)
        ax.plot(time_steps, trajectories['final'][:, i], 'b--', linewidth=2, 
                label=f'Final (σ={final_sigma:.2f}, ρ={final_rho:.2f}, β={final_beta:.4f})', 
                alpha=0.7, zorder=3)
        
        ax.set_ylabel(f'{name.upper()}(t)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    
    axes[-1].set_xlabel('Time step', fontsize=12, fontweight='bold')
    fig.suptitle('Trajectory Comparison: Learning All Three Parameters from Noisy Observations (EnKF)', 
                fontsize=14, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Trajectory comparison plot saved to {output_dir / 'trajectory_comparison.png'}")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig) -> None:
    """Main entry point for the experiment."""
    # Run experiment
    results = run_experiment(config)
    
    # Save results to script directory (not Hydra's output dir)
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract parameter estimates over time (ensemble mean)
    sigma_hist_mean = results['theta_hist'][:, :, 0].mean(dim=1)
    sigma_hist_std = results['theta_hist'][:, :, 0].std(dim=1)
    rho_hist_mean = results['theta_hist'][:, :, 1].mean(dim=1)
    rho_hist_std = results['theta_hist'][:, :, 1].std(dim=1)
    beta_hist_mean = results['theta_hist'][:, :, 2].mean(dim=1)
    beta_hist_std = results['theta_hist'][:, :, 2].std(dim=1)
    
    # Save PyTorch results
    torch.save({
        'X_hist': results['X_hist'],
        'theta_hist': results['theta_hist'],
        'sigma_hist_mean': sigma_hist_mean,
        'sigma_hist_std': sigma_hist_std,
        'rho_hist_mean': rho_hist_mean,
        'rho_hist_std': rho_hist_std,
        'beta_hist_mean': beta_hist_mean,
        'beta_hist_std': beta_hist_std,
        'true_params': results['true_params'],
        'initial_params_mean': results['initial_params_mean'],
    }, output_dir / 'results.pt')
    
    # Create plots
    plot_results(results, output_dir)
    
    # Create trajectory comparison plot
    plot_trajectory_comparison(results, output_dir)
    
    print("\n" + "="*70)
    print("Experiment complete!")
    print("="*70)


if __name__ == "__main__":
    main()
