"""
Experiment: Learn Sigma Parameter with Perfect Observations

This experiment tests the NeuralODE's ability to fit Lorentz63 data perfectly
when supplied with the true dynamics function but a very wrong initial guess
of the sigma parameter.

Goal: Verify that the NeuralODE can recover the true sigma parameter
from noise-free observations when the dynamics model is correct.

Configuration is loaded from config.yaml using Hydra.
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
from enkf_ppe.Models.NeuralODE.neural_ode import NeuralODEAdjoint
from enkf_ppe.Utils.observation_fns import FullObservation
    

def load_data(config: DictConfig) -> torch.Tensor:
    """
    Load the Lorentz63 dataset.
    
    Args:
        config: Experiment configuration (DictConfig)
        
    Returns:
        observations: (T, 3) tensor of observations
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
    
    print(f"Loaded data shape: {observations.shape}")
    print(f"Data file: {data_path}")
    
    return observations


def create_dynamics_wrapper(config: DictConfig) -> Tuple[nn.Module, nn.Module]:
    """
    Create the dynamics and observation functions.
    
    The dynamics function is the TRUE Lorentz63 system (not learned).
    This tests whether NeuralODE can recover parameters when the model is correct.
    
    Args:
        config: Experiment configuration
        
    Returns:
        trans_fn: transition function (X, theta) -> dX/dt
        obs_fn: observation function X -> Y
    """
    # Dynamics: true Lorentz63 system
    dynamics = Lorenz63derivs()
    
    # Observation: full state observation (identity function)
    obs_fn = FullObservation()
    
    return dynamics, obs_fn


def run_experiment(config: DictConfig) -> Dict:
    """
    Run the learn_sigma_perfect experiment.
    
    Args:
        config: Experiment configuration (DictConfig)
        
    Returns:
        results: Dictionary containing:
            - X_hist: state trajectory history (T, Epochs, 3)
            - theta_hist: parameter history (Epochs, 3)
            - true_params: true parameters
            - initial_params: initial parameter guess
            - observations: observation data
    """
    # Extract config values
    true_params = torch.tensor([[
        config.parameters.truth.sigma,
        config.parameters.truth.rho,
        config.parameters.truth.beta
    ]])
    
    initial_params = torch.tensor([[
        config.parameters.initial_guess.sigma,
        config.parameters.initial_guess.rho,
        config.parameters.initial_guess.beta
    ]])
    
    learn_params_mask = torch.tensor([
        config.parameters.learn_mask.sigma,
        config.parameters.learn_mask.rho,
        config.parameters.learn_mask.beta
    ])
    
    x0 = torch.tensor([[
        config.initial_condition.x,
        config.initial_condition.y,
        config.initial_condition.z
    ]])
    
    dt_model = config.data.dt_model
    dt_obs = config.data.dt_obs
    epochs = int(config.neural_ode.epochs)
    learning_rate = config.neural_ode.learning_rate
    max_grad_norm = config.neural_ode.max_grad_norm
    print("\n" + "="*70)
    print("EXPERIMENT: Learn Sigma Parameter with Perfect Observations")
    print("="*70)
    
    # Load data
    print("\n[1/4] Loading data...")
    observations = load_data(config)
    
    # Create dynamics and observation functions
    print("[2/4] Setting up dynamics...")
    trans_fn, obs_fn = create_dynamics_wrapper(config)
    
    # Create NeuralODE model
    print("[3/4] Setting up NeuralODE model...")
    model = NeuralODEAdjoint(
        epochs_per_run=epochs,
        lr=learning_rate,
        max_grad_norm=max_grad_norm,
    )
    
    # Run the model
    print("[4/4] Running optimization...")
    print(f"  - Initial sigma: {initial_params[0, 0].item():.4f}")
    print(f"  - True sigma:    {true_params[0, 0].item():.4f}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Epochs: {epochs}")
    
    X_hist, theta_hist, loss_hist = model.run(
        X0=x0,
        theta0=initial_params.clone(),
        observations=observations,
        trans_dt=dt_model,
        obs_dt=dt_obs,
        trans_fn=trans_fn,
        obs_fn=obs_fn,
        learn_params_mask=learn_params_mask,
    )
    
    # Compute final metrics
    final_sigma = theta_hist[-1, 0].item()
    true_sigma = true_params[0, 0].item()
    sigma_error = abs(final_sigma - true_sigma)
    sigma_relative_error = sigma_error / true_sigma * 100
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Final sigma:          {final_sigma:.6f}")
    print(f"True sigma:           {true_sigma:.6f}")
    print(f"Absolute error:       {sigma_error:.6f}")
    print(f"Relative error:       {sigma_relative_error:.2f}%")
    
    results = {
        'X_hist': X_hist,
        'theta_hist': theta_hist,
        'loss_hist': loss_hist,
        'true_params': true_params,
        'initial_params': initial_params,
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
    
    T, epochs, n = X_hist.shape
    
    # Figure 1: Sigma parameter convergence
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sigma_hist = theta_hist[:, 0].numpy()
    true_sigma = results['true_params'][0, 0].item()
    initial_sigma = results['initial_params'][0, 0].item()
    
    # Prepend initial sigma value to show starting point
    sigma_hist_with_initial = np.concatenate([[initial_sigma], sigma_hist])
    epochs_axis = np.arange(len(sigma_hist_with_initial))
    
    ax.plot(epochs_axis, sigma_hist_with_initial, 'b-', linewidth=2, label='Estimated sigma')
    ax.axhline(true_sigma, color='r', linestyle='--', linewidth=2, label='True sigma')
    ax.scatter([0], [initial_sigma], color='g', s=100, marker='o', 
               label='Initial guess', zorder=5, edgecolors='darkgreen', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Sigma value', fontsize=12)
    ax.set_title('Sigma Parameter Convergence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sigma_convergence.png', dpi=150)
    plt.close()
    
    # Figure 3: Parameter history
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    param_names = ['sigma', 'rho', 'beta']
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        param_hist = theta_hist[:, i].numpy()
        true_val = results['true_params'][0, i].item()
        initial_val = results['initial_params'][0, i].item()
        
        ax.plot(param_hist, 'b-', linewidth=2, label='Estimated')
        ax.axhline(true_val, color='r', linestyle='--', linewidth=2, label='True')
        ax.axhline(initial_val, color='g', linestyle=':', linewidth=2, label='Initial guess')
        
        ax.set_ylabel(name, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Epoch', fontsize=12)
    fig.suptitle('Parameter Evolution During Training', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_evolution.png', dpi=150)
    plt.close()
    
    # Figure 4: Prediction error over epochs
    fig, ax = plt.subplots(figsize=(10, 6))
    
    loss_hist = results['loss_hist']
    
    # Compute loss for initial parameters (before first epoch)
    # We need to generate trajectory with initial params to compute initial loss
    from enkf_ppe.Dynamics.Lorentz63 import Lorentz63
    dynamics_rk4 = Lorentz63()
    
    x0 = observations[0:1]
    initial_params_full = results['initial_params']
    rho = results['true_params'][0, 1].item()
    beta = results['true_params'][0, 2].item()
    initial_sigma = initial_params_full[0, 0].item()
    
    dt_obs = float(results['config'].data.dt_obs)
    dt_model = float(results['config'].data.dt_model)
    X_initial = x0.clone()
    initial_traj = [X_initial.numpy()]
    initial_param_tensor = torch.tensor([[initial_sigma, rho, beta]])
    
    for _ in range(len(observations) - 1):
        # Use RK4 integration with model time step
        n_steps = int(round(dt_obs / dt_model))
        for _ in range(n_steps):
            X_initial = dynamics_rk4(X_initial, initial_param_tensor, dt=dt_model)
        initial_traj.append(X_initial.detach().numpy())
    
    initial_traj = torch.tensor(initial_traj).squeeze()
    
    epoch_axis = np.arange(len(loss_hist))
    ax.semilogy(epoch_axis, loss_hist, 'b-', linewidth=2, marker='o', markersize=4)
    ax.scatter([0], [loss_hist[0]], color='orange', s=100, marker='s', 
               label='Initial loss', zorder=5, edgecolors='darkorange', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss (log scale)', fontsize=12)
    ax.set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_loss.png', dpi=150)
    plt.close()
    
    print(f"\nPlots saved to: {output_dir}")


def plot_trajectory_comparison(results: Dict, output_dir: Path, trans_fn: nn.Module) -> None:
    """
    Plot trajectories for true sigma, initial sigma, and final sigma.
    
    Args:
        results: Results dictionary from run_experiment
        output_dir: Directory to save plot
        trans_fn: Dynamics function to generate trajectories (used for derivatives only)
    """
    from enkf_ppe.Dynamics.Lorentz63 import Lorentz63
    
    observations = results['observations']
    true_params = results['true_params']
    initial_params = results['initial_params']
    theta_hist = results['theta_hist']
    
    final_sigma = theta_hist[-1, 0].item()
    initial_sigma = initial_params[0, 0].item()
    
    # Get other parameters (rho, beta) from true params
    rho = true_params[0, 1].item()
    beta = true_params[0, 2].item()
    
    # Get initial state from observations
    x0 = observations[0:1]
    dt_obs = results['config'].data.dt_obs
    dt_model = results['config'].data.dt_model
    
    # Initialize RK4 dynamics
    dynamics_rk4 = Lorentz63()
    
    # Generate trajectories for each sigma value
    trajectories = {}
    for label, sigma_val in [ ('initial', initial_sigma), 
                              ('final', final_sigma)]:
        traj = [x0.numpy()]
        X = x0.clone()
        params = torch.tensor([[sigma_val, rho, beta]])
        
        for _ in range(len(observations) - 1):
            # Use RK4 integration with model time step
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
                label='Ground truth (σ=10.0)', alpha=0.8, zorder=4)
        
        # Plot trajectories
        ax.plot(time_steps, trajectories['initial'][:, i], 'orange', 
                linestyle=':', linewidth=2, label=f'Initial σ ({initial_sigma:.4f})', 
                alpha=0.7, zorder=2)
        ax.plot(time_steps, trajectories['final'][:, i], 'b--', linewidth=2, 
                label=f'Final σ ({final_sigma:.4f})', alpha=0.7, zorder=3)
        
        ax.set_ylabel(f'{name.upper()}(t)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
    
    axes[-1].set_xlabel('Time step', fontsize=12, fontweight='bold')
    fig.suptitle('Trajectory Comparison: Different Sigma Values', 
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
    
    # Extract sigma estimates over epochs
    sigma_hist = results['theta_hist'][:, 0]
    
    # Save PyTorch results
    torch.save({
        'X_hist': results['X_hist'],
        'theta_hist': results['theta_hist'],
        'sigma_hist': sigma_hist,
        'true_params': results['true_params'],
        'initial_params': results['initial_params'],
    }, output_dir / 'results.pt')
    
    # Create plots
    plot_results(results, output_dir)
    
    # Create trajectory comparison plot
    trans_fn, _ = create_dynamics_wrapper(config)
    plot_trajectory_comparison(results, output_dir, trans_fn)
    
    print("\n" + "="*70)
    print("Experiment complete!")
    print("="*70)


if __name__ == "__main__":
    main()
