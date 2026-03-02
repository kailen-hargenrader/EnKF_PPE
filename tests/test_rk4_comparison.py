"""
Test script to compare RK4 integration methods for Lorenz63.

This script compares:
1. Lorentz63._rk4_step (custom RK4 in Lorentz63 class)
2. torchdiffeq.odeint with method='rk4' (external ODE solver)

The key differences and potential sources of disagreement are documented.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from torchdiffeq import odeint

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enkf_ppe.Dynamics.Lorentz63 import Lorentz63, Lorenz63derivs
from enkf_ppe.Utils.rk4 import RK4


class OdeWrapper(nn.Module):
    """Wrapper to match torchdiffeq signature: f(t, x)"""
    def __init__(self, dynamics, theta):
        super().__init__()
        self.dynamics = dynamics
        self.theta = theta
    
    def forward(self, t, x):
        # Expand dimensions if needed to match expected shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
        theta = self.theta.unsqueeze(0) if self.theta.dim() == 1 else self.theta
        result = self.dynamics(x, theta)
        return result.squeeze(0) if result.dim() > 1 else result


def test_rk4_comparison():
    """Compare the two RK4 implementations."""
    
    # Setup
    torch.manual_seed(42)
    dt = 0.01
    num_steps = 1000
    
    # Initial conditions
    x0 = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    theta = torch.tensor([10.0, 28.0, 8.0/3.0], dtype=torch.float32)
    
    print("=" * 80)
    print("RK4 INTEGRATION METHOD COMPARISON")
    print("=" * 80)
    print(f"Initial state: {x0.tolist()}")
    print(f"Parameters (sigma, rho, beta): {theta.tolist()}")
    print(f"Time step (dt): {dt}")
    print(f"Number of steps: {num_steps}")
    print()
    
    # Method 1: Lorentz63._rk4_step (custom, step-by-step)
    print("Method 1: Lorentz63._rk4_step (custom RK4)")
    print("-" * 80)
    lorentz = Lorentz63()
    trajectory_custom = [x0.clone()]
    x_current = x0.clone()
    
    for i in range(num_steps):
        x_current = lorentz(x_current.unsqueeze(0), theta.unsqueeze(0), dt=dt).squeeze(0)
        trajectory_custom.append(x_current.clone())
    
    trajectory_custom = torch.stack(trajectory_custom)
    print(f"Trajectory shape: {trajectory_custom.shape}")
    print(f"Final state: {trajectory_custom[-1].tolist()}")
    print()
    
    # Method 2: torchdiffeq.odeint with method='rk4'
    print("Method 2: torchdiffeq.odeint (method='rk4')")
    print("-" * 80)
    
    # Create time grid for odeint (must include all intermediate points)
    t_eval = torch.linspace(0, num_steps * dt, num_steps + 1)
    
    odefunc = OdeWrapper(Lorenz63derivs(), theta)
    trajectory_odeint = odeint(odefunc, x0, t_eval, method='rk4')
    print(f"Trajectory shape: {trajectory_odeint.shape}")
    print(f"Final state: {trajectory_odeint[-1].tolist()}")
    print()
    
    # Compute differences
    print("DIFFERENCE ANALYSIS")
    print("-" * 80)
    
    # Ensure both trajectories have the same length
    min_len = min(len(trajectory_custom), len(trajectory_odeint))
    diff = trajectory_custom[:min_len] - trajectory_odeint[:min_len]
    
    mae = torch.abs(diff).mean()
    max_error = torch.abs(diff).max()
    rmse = torch.sqrt((diff ** 2).mean())
    
    print(f"Mean Absolute Error (MAE): {mae.item():.2e}")
    print(f"Root Mean Squared Error (RMSE): {rmse.item():.2e}")
    print(f"Max Absolute Error: {max_error.item():.2e}")
    print()
    
    # Per-component analysis
    print("Per-component analysis (x, y, z):")
    for i, name in enumerate(['x', 'y', 'z']):
        mae_i = torch.abs(diff[:, i]).mean()
        max_i = torch.abs(diff[:, i]).max()
        print(f"  {name}: MAE={mae_i.item():.2e}, Max={max_i.item():.2e}")
    print()
    
    return {
        'trajectory_custom': trajectory_custom,
        'trajectory_odeint': trajectory_odeint,
        'diff': diff,
        'mae': mae.item(),
        'rmse': rmse.item(),
        'max_error': max_error.item(),
        't_eval': t_eval,
        'dt': dt,
        'x0': x0,
        'theta': theta,
    }


def plot_comparison(results):
    """Create comparison plots."""
    
    trajectory_custom = results['trajectory_custom']
    trajectory_odeint = results['trajectory_odeint']
    diff = results['diff']
    t_eval = results['t_eval']
    mae = results['mae']
    rmse = results['rmse']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RK4 Integration Method Comparison (Lorenz63)', fontsize=16, fontweight='bold')
    
    # Plot 1: State trajectories overlay
    ax = axes[0, 0]
    ax.plot(t_eval, trajectory_custom[:, 0], label='Custom RK4 (x)', linewidth=2, alpha=0.7)
    ax.plot(t_eval, trajectory_odeint[:, 0], label='odeint RK4 (x)', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('x coordinate')
    ax.set_title('X-component Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: 3D phase space (custom)
    ax = axes[0, 1]
    ax.plot(trajectory_custom[:, 0], trajectory_custom[:, 2], 'b-', label='Custom RK4', linewidth=1.5)
    ax.plot(trajectory_odeint[:, 0], trajectory_odeint[:, 2], 'r--', label='odeint RK4', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_title('Phase Space (x-z plane)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Absolute error over time
    ax = axes[1, 0]
    error_magnitude = torch.norm(diff, dim=1)
    ax.semilogy(t_eval[:len(error_magnitude)], error_magnitude, 'g-', linewidth=2, label='Euclidean error')
    ax.axhline(y=mae, color='r', linestyle='--', label=f'MAE = {mae:.2e}')
    ax.axhline(y=rmse, color='b', linestyle='--', label=f'RMSE = {rmse:.2e}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Error magnitude')
    ax.set_title('Integration Error Over Time (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 4: Per-component errors
    ax = axes[1, 1]
    for i, name in enumerate(['x', 'y', 'z']):
        ax.semilogy(t_eval[:len(error_magnitude)], torch.abs(diff[:, i]), label=f'{name}-component', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Absolute error')
    ax.set_title('Per-Component Error (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    return fig


def generate_analysis_document():
    """Generate a document explaining the potential sources of disagreement."""
    
    analysis = """
ANALYSIS: Why RK4 Implementations Might Disagree
=================================================

1. SIGNATURE DIFFERENCE
   Custom RK4 (Lorentz63._rk4_step):
      - Signature: forward(X, Theta, dt=0.01)
      - Dynamics call: dynamics(X, Theta) -> dX/dt
      - Takes X as [..., n], Theta as [..., 3]
   
   torchdiffeq.odeint (RK4 method):
      - Signature: odeint(odefunc, y0, t, method='rk4', ...)
      - odefunc signature: f(t, y) -> dy/dt  [IMPORTANT: includes time t!]
      - Time-dependent dynamics (even if not used)

2. TIME PARAMETER HANDLING
   The key difference is that torchdiffeq requires the ODE function to accept 
   time as the first argument: f(t, y). Our Lorenz63 doesn't use time explicitly,
   but this can cause subtle issues:
   
   - The OdeWrapper must match the signature exactly
   - torchdiffeq may use adaptive step sizes with options parameter
   - Our custom RK4 uses fixed step sizes exclusively

3. STEP SIZE BEHAVIOR
   Custom RK4:
      - Fixed step size: always uses the provided dt value
      - Deterministic: same result every time
   
   torchdiffeq RK4:
      - Can use adaptive stepping by default
      - Our code specifies method='rk4' (standard RK4, not adaptive)
      - But torchdiffeq may apply small corrections or different tolerances

4. DIMENSION HANDLING
   torchdiffeq expects:
      - Input shape: (n,) or (batch, n)
      - Output shape: (T, n) or (T, batch, n)
   
   Our wrapper must properly squeeze/unsqueeze dimensions to match.
   Any mismatch here causes broadcasting errors or incorrect computations.

5. NUMERICAL DIFFERENCES
   Even with identical algorithms, numerical precision can differ due to:
      - Different order of operations (floating-point associativity)
      - Compiler optimizations
      - PyTorch backend differences (CPU vs GPU)

6. INTEGRATION TIME GRID
   - Custom RK4: takes fixed steps of size dt, implicit time grid
   - torchdiffeq: requires explicit time grid as input (t_eval)
   - If t_eval doesn't align with desired dt, solver may take different steps

RECOMMENDATIONS FOR INVESTIGATION
==================================

Step 1: Check step-by-step agreement
   Compare one RK4 step from each method. If they disagree on a single step,
   the issue is in the algorithm or dimension handling.

Step 2: Verify dimension shapes
   Add assertions to confirm:
      - x has shape (1, 3) when passed to odeint
      - Output has expected shape (T, 3)
      - No unexpected broadcasting

Step 3: Test time grid sensitivity
   Try different explicit time grids in torchdiffeq to see if the step size matters.

Step 4: Check for time-dependency
   Verify Lorenz63 equations don't have implicit time dependence hidden somewhere.

Step 5: Compare accumulated error
   If single steps agree but trajectories diverge, it's likely floating-point 
   accumulation. Very small differences compound in chaotic systems like Lorenz!
"""
    
    return analysis


if __name__ == "__main__":
    # Run comparison
    results = test_rk4_comparison()
    
    # Create plots
    fig = plot_comparison(results)
    
    # Save figure
    test_dir = Path(__file__).parent
    test_dir.mkdir(exist_ok=True)
    fig_path = test_dir / "rk4_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {fig_path}")
    
    # Save analysis
    analysis = generate_analysis_document()
    analysis_path = test_dir / "rk4_comparison_analysis.txt"
    with open(analysis_path, 'w') as f:
        f.write(analysis)
    print(f"Analysis saved to: {analysis_path}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(analysis)
