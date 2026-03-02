"""Neural ODE with Adjoint Sensitivity Method

This model uses the torchdiffeq library to compute parameter gradients with 
O(1) memory cost. Instead of backpropagating through every solver step, 
it solves an adjoint ODE backward in time.

Notation:
  n  – state dimension
  p  – parameter dimension
  m  – observation dimension
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable
from torchdiffeq import odeint_adjoint as odeint

from enkf_ppe.Models.base_model import BaseModel


class NeuralODEAdjoint(nn.Module, BaseModel):
    """
    Neural ODE implementation using the Adjoint Sensitivity Method.

    Args:
        n:                 state dimension
        p:                 parameter dimension
        lr:                learning rate for the parameter optimizer
    """

    def __init__(
        self,
        epochs_per_run:    int = 10,
        lr:                float = 1e-2,
        max_grad_norm:     float = 1.0,
    ) -> None:
        super().__init__()
        self.epochs_per_run = epochs_per_run
        self.lr    = lr
        self.max_grad_norm = max_grad_norm

    def run(self, X0: Tensor, theta0: Tensor, observations: Tensor, trans_dt: float, obs_dt: float, trans_fn: nn.Module, obs_fn: nn.Module, learn_params_mask: Tensor = None) -> [Tensor, Tensor]:
        """
        Run the model over a full observation sequence.

        Args:
            X0:               initial state      (1, n)
            theta0:           initial parameters (1, p)
            observations:     observation time series  (T, m)
            obs_dt:           time step between observations
            trans_dt:         time step for the model transition
            trans_fn:         transition function (X, theta) -> dX/dt
            obs_fn:           observation function (X) -> Y_hat
            learn_params_mask: boolean tensor (p,) indicating which parameters to learn
        
        Returns:
            X_hist: state history  (T, Epochs, n)
            theta_hist: parameter history  (Epochs, p)
            loss_hist: loss history  (Epochs,)
        """

        # ── get basic metadata ──
        n_forecasts = int(round(obs_dt / trans_dt))
        n = X0.shape[1]
        p = theta0.shape[1]
        m = observations.shape[1]
        T = observations.shape[0]

        # ── check input dimensions ──
        assert abs(n_forecasts * trans_dt - obs_dt) < 1e-9 * obs_dt, (
            f"model time step ({trans_dt}) must divide the observation time step ({obs_dt})"
        )
        assert X0.shape[0] == theta0.shape[0] == 1, "No ensemble support for Neural ODE"

        # ── make time grid ──
        t_steps = torch.linspace(0, T * obs_dt, T * n_forecasts)

        # Internal wrapper to match torchdiffeq signature: f(t, x)
        # theta is captured from the outer scope
        class OdeWrapper(nn.Module):
            def __init__(self, dynamics, theta):
                super().__init__()
                self.dynamics = dynamics
                self.theta = nn.Parameter(theta)

            def forward(self, t, x):
                # theta is [sigma, rho, beta]
                return self.dynamics(x, self.theta)
        odefunc = OdeWrapper(dynamics=trans_fn, theta=theta0.squeeze(0))
        
        # ── initialize parameters ──
        optimizer = torch.optim.Adam(odefunc.parameters(), lr=self.lr)

        # ── initialize history ──
        X_epoch_hist = []
        theta_epoch_hist = []
        loss_epoch_hist = []

        # ── run optimization ──
        for _ in range(self.epochs_per_run):
            optimizer.zero_grad()

            # 1. FORWARD PASS: Solve ODE using the Adjoint Method wrapper
            # method='rk4' ensures it uses the same integration logic as your EnKF
            X_traj = odeint(odefunc, X0, t_steps, method='rk4') # (T * n_forecasts, 1, n)
            X_pred = X_traj[::n_forecasts].squeeze(1) # (T, n)
            
            # 2. LOSS: Match trajectory to observations
            Y_hat = obs_fn(X_pred) # (T, m)
            loss = torch.mean((Y_hat - observations)**2)
            
            # 3. BACKWARD PASS: Compute dLoss/dTheta via Adjoint ODE
            loss.backward()
            
            # 4. APPLY PARAMETER MASK: Zero gradients for fixed parameters
            if learn_params_mask is not None:
                with torch.no_grad():
                    for i, should_learn in enumerate(learn_params_mask):
                        if not should_learn:
                            odefunc.theta.grad[i] = 0.0
            
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(odefunc.parameters(), self.max_grad_norm)
                
            optimizer.step()
            
            # 5. ENFORCE FIXED PARAMETERS: Reset fixed parameters to their initial values
            if learn_params_mask is not None:
                with torch.no_grad():
                    for i, should_learn in enumerate(learn_params_mask):
                        if not should_learn:
                            odefunc.theta[i] = theta0.squeeze(0)[i]
            
            # Record results for this epoch
            X_epoch_hist.append(X_pred.detach())
            theta_epoch_hist.append(odefunc.theta.detach().clone())
            loss_epoch_hist.append(loss.detach().clone())
            
        return torch.stack(X_epoch_hist, dim=1), torch.stack(theta_epoch_hist, dim=0), torch.stack(loss_epoch_hist, dim=0) # (T, Epochs, n), (Epochs, p), (Epochs,)