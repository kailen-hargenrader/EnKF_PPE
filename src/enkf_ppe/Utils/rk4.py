import torch
import torch.nn as nn

class RK4(nn.Module):
    def __init__(self, dynamics: nn.Module):
        """
        Args:
            dynamics: the dynamics function (X, Theta) -> dX/dt
        """
        super().__init__()
        self.dynamics = dynamics
    
    def forward(self, X, Theta, *, dt):
        k1 = self.dynamics(X, Theta)
        k2 = self.dynamics(X + 0.5 * dt * k1, Theta)
        k3 = self.dynamics(X + 0.5 * dt * k2, Theta)
        k4 = self.dynamics(X + dt * k3, Theta)
        return X + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)