import torch
import torch.nn as nn


class Lorenz63derivs(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, state, params):
        """
        Calculates the derivatives of the Lorenz '63 system.

        Args:
            state (torch.Tensor): The current state [x, y, z].
            params (torch.Tensor): The parameters [sigma, rho, beta].
            
        Returns:
            torch.Tensor: The derivatives [dx/dt, dy/dt, dz/dt].
        """
        sigma, rho, beta = params[..., 0], params[..., 1], params[..., 2]
        x, y, z = state[..., 0], state[..., 1], state[..., 2]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return torch.stack([dx, dy, dz], dim=-1)


class Lorentz63(nn.Module):
    def __init__(self):
        super().__init__()
        self.get_derivs = Lorenz63derivs()

    def _rk4_step(self, X, Theta, *, dt):
        """
        One Runge-Kutta 4 integration step matching the Ψ signature expected by
        StateAugEnKF.  Partially initialise with dt to obtain psi:

            psi = functools.partial(rk4_step, dt=0.01)
            # psi(X, Theta) -> X_next

        Args:
            X     (torch.Tensor): State      [..., n].
            Theta (torch.Tensor): Parameters [..., 3]  – [sigma, rho, beta].
            dt    (float):        Integration time step.

        Returns:
            torch.Tensor: Next state [..., n].
        """
        k1 = self.get_derivs(X, Theta)
        k2 = self.get_derivs(X + 0.5 * dt * k1, Theta)
        k3 = self.get_derivs(X + 0.5 * dt * k2, Theta)
        k4 = self.get_derivs(X + dt * k3, Theta)
        return X + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def forward(self, X, Theta, dt=0.01):
        """
        Forward pass of the Lorenz '63 system.

        Args:
            X     (torch.Tensor): State      [..., n].
            Theta (torch.Tensor): Parameters [..., 3]  – [sigma, rho, beta].
            dt    (float):        Integration time step.

        Returns:
            torch.Tensor: Next state [..., n].
        """
        return self._rk4_step(X, Theta, dt=dt)



