import torch
from torch import Tensor
from abc import ABC, abstractmethod
import torch.nn as nn


class BaseModel(ABC):
    """
    Base class for all timeseriese forecasting models.
    """
    @abstractmethod
    def run(self, X0: Tensor, theta0: Tensor, observations: Tensor, trans_dt: float, obs_dt: float, trans_fn: nn.Module, obs_fn: nn.Module) -> [Tensor, Tensor]:
        """
        Run the model over a full observation sequence.

        Args:
            X0:          initial state      (N, n)
            theta0:      initial parameters (N, p)
            observations: observation time series  (T, m)
            obs_dt:      time step between observations
            trans_dt:    time step for the model transition
            trans_fn:    transition function (X, theta) -> dX/dt
            obs_fn:      observation function (X) -> Y_hat
        
        Returns:
            X_hist: state history
            theta_hist: parameter history 
            ***see child classes for shape details***
        """
        pass