# Setup

## uv

- install uv with `pip install uv`
- sync environment with `uv sync`

# Ensemble Kalman Filter (EnKF) for Physical Parameter Estimation (PPE)

This repository implements the Ensemble Kalman Filter (EnKF) for time series forecasting and Physical Parameter Estimation (PPE). This approach allows for the joint estimation of the state of a physical system and its underlying physical parameters using an ensemble-based data assimilation technique.

## 1. Formal Description of the Timeseries Forecasting Problem

We consider a discrete-time nonlinear dynamical system. The goal is to estimate the state of the system $x_k \in \mathbb{R}^n$ and a set of physical parameters $\theta \in \mathbb{R}^p$ at time $k$, given a sequence of noisy observations $y_{1:k} = \{y_1, y_2, \dots, y_k\}$, where $y_k \in \mathbb{R}^m$.

The system is governed by the following state-space model:

### State Transition Model (Evolution)
$$x_k = \Psi(x_{k-1}; \theta) + \eta_k$$
where:
- $\Psi(\cdot)$ is the nonlinear transition function (often a numerical model of physical processes).
- $\eta_k \sim \mathcal{N}(0, \Sigma)$ is the process noise, representing model deficiencies and stochasticity.

### Observation Model
$$y_k = h(x_k) + \xi_k$$
where:
- $h(\cdot)$ is the nonlinear observation operator mapping the state space to the observation space.
- $\xi_k \sim \mathcal{N}(0, \Gamma)$ is the observation noise.

### The Objective
The objective is to find the posterior distribution $p(x_k, \theta | y_{1:k})$. In practice, we often seek the mean and covariance of this distribution to provide an optimal estimate and its uncertainty.

---

## 2. Ensemble Kalman Filter (EnKF)

The EnKF is a Monte Carlo implementation of the Kalman Filter. Instead of evolving the state covariance matrix explicitly (which is computationally prohibitive for high-dimensional systems), the EnKF represents the state distribution using an ensemble of $N$ realizations:
$$\mathbf{X}_k = [x_k^{(1)}, x_k^{(2)}, \dots, x_k^{(N)}]$$

The algorithm proceeds in two main steps:

### Forecast (Predict) Step
Each ensemble member is evolved forward in time using the model:
$$\hat{x}_k^{(i)} = \Psi(x_{k-1}^{(i)}, \theta_{k-1}^{(i)}) + \eta_k^{(i)}$$
This step produces the background (prior) ensemble.

### Analysis (Update) Step
When a new observation $y_k$ becomes available, each ensemble member is updated using the Kalman gain $K_k$:
$$x_k^{(i)} = \hat{x}_k^{(i)} + K_k (y_k + \xi_k^{(i)} - h(\hat{x}_k^{(i)}))$$
where $\xi_k^{(i)}$ are synthetic observation perturbations.

The Kalman gain $K_k$ is computed using the sample covariance $P_k^f$ of the forecast ensemble:
$$P_k^f = \frac{1}{N-1} \sum_{i=1}^N (\hat{x}_k^{(i)} - \bar{\hat{x}}_k) (\hat{x}_k^{(i)} - \bar{\hat{x}}_k)^T$$

The gain is then expressed in the standard Kalman form:
$$K_k = P_k^f \mathcal{H}^T (\mathcal{H} P_k^f \mathcal{H}^T + \Gamma)^{-1}$$
where $\mathcal{H}$ is the observation operator.

To avoid the explicit computation of the large covariance matrix $P_k^f$ and save memory, we use the ensemble perturbation matrix $A$:
$$A = [\hat{x}_k^{(1)} - \bar{\hat{x}}_k, \dots, \hat{x}_k^{(N)} - \bar{\hat{x}}_k]$$
The Kalman gain is then computed as:
$$K_k = \frac{1}{N-1} AA^T \mathcal{H}^T \left( \frac{1}{N-1} \mathcal{H} AA^T \mathcal{H}^T + \Gamma \right)^{-1}$$

---
