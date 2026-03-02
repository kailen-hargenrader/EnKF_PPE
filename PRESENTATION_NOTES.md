# Physical Parameter Estimation: Three Approaches

## Setup: The Problem

We want to estimate the **state** $x_k \in \mathbb{R}^n$ and **physical parameters** $\theta \in \mathbb{R}^p$ of a dynamical system from noisy observations $y_k \in \mathbb{R}^m$.

### State-Space Model
- **Dynamics**: $x_k = \Psi(x_{k-1}; \theta) + \eta_k$ where $\eta_k \sim \mathcal{N}(0, \Sigma)$
- **Observations**: $y_k = h(x_k) + \xi_k$ where $\xi_k \sim \mathcal{N}(0, \Gamma)$

**System**: Lorenz-63 with true parameters $\theta^* = [\sigma, \rho, \beta]^T = [10, 28, 2.6667]^T$

---

## Method 1: Neural ODE Baseline (Naive Parameter Learning)

### Core Idea
Treat **parameter estimation as global trajectory optimization**. Learn parameters via gradient descent by minimizing the prediction error over the entire observation horizon.

### Mathematical Formulation

**Continuous Dynamics**:
$$\frac{dx}{dt} = f(x, t; \theta)$$

**Predicted Trajectory**:
$$\hat{x}_{1:T} = \text{ODESolve}(x_0, f, \theta, t_{1:T})$$

**Loss Function** (MSE):
$$L(\theta) = \sum_{k=1}^{T} \|y_k - h(\hat{x}_k(\theta))\|^2$$

**Gradient Computation** (Adjoint Sensitivity Method):
- Solve adjoint ODE backward in time:
  $$\frac{da}{dt} = -a(t)^T \frac{\partial f}{\partial x}$$
- Compute gradient: $\nabla_\theta L(\theta)$ with **O(1) memory** (no backprop through time)

**Parameter Update** (Adam Optimizer):
$$\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)$$

### Why This Approach?

✓ **Pros**:
- Simple and intuitive: minimize global prediction error
- Memory-efficient gradient computation (adjoint method)
- Treats time continuously, good for irregularly-sampled data

✗ **Cons**:
- **No sequential uncertainty**: doesn't track how certainty improves with new observations
- Assumes we see the entire trajectory before learning
- Requires good initial conditions and parameter initialization
- Single batch optimization over entire horizon

### Implementation
- Framework: PyTorch + torchdiffeq
- Parameters initialized with noise: $\theta_0 \sim \mathcal{N}(\theta^*, \sigma_{init}^2)$
- Training: 500-1000 epochs with learning rate $\alpha = 0.01$

---

## Method 2: State Augmented EnKF (Parameter Learning via Kalman Update)

### Core Idea
Treat **parameters as part of the state**. Use the EnKF to jointly estimate state and parameters sequentially, leveraging cross-correlations between state and parameter forecast errors.

### Mathematical Formulation

**Augmented State**:
$$z_k = \begin{bmatrix} x_k \\ \theta_k \end{bmatrix}$$

**Augmented Transition**:
$$\begin{bmatrix} x_k \\ \theta_k \end{bmatrix} = \begin{bmatrix} \Psi(x_{k-1}, \theta_{k-1}) \\ \theta_{k-1} \end{bmatrix} + \begin{bmatrix} \eta_k \\ \zeta_k \end{bmatrix}$$

where:
- $\eta_k \sim \mathcal{N}(0, \Sigma)$ = model process noise
- $\zeta_k \sim \mathcal{N}(0, Q)$ = **artificial parameter noise** (crucial!)

**Augmented Observation**:
$$y_k = h(x_k, \theta_k) + \xi_k$$

**EnKF Algorithm**:

1. **Forecast Step**: Evolve ensemble forward
   - Generate ensemble: $\mathbf{X}_k = [x_k^{(1)}, \ldots, x_k^{(N)}]$
   - Compute forecast ensemble mean: $\bar{\hat{x}}_k$, $\bar{\hat{\theta}}_k$
   - Compute forecast covariance $P_k^f$ (includes state-parameter cross-covariance!)

2. **Update Step**: Use Kalman gain to correct state AND parameters
   - Compute Kalman gain:
     $$K_k = P_k^f \mathcal{H}^T (\mathcal{H} P_k^f \mathcal{H}^T + \Gamma)^{-1}$$
   - Update each ensemble member:
     $$z_k^{(i)} = \hat{z}_k^{(i)} + K_k (y_k + \xi_k^{(i)} - h(\hat{z}_k^{(i)}))$$
   - This update affects both $x_k^{(i)}$ and $\theta_k^{(i)}$

### The Parameter Noise Role

The artificial parameter noise $\zeta_k$ is **essential**:
- **Prevents collapse**: Without noise, ensemble parameters converge to a single value
- **Maintains spread**: Keeps exploring the parameter space
- **Sequential learning**: Each new observation constrains parameters via the Kalman gain

Typical: $Q = \text{diag}(\sigma_\zeta^2)$ where $\sigma_\zeta \ll \sigma_{init}$ (e.g., $10^{-5}$ or $10^{-6}$)

### Why This Approach?

✓ **Pros**:
- **Sequential uncertainty quantification**: encodes how parameters improve as data arrives
- **Cross-correlations**: observations update both state and parameters jointly
- **Data assimilation framework**: statistically principled, Bayesian
- Handles partial observations naturally

✗ **Cons**:
- Requires tuning parameter noise $Q$ (crucial hyperparameter!)
- Ensemble collapse risk if $Q$ is too small
- Requires sufficient ensemble size $N$
- Slower than batch methods (sequential updates)

### Implementation
- Ensemble size: $N = 100-200$ members
- Parameter noise: $Q = 10^{-6} \cdot \text{diag}(I_p)$ (tiny but non-zero)
- Integration: RK4 with $dt = 0.01$

---

## Method 3: Autodifferentiable EnKF (Gradient-Based Parameter Learning)

### Core Idea
Make the **entire EnKF process differentiable**. Learn parameters via gradient descent by backpropagating through the forecast and analysis steps.

### Mathematical Formulation

**Differentiable EnKF Components**:
1. Forecast: $\hat{\mathbf{X}}_k = \Psi(\mathbf{X}_{k-1}; \theta)$ (differentiable dynamics)
2. Kalman gain: $K_k(\theta)$ (computed from perturbed observations)
3. Analysis: $\mathbf{X}_k = \hat{\mathbf{X}}_k + K_k (y_k + \text{noise} - h(\hat{\mathbf{X}}_k))$

**Loss Function** (Negative Log-Likelihood):
$$L(\theta) = \sum_{k=1}^{T} \left[ \log|\mathbf{S}_k(\theta)| + (y_k - h(\bar{\hat{x}}_k(\theta)))^T \mathbf{S}_k(\theta)^{-1} (y_k - h(\bar{\hat{x}}_k(\theta))) \right]$$

where:
- $\mathbf{S}_k(\theta) = \mathcal{H} P_k^f(\theta) \mathcal{H}^T + \Gamma$ = innovation covariance (depends on $\theta$!)
- $\bar{\hat{x}}_k(\theta)$ = ensemble mean from forecast step

Alternative simpler loss (MSE):
$$L(\theta) = \sum_{k=1}^{T} \|y_k - h(\bar{\hat{x}}_k(\theta))\|^2$$

**Gradient Computation**:
$$\nabla_\theta L(\theta) = \text{backprop through EnKF operations}$$

All operations must be implemented in a differentiable framework (PyTorch/JAX).

**Parameter Update**:
$$\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)$$

### Why This Approach?

✓ **Pros**:
- **Hybrid of best of both worlds**: sequential ensemble updates + global gradient optimization
- **Principled likelihood**: uses full ensemble covariance in loss
- **Optimal for high-dimensional parameters**: gradient descent scales better
- **Adaptive learning**: data-driven through innovation covariance

✗ **Cons**:
- **Implementation complexity**: entire EnKF must be differentiable
- **Computational cost**: backprop through all timesteps
- **Memory**: must store activations for gradient computation
- Requires careful numerical stability (matrix inversions in gradients)

### Implementation
- EnKF updates are computed in PyTorch (forward-mode or automatic differentiation)
- Loss computed over observation window
- Adam optimizer with learning rate $\alpha = 0.001-0.01$

---

## Comparison Table

| Aspect | Neural ODE Baseline | State Aug. EnKF | Autodiff EnKF |
|--------|-------------------|-----------------|---------------|
| **Parameter Evolution** | Global trajectory fit | Sequential Kalman updates | Sequential + gradient descent |
| **Uncertainty Quantification** | None | Ensemble spread via $Q$ | Ensemble + gradient info |
| **Cross-Correlations** | No | **Yes** (state-param) | **Yes** (learned via gradients) |
| **Memory Cost** | O(1) (adjoint) | O(N) (ensemble) | O(N·T) (backprop) |
| **Computational Speed** | Fast | Medium | Slower |
| **Parameter Initialization** | Sensitive | Robust (ensemble) | Moderately robust |
| **Hyperparameter Tuning** | Learning rate $\alpha$ | **Parameter noise $Q$** | Learning rate $\alpha$ |
| **Data Usage Pattern** | Batch over full horizon | Sequential, one obs at a time | Mini-batches or full horizon |

---

## Key Insights for Your Talk

### 1. **Why Parameters Need Noise in State Augmentation**
- Kalman filter updates are proportional to forecast covariance
- Without artificial noise $\zeta_k$, parameters converge to single ensemble value
- Cross-covariance $\text{Cov}(x_k^f, \theta_k^f)$ becomes zero → no parameter updates!
- $Q$ is the "temperature" of parameter exploration

### 2. **Neural ODE vs Sequential Methods**
- **Neural ODE**: "Here's all the data. Find parameters that fit the whole trajectory."
- **EnKF approaches**: "Here's one observation. Update state and parameters. Repeat."
- Different information flows → different convergence properties

### 3. **When to Use Which**
- **Neural ODE**: Offline learning, known initial conditions, simple parameter spaces
- **State Aug. EnKF**: Real-time filtering, online learning, need to track uncertainty
- **Autodiff EnKF**: When you want sequential Bayesian inference with gradient-based scaling

### 4. **The Parameter Noise Paradox**
- Without $\zeta_k$: parameters don't learn (zero cross-covariance)
- Too much $\zeta_k$: parameters won't converge (keep drifting)
- Goldilocks zone: $10^{-6}$ to $10^{-5}$ often works well

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| $x_k$ | System state at time $k$ |
| $\theta$ | Physical parameters (constant or slowly varying) |
| $\Psi(\cdot)$ | Nonlinear state transition function |
| $h(\cdot)$ | Observation operator |
| $y_k$ | Observed data at time $k$ |
| $\eta_k, \xi_k$ | Process and observation noise |
| $\zeta_k$ | **Artificial parameter noise** (EnKF-specific) |
| $P_k^f$ | Forecast covariance (or ensemble approximation) |
| $K_k$ | Kalman gain |
| $\mathcal{H}$ | Jacobian of observation operator |
| $\Gamma$ | Observation noise covariance |
| $\Sigma$ | Process noise covariance |
| $L(\theta)$ | Loss/objective function |
| $\alpha$ | Learning rate |
| $N$ | Ensemble size |
