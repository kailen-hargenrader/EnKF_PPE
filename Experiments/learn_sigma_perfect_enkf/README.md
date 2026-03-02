# Experiment: Learn Sigma Parameter with Perfect Observations (EnKF)

## Overview

This is the **ensemble Kalman filter equivalent** of the `learn_sigma_perfect` experiment. It tests the **State Augmented EnKF** model's ability to learn the `sigma` parameter of Lorentz63 from perfectly observed (noise-free) data.

## Comparison: NeuralODE vs State Augmented EnKF

| Aspect | NeuralODE | State Augmented EnKF |
|--------|-----------|----------------------|
| **Paradigm** | Batch gradient-based optimization | Sequential Bayesian estimation |
| **Ensemble** | Single member | Multiple members (N=50) |
| **Parameter Learning** | Via backprop through ODE solver | Via cross-correlations in covariance |
| **Uncertainty** | Point estimate only | Full ensemble-based uncertainty |
| **Computational** | GPU-friendly, differentiable | Requires ensemble propagations |
| **Dynamics** | Supplied as derivatives | Supplied as derivatives |

## Motivation

Why compare two approaches?

1. **Validation**: Different algorithms should converge to the same parameters
2. **Uncertainty Quantification**: EnKF provides ensemble-based confidence bounds
3. **Scalability**: Understand trade-offs between batch and sequential methods
4. **Robustness**: Test parameter learning under different estimation paradigms

## Experiment Setup

### True System
- **System:** Lorentz63 with σ=10.0, ρ=28.0, β=2.6667
- **Data:** 500 time steps at dt=0.1
- **Initial condition:** (x₀, y₀, z₀) = (1.0, 1.0, 1.0)

### Learning Configuration
- **Observable:** Full state (x, y, z)
- **Noise:** None - perfect observations
- **Learnable parameter:** σ (sigma) only (implicit through ensemble)
- **Fixed parameters:** ρ=28.0, β=8/3 (not explicitly constrained, but initialized correctly)
- **Initial guess (ensemble mean):** σ₀ = 5.0 (50% error from true value)

### EnKF Configuration
- **Ensemble size:** N=50 members
- **Process noise:** σ_process = 0.01 (small system dynamics uncertainty)
- **Parameter noise:** σ_param = 0.1 (maintains ensemble spread for parameters)
- **Observation noise:** σ_obs = 0.0 (perfect observations, no measurement error)

The ensemble is initialized with:
- **State members:** Small perturbation (±0.01) around true initial state
- **Parameter members:** Spread (±0.5 for sigma, ±0.1 for rho/beta) around the wrong initial guess

## Expected Results

### Success Criteria
1. **Ensemble mean σ converges** to 10.0 over time
2. **Ensemble uncertainty** should decrease as more observations are assimilated
3. **Final predictions** match observations closely
4. **Convergence speed** may differ from NeuralODE (sequential vs batch)

### Key Differences from NeuralODE
- **Convergence pattern:** Sequential updates at each observation time (vs batch over all data)
- **Uncertainty:**  Ensemble provides confidence intervals (vs point estimate)
- **Ensemble collapse:** Parameter noise prevents premature convergence
- **Nonlinear filtering:** Handles model nonlinearity through ensemble sampling

## Running the Experiment

### Prerequisites
```bash
# Ensure data is generated
python Data/Lorentz63/generate_data.py --steps 500 --dt 0.1
```

### Execute
```bash
# Option 1: Direct Python
python Experiments/learn_sigma_perfect_enkf/learn_sigma_perfect_enkf.py

# Option 2: With configuration overrides
python Experiments/learn_sigma_perfect_enkf/learn_sigma_perfect_enkf.py \
  enkf.ensemble_size=100 \
  enkf.param_noise_std=0.2
```

## Outputs

Results saved to `Experiments/learn_sigma_perfect_enkf/`:

### Files
- **`results.pt`** - PyTorch checkpoint:
  - `X_hist` - State history (T, N, 3)
  - `theta_hist` - Parameter history (T, N, 3)
  - `sigma_hist_mean` - Ensemble mean sigma over time
  - `sigma_hist_std` - Ensemble std sigma over time

### Plots

1. **`sigma_convergence.png`**
   - Ensemble mean sigma vs observation time
   - Uncertainty band (±1 std) showing ensemble spread
   - True and initial values marked
   - Shows how EnKF reduces uncertainty sequentially

2. **`trajectory_fit.png`**
   - Ensemble mean state trajectory vs observations
   - Three subplots for x, y, z components
   - Measures goodness-of-fit after learning

3. **`parameter_evolution.png`**
   - All three parameters (σ, ρ, β) for all ensemble members
   - Individual members shown as light blue lines
   - Ensemble mean as thick blue line
   - True and initial values marked
   - Shows ensemble convergence and spread

4. **`trajectory_comparison.png`**
   - Same as NeuralODE version
   - Compares initial, final, and true sigma trajectories
   - Useful for comparing with NeuralODE results

## Interpretation Guide

### Sigma Convergence Plot

**Healthy behavior:**
- Ensemble mean increases from 5.0 toward 10.0
- Ensemble spread (blue band) decreases over time
- Convergence may be faster initially, then plateau
- Final value should be ≈10.0 (may not be perfect due to ensemble sampling)

**Potential issues:**
- **No convergence:** Parameters noise too high, ensemble doesn't learn
- **Premature collapse:** Ensemble std → 0 too quickly, filter gets stuck
- **Slow convergence:** Parameter noise too low, insufficient exploration

### Parameter Evolution Plot

**What to observe:**
- **Rho and Beta:** Should cluster around their true values (not learned, but stable)
- **Sigma:** All ensemble members should gradually move toward 10.0
- **Ensemble spread:** Should decrease as information accumulates
- **Individual members:** More dispersed initially, converging over time

### Comparison with NeuralODE

**Expected differences:**
| Aspect | NeuralODE | EnKF |
|--------|-----------|------|
| **Convergence** | Smooth (batch optimization) | Stepwise (sequential updates) |
| **Final accuracy** | Potentially higher (exhaustive optimization) | Good but less precise |
| **Ensemble** | N/A | Shows uncertainty quantification |
| **Runtime** | Longer (100 epochs) | Faster (sequential only) |

## Configuration Tuning

### Key Parameters

```yaml
enkf:
  ensemble_size: 50          # Increase for better sampling (but slower)
  process_noise_std: 0.01    # System uncertainty (very small if true model)
  param_noise_std: 0.1       # Maintains ensemble diversity (crucial for learning)
  obs_noise_std: 0.0         # Set >0 for noisy observations
```

### Suggested Experiments

1. **Increase ensemble size** → Better parameter estimates but slower
   ```bash
   python ... enkf.ensemble_size=200
   ```

2. **Reduce parameter noise** → Faster convergence but risk of collapse
   ```bash
   python ... enkf.param_noise_std=0.05
   ```

3. **Add observation noise** → Realistic scenario
   ```bash
   python ... enkf.obs_noise_std=0.5
   ```

4. **Increase process noise** → Slower learning, more conservative
   ```bash
   python ... enkf.process_noise_std=0.1
   ```

## Next Steps

After validating this experiment:

1. **Compare with NeuralODE** - Plot both results together
2. **Add observation noise** - Test robustness to measurement error
3. **Learn multiple parameters** - Can EnKF learn rho or beta?
4. **Sparse observations** - How often do you need data?
5. **Wrong dynamics model** - What if trans_fn is incorrect?
6. **Hybrid approaches** - Combine EnKF with Neural Networks

## References

- **EnKF Theory:** Evensen, G., "The Ensemble Kalman Filter" (2003)
- **Parameter estimation:** Aanonsen et al., "Ensemble Kalman Filter Tutorial" (2009)
- **State augmentation:** Naevdal et al., "Estimation of global properties with the Ensemble Kalman Filter" (2002)

## Contact

For questions about this experiment, see the main README.md.
