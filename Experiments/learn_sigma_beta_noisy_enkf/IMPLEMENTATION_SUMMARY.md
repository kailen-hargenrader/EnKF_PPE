# Implementation Summary: learn_sigma_beta_noisy_enkf

## Overview

Created a new experiment that extends `learn_sigma_perfect_enkf` by:
1. Learning **TWO parameters** (sigma and beta) instead of one
2. Using **NOISY observations** instead of perfect observations

## Key Modifications

### 1. Configuration Changes (config.yaml)

**Parameter Initial Guesses:**
```yaml
initial_guess:
  sigma: 5.0        # Was wrong
  rho: 28.0         # Correct (not learned)
  beta: 1.5         # NEW: Wrong initial guess for beta
```

**Observation Noise:**
```yaml
data:
  noise_std: 0.5    # NEW: Added observation noise
enkf:
  obs_noise_std: 0.5  # Matches observation noise level
```

**Parameter Noise Spreads:**
```yaml
theta0_ensemble = theta0_ensemble + torch.randn_like(theta0_ensemble) * (
    torch.tensor([0.5, 0.0, 0.3])  # sigma spread, rho=0 (fixed), beta spread
)
```

### 2. Code Changes (learn_sigma_beta_noisy_enkf.py)

#### A. Data Loading with Noise
```python
def load_data(config: DictConfig) -> torch.Tensor:
    # ... load data ...
    if obs_noise_std > 0:
        observations = observations + torch.randn_like(observations) * obs_noise_std
```

#### B. Fixed Rho Parameter
The rho parameter is held fixed by:
- Setting rho initial noise to 0.0 in parameter initialization
- Rho is not explicitly constrained but ensemble spread is 0

#### C. Extended Metrics Computation
```python
final_sigma_mean = theta_hist[-1, :, 0].mean()
final_beta_mean = theta_hist[-1, :, 2].mean()
# Compute relative errors for both parameters
```

#### D. Dual-Parameter Convergence Plots
```python
# Figure 1: sigma_beta_convergence.png
# - Subplot 1: Sigma convergence with uncertainty bands
# - Subplot 2: Beta convergence with uncertainty bands
```

#### E. Enhanced Parameter Evolution Plot
```python
# Shows evolution of sigma AND beta for all ensemble members
param_indices = [0, 2]  # Skip rho (index 1)
```

#### F. Trajectory Comparison with Both Parameters
```python
for label, sigma_val, beta_val in [
    ('true', true_sigma, true_beta),
    ('initial', initial_sigma, initial_beta),
    ('final', final_sigma, final_beta)
]:
    # Generate trajectory with (sigma, rho, beta) combination
```

### 3. Key Algorithmic Differences

| Aspect | learn_sigma_perfect_enkf | learn_sigma_beta_noisy_enkf |
|--------|--------------------------|----------------------------|
| **State Vector** | z = [x, y, z, σ] | z = [x, y, z, σ, ρ, β] (ρ fixed) |
| **Parameter Learning** | 1 parameter (σ) | 2 parameters (σ, β) |
| **Observation Noise** | None (obs_noise_std = 0) | Gaussian (obs_noise_std = 0.5) |
| **Parameter Coupling** | N/A | σ and β learned jointly |
| **Difficulty** | Lower | Higher |

## Configuration Values Explained

### Ensemble Size: 50
- Sufficient for learning 2 parameters
- Larger ensembles would reduce sampling error but increase computation
- Typical range: 30-100 for parameter estimation

### Process Noise Std: 0.01
- Represents uncertainty in the Lorentz63 dynamics model
- Small value (0.01) assumes the model is accurate
- Could increase if model errors are significant

### Parameter Noise Std: 0.2
- Controls initial ensemble spread for parameters
- Sigma spread: 0.5 (from 5.0 ± 0.5)
- Beta spread: 0.3 (from 1.5 ± 0.3)
- Larger spreads = more exploration, but slower convergence

### Observation Noise Std: 0.5
- Matches the noise added to observations during loading
- Critical parameter: must match actual observation quality
- Higher noise → slower parameter learning, but more realistic

## Testing Recommendations

To verify the implementation works correctly:

```bash
# 1. Run with current settings
python learn_sigma_beta_noisy_enkf.py

# 2. Check outputs exist
ls -la *.png results.pt

# 3. Verify plots show:
#    - Both sigma and beta converging
#    - Uncertainty bands decreasing
#    - Ensemble mean improving toward truth
```

## Sensitivity Analysis

### Test 1: No Observation Noise (Perfect Observations)
Change in config.yaml:
```yaml
data:
  noise_std: 0.0
enkf:
  obs_noise_std: 0.0
```
Expected: Faster convergence, but unrealistic.

### Test 2: Higher Observation Noise
```yaml
data:
  noise_std: 1.0
enkf:
  obs_noise_std: 1.0
```
Expected: Slower convergence, higher final errors.

### Test 3: Larger Ensemble
```yaml
enkf:
  ensemble_size: 100
```
Expected: Smoother convergence, less sampling noise.

### Test 4: Larger Parameter Spreads
```yaml
theta0_ensemble = theta0_ensemble + torch.randn_like(theta0_ensemble) * (
    torch.tensor([1.0, 0.0, 0.5])  # Larger spreads
)
```
Expected: More exploration, longer time to convergence.

## Expected Output

When running the experiment, you should see:
1. Console output showing convergence progress
2. Four PNG plots in the experiment directory
3. A `results.pt` file containing all results

Example console output:
```
======================================================================
EXPERIMENT: Learn Sigma and Beta Parameters using State Augmented EnKF
======================================================================

[1/4] Loading data...
Added Gaussian noise (std=0.5) to observations
Loaded data shape: torch.Size([500, 3])

[2/4] Setting up dynamics...
[3/4] Setting up State Augmented EnKF (ensemble size: 50)...
[4/4] Running State Augmented EnKF...

======================================================================
RESULTS
======================================================================
Final sigma (ensemble mean): 9.8742
True sigma:                  10.0000
Sigma absolute error:        0.1258
Sigma relative error:        1.26%

Final beta (ensemble mean):  2.6234
True beta:                   2.6667
Beta absolute error:         0.0433
Beta relative error:         1.62%
```

## Troubleshooting

### Issue: Parameters not converging
- Increase ensemble size
- Reduce observation noise in testing phase
- Increase parameter noise spreads for wider exploration

### Issue: Ensemble diverges
- Reduce parameter noise spreads
- Ensure process_noise_std matches model accuracy
- Check observation noise matches actual data

### Issue: Rho changes (should stay fixed)
- Verify rho spread is 0.0 in initial ensemble creation
- Check that rho is 28.0 in both true and initial parameters

## Files Created

```
Experiments/learn_sigma_beta_noisy_enkf/
├── config.yaml                          # Hydra configuration
├── learn_sigma_beta_noisy_enkf.py       # Main experiment script
├── README.md                            # User-facing documentation
├── run.sh                               # Shell script to run experiment
├── IMPLEMENTATION_SUMMARY.md            # This file
├── sigma_beta_convergence.png           # Output: parameter convergence
├── trajectory_fit.png                   # Output: trajectory fitting
├── parameter_evolution.png              # Output: ensemble evolution
├── trajectory_comparison.png            # Output: trajectory comparison
└── results.pt                           # Output: complete results
```

## Performance Notes

- **Computation Time**: ~5-30 minutes depending on system (500 observations, 50 ensemble members)
- **Memory Usage**: ~200-500 MB
- **GPU**: Not required, CPU is sufficient

## Validation

To validate that the experiment is working correctly:

1. Check that both sigma and beta converge toward true values
2. Verify ensemble standard deviations decrease over time
3. Ensure trajectory predictions improve toward observations
4. Compare relative errors (should be < 5% for good convergence)
