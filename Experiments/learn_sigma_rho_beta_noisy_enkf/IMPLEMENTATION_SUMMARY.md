# Implementation Summary: learn_sigma_rho_beta_noisy_enkf

## Overview

This experiment extends `learn_sigma_beta_noisy_enkf` to learn **all three parameters** instead of just two:
- Sigma (σ): Prandtl number
- Rho (ρ): Rayleigh number  
- Beta (β): Geometric parameter

## Key Differences from Baseline (learn_sigma_beta_noisy_enkf)

### 1. Parameter Vector

**learn_sigma_beta_noisy_enkf**:
```python
theta = [σ, ρ, β]
# Learn: [σ, ?, β]  where ? = fixed rho
param_noise_spread = [0.5, 0.0, 0.3]  # No noise on rho
```

**learn_sigma_rho_beta_noisy_enkf**:
```python
theta = [σ, ρ, β]
# Learn: [σ, ρ, β]  all three parameters
param_noise_spread = [0.5, 1.0, 0.3]  # Noise on all three
```

### 2. Ensemble Size

**Reason**: Learning 3 parameters requires larger ensemble to avoid collapse
```yaml
# Previous
ensemble_size: 50

# Current
ensemble_size: 75  # 50% larger
```

The rule of thumb: ensemble size ≥ 10 × number of parameters
- 1 parameter: 50 ensemble members
- 2 parameters: 50-60 ensemble members
- **3 parameters: 75-100 ensemble members** ✓

### 3. Initial Parameter Guesses

```yaml
# Very wrong initial values to test identifiability
initial_guess:
  sigma: 5.0        # True: 10.0    (50% error)
  rho: 20.0         # True: 28.0    (28.6% error)
  beta: 1.5         # True: 2.6667  (43.8% error)
```

### 4. Parameter Noise Spreads

```python
# Initial ensemble initialization
theta0_ensemble = initial_mean.unsqueeze(0).repeat(ensemble_size, 1)
theta0_ensemble = theta0_ensemble + torch.randn_like(theta0_ensemble) * (
    torch.tensor([0.5, 1.0, 0.3])  # [sigma spread, rho spread, beta spread]
)
```

**Rationale**:
- Sigma spread: 0.5 (centered at 5.0 → range ~4.5-5.5)
- Rho spread: 1.0 (centered at 20.0 → range ~19-21)
- Beta spread: 0.3 (centered at 1.5 → range ~1.2-1.8)

These spreads represent the prior uncertainty about each parameter.

### 5. Metrics and Output

**Previous** (2 parameters):
```python
final_sigma_mean = theta_hist[-1, :, 0].mean()
final_beta_mean = theta_hist[-1, :, 2].mean()
# Print sigma and beta errors
```

**Current** (3 parameters):
```python
final_sigma_mean = theta_hist[-1, :, 0].mean()
final_rho_mean = theta_hist[-1, :, 1].mean()
final_beta_mean = theta_hist[-1, :, 2].mean()
# Print sigma, rho, and beta errors
```

### 6. Plotting

Added **three subplots** in convergence plot instead of two:

```python
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Subplot 0: Sigma convergence (blue)
ax = axes[0]
ax.plot(sigma_mean, 'b-', ...)

# Subplot 1: Rho convergence (purple)
ax = axes[1]
ax.plot(rho_mean, 'purple', ...)

# Subplot 2: Beta convergence (green)
ax = axes[2]
ax.plot(beta_mean, 'g-', ...)
```

Similarly for parameter evolution plot - now shows all three parameters instead of two.

## Configuration Changes Explained

### ensemble_size: 50 → 75

```
Ensemble size vs Parameter count relationship:
────────────────────────────────────────────────
Params  Min Size  Recommended  Safe
1       10        30           50
2       20        40           60
3       30        75           100
4       40        100          150
```

Larger ensemble means:
- ✓ More particles to explore parameter space
- ✓ Better ensemble statistics
- ✓ Less sampling error
- ✗ Longer computation time (3× slower)

### param_noise_std: 0.2 → 0.25

Slightly increased to give wider initial spread in parameter space when learning more parameters.

### rho: 28.0 (fixed) → 20.0 (learned from wrong initial)

Rho is now part of the learning problem, so it needs an initial guess that's sufficiently wrong to test whether EnKF can correct it.

## Expected Difficulties

### 1. Curse of Dimensionality

With 3 parameters instead of 2:
- Parameter space volume grows exponentially
- Harder to sample parameter space adequately
- May need more observations
- Convergence could be slower

### 2. Parameter Identifiability

Not all parameters may be equally observable:
- Sigma: Controls the "size" of attractor - observable from state magnitude
- Rho: Controls bifurcation to chaos - observable from behavior type
- Beta: Controls return rate - observable from oscillation frequency

Rho might be harder to identify than sigma or beta.

### 3. Ensemble Collapse Risk

With weak observability of some parameters:
- Ensemble might collapse in parameter space
- Multiple parameter sets might give similar predictions
- Could converge to wrong local minimum

### 4. Computational Cost

```
Computation complexity:
- State dimension: 3
- Ensemble size: 75 (vs 50 before)
- Time steps: 500
- Total: 75 × 500 × 3 = 112,500 state updates

Expected runtime:
- CPU: 30-60 minutes
- GPU: 5-10 minutes (if using GPU)
```

## Code Structure

### Main Function Flow

1. **load_data()**: Load observations, add noise
2. **create_dynamics_wrapper()**: Create Lorentz63 system
3. **run_experiment()**: 
   - Initialize 75-member ensemble
   - Run EnKF with 3-parameter augmentation
   - Compute convergence metrics for all 3 parameters
4. **plot_results()**: 
   - Create 3-panel convergence plot
   - Create trajectory fit plot
   - Create 3-panel parameter evolution plot
5. **plot_trajectory_comparison()**: Compare trajectories

### Key Code Sections

**Parameter initialization (all three now random)**:
```python
theta0_ensemble = theta0_ensemble + torch.randn_like(theta0_ensemble) * (
    torch.tensor([0.5, 1.0, 0.3])  # All three have non-zero spread
)
```

**Metrics computation (all three parameters)**:
```python
for param_idx in range(3):
    final_mean = theta_hist[-1, :, param_idx].mean().item()
    true_val = true_params[0, param_idx].item()
    error = abs(final_mean - true_val)
    rel_error = error / true_val * 100
    # Print results
```

**Plotting (three subplots)**:
```python
for ax, name, idx in zip(axes, ['sigma', 'rho', 'beta'], [0, 1, 2]):
    # Plot convergence for each parameter
```

## Testing the Implementation

### Unit Test: Syntax
```bash
python -m py_compile learn_sigma_rho_beta_noisy_enkf.py
# Should complete without errors
```

### Integration Test: Quick Run
```yaml
# config.yaml - temporary settings for testing
data:
  n_obs: 50           # Very small for quick test
enkf:
  ensemble_size: 20   # Very small ensemble
```
```bash
python learn_sigma_rho_beta_noisy_enkf.py
# Should complete in < 1 minute
```

### Full Test: Normal Run
```bash
python learn_sigma_rho_beta_noisy_enkf.py
# Should complete in 30-60 minutes with sensible results
```

## Validation Checklist

After running, verify:

- [ ] All 4 output PNG files exist
- [ ] results.pt file created (50-100 MB)
- [ ] Console output shows 3 parameters being learned
- [ ] sigma_rho_beta_convergence.png has 3 subplots
- [ ] All three subplots show trend toward true values
- [ ] parameter_evolution.png shows all 3 parameters
- [ ] trajectory_comparison.png shows improvement

## Parameter Sensitivity

### If rho doesn't converge:

**Reason**: Rho may not be well-observable from full-state observations

**Try**:
- Larger ensemble: `ensemble_size: 100` or `150`
- More observations: `n_obs: 1000` or `2000`
- Reduce noise: `obs_noise_std: 0.1` (for testing)
- Better initial guess: `rho: 25.0` instead of `20.0`

### If all parameters diverge:

**Reason**: Ensemble size too small for parameter space

**Try**:
- Larger ensemble: `ensemble_size: 100`
- Smaller parameter spreads: Change multipliers to `[0.3, 0.5, 0.2]`
- Keep initial guesses closer to truth

### If convergence is very slow:

**Reason**: Normal for 3-parameter learning

**Accept**: This is expected behavior
**Or try**: Reduce `n_obs` for testing, then scale up

## Comparison with Related Experiments

```python
# Quick comparison code
experiments = {
    'σ_perfect': {
        'params': 1,
        'noise': 0.0,
        'ensemble': 50,
        'time_to_converge': '~100 obs'
    },
    'σβ_noisy': {
        'params': 2,
        'noise': 0.5,
        'ensemble': 50,
        'time_to_converge': '~250 obs'
    },
    'σρβ_noisy': {  # This experiment
        'params': 3,
        'noise': 0.5,
        'ensemble': 75,
        'time_to_converge': '~400-500 obs'
    }
}
```

## Performance Profiling

If you want to optimize:

```python
import time

# In run_experiment(), add timing:
start = time.time()
X_hist, theta_hist = model.run(...)
elapsed = time.time() - start

print(f"EnKF runtime: {elapsed:.1f} seconds")
print(f"Per observation: {elapsed / len(observations):.3f} seconds")
```

Expected:
- σ_perfect: ~5 seconds
- σβ_noisy: ~10 seconds
- σρβ_noisy: ~30-60 seconds

## Extending Further

If you want to go beyond 3 parameters:

### 4-Parameter Learning
```yaml
# Learn all three plus one more (e.g., initial condition uncertainty)
ensemble_size: 100
param_noise_std: 0.3
```

### Time-Varying Parameters
```python
# Make parameters time-dependent
theta_t = theta + epsilon * sin(2 * pi * t / T)
```

### State-Dependent Parameters
```python
# Adaptation based on mismatch
if innovation_norm > threshold:
    param_noise_cov *= 1.1  # Increase uncertainty
```

## Known Limitations

1. **Rho Identifiability**: Rho may not be well-identifiable
2. **Noise Sensitivity**: High observation noise may prevent convergence
3. **Ensemble Size**: 75 is minimum; 100+ may be needed for stability
4. **Computational Cost**: 30-60 minute runtime on CPU
5. **Parameter Trading**: Multiple parameter sets might give similar predictions

## Files Modified from Baseline

| File | Changes |
|------|---------|
| config.yaml | ensemble_size 50→75, rho 28→20, param spreads |
| learn_sigma_rho_beta_noisy_enkf.py | All 3 parameters, more metrics, 3-panel plots |
| README.md | New content for 3-parameter learning |
| QUICKSTART.md | New file, quick reference |

## Success Metrics

| Metric | Excellent | Good | OK | Poor |
|--------|-----------|------|----|----|
| σ error | <1% | <2% | <5% | >5% |
| ρ error | <2% | <5% | <10% | >10% |
| β error | <1% | <2% | <5% | >5% |
| Ensemble spread reduction | >90% | >75% | >50% | <50% |
| Trajectory fit | Excellent | Good | Fair | Poor |
