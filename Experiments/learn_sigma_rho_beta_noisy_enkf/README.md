# Learn Sigma, Rho, and Beta Parameters from Noisy Observations using EnKF

## Overview

This experiment learns **ALL THREE parameters** of the Lorentz63 system from **noisy observations** using the State Augmented Ensemble Kalman Filter. It represents the most challenging parameter learning scenario.

## Key Features

| Feature | Value |
|---------|-------|
| Parameters Learned | 3 (sigma, rho, beta) |
| Observation Quality | Noisy (std = 0.5) |
| Ensemble Size | 75 (larger for 3-parameter learning) |
| Learning Difficulty | **Very High** |
| Real-world Relevance | **Very High** |

## Comparison with Related Experiments

| Aspect | σ perfect | σβ noisy | σρβ noisy |
|--------|-----------|----------|----------|
| Parameters learned | 1 | 2 | **3** |
| Observation noise | None | Moderate | Moderate |
| Learning difficulty | Low | Medium | **Very High** |
| Typical convergence time | ~100 obs | ~200 obs | ~300-500 obs |
| Expected final error | <1% | 1-5% | 2-10% |
| Parameter identifiability | High | Medium | **Low** |
| Ensemble spread | Small | Medium | **Large** |

## Configuration

The experiment is configured in `config.yaml`:

```yaml
parameters:
  truth:
    sigma: 10.0
    rho: 28.0         # Now being learned!
    beta: 2.6667

  initial_guess:
    sigma: 5.0        # 50% error
    rho: 20.0         # 28.6% error  
    beta: 1.5         # 43.8% error

enkf:
  ensemble_size: 75   # Larger ensemble for 3-parameter learning
  param_noise_std: 0.25
  obs_noise_std: 0.5  # Noisy observations
```

## Challenges

Learning three parameters simultaneously introduces significant challenges:

### 1. **Parameter Identifiability**
   - Not all parameters may be identifiable from the observations
   - Some parameters may have weak influence on the observed variables
   - Risk of parameter trading (multiple parameter sets giving similar predictions)

### 2. **Ensemble Collapse**
   - Larger parameter space increases sample error
   - May need ensemble size > 50 for stable convergence
   - Default: 75 ensemble members (vs 50 for 2-parameter learning)

### 3. **Curse of Dimensionality**
   - More parameters = exponential increase in parameter space volume
   - Harder to explore the space adequately
   - May need more observations for convergence

### 4. **Rho's Special Role**
   - Rho is the Rayleigh number - controls transition to chaos
   - Small errors in rho can significantly affect dynamics
   - May be difficult to estimate accurately

## Expected Behavior

### Likely Convergence Pattern

1. **Sigma**: Should converge relatively well (affects attractor scale)
2. **Beta**: Should converge moderately (affects return rate)
3. **Rho**: May converge slowly or remain uncertain (controls bifurcation)

### Success Indicators

✓ **Good convergence**:
- All three parameters converge within 5-10% of truth
- Ensemble spreads decrease significantly
- Trajectories match observations well

⚠ **Moderate success**:
- Sigma and beta converge, rho diverges or remains uncertain
- Only 2 parameters are truly identifiable
- Suggests rho information is not observable

✗ **Poor convergence**:
- Multiple parameters diverge or stick at initial values
- Ensemble spreads remain large throughout
- Indicates insufficient parameter identifiability

## Experimental Hypothesis

**Can the EnKF learn all three Lorentz63 parameters from noisy observations?**

- **Hypothesis A**: All three parameters are identifiable → all converge
- **Hypothesis B**: Only 2 parameters are identifiable → sigma and beta converge, rho diverges
- **Hypothesis C**: Only 1 parameter is identifiable → only sigma converges

## Running the Experiment

```bash
cd Experiments/learn_sigma_rho_beta_noisy_enkf
python learn_sigma_rho_beta_noisy_enkf.py
```

Expected runtime: **30-60 minutes** (CPU)

## Output Files

After running, you'll get:
- **sigma_rho_beta_convergence.png**: Three-panel parameter convergence plot
- **trajectory_fit.png**: Ensemble mean vs observations
- **parameter_evolution.png**: All ensemble members for all three parameters
- **trajectory_comparison.png**: Trajectory comparison with different parameters
- **results.pt**: Complete results for analysis

## Analysis Guide

### Interpreting Parameter Convergence

```python
import torch
import matplotlib.pyplot as plt

results = torch.load('results.pt')
sigma_mean = results['sigma_hist_mean']
sigma_std = results['sigma_hist_std']
true_sigma = results['true_params'][0, 0]

# Check convergence
final_error = abs(sigma_mean[-1] - true_sigma) / true_sigma * 100
print(f"Final sigma error: {final_error:.2f}%")

# Check ensemble spread reduction
initial_spread = sigma_std[10]  # Skip first few steps
final_spread = sigma_std[-1]
print(f"Ensemble spread reduction: {final_spread / initial_spread:.2%}")
```

### Computing Parameter Errors

```python
errors = {}
for i, name in enumerate(['sigma', 'rho', 'beta']):
    hist_mean = results[f'{name}_hist_mean']
    true_val = results['true_params'][0, i]
    final_error = abs(hist_mean[-1] - true_val) / true_val * 100
    errors[name] = final_error
    print(f"{name}: {final_error:.2f}%")
```

## Advanced Variations

### Variation 1: Larger Ensemble
```yaml
enkf:
  ensemble_size: 100  # vs default 75
```
→ More accurate parameter estimates, slower computation

### Variation 2: More Observations
```yaml
data:
  n_obs: 1000  # vs default 500
```
→ Better parameter identifiability, longer runtime

### Variation 3: Lower Observation Noise
```yaml
data:
  noise_std: 0.1  # vs default 0.5
```
→ Faster convergence, less realistic

### Variation 4: Higher Observation Noise
```yaml
data:
  noise_std: 1.0  # vs default 0.5
```
→ More challenging, tests robustness

## Key Metrics to Track

1. **Convergence Speed**: How many observations until each parameter settles?
2. **Final Accuracy**: What is the % error for each parameter?
3. **Ensemble Spread**: How much do ensemble members differ?
4. **Parameter Coupling**: Do parameters converge together or separately?

## Physical Interpretation

- **Sigma** (≈10): Controls momentum - related to thermal conductivity ratio
- **Rho** (≈28): Controls forcing - the Rayleigh number
- **Beta** (≈2.67): Controls geometry - aspect ratio parameter

Understanding which parameters are observable is important for real applications!

## Related Work

- `learn_sigma_perfect_enkf`: Baseline with 1 parameter, perfect observations
- `learn_sigma_beta_noisy_enkf`: 2-parameter learning from noisy observations
- `obs_dt_sweep`: Parameter learning across observation frequencies

## Expected Outcomes

### Scenario 1: All Parameters Converge
```
σ error: 1.2% | ρ error: 2.1% | β error: 1.8%
→ All three parameters are identifiable
→ System provides sufficient information
→ EnKF is capable of 3-parameter learning
```

### Scenario 2: Two Parameters Converge
```
σ error: 1.0% | ρ error: 15% | β error: 2.0%
→ Rho is not well-identifiable
→ Possibly needs additional observations or different observation types
→ Consider learning only 2 parameters in practice
```

### Scenario 3: All Parameters Diverge
```
σ error: 8% | ρ error: 12% | β error: 9%
→ 3-parameter learning fails with current setup
→ Increase ensemble size or number of observations
→ Reduce observation noise for testing
```

## Troubleshooting

**Issue**: Rho doesn't converge
- Try: Larger ensemble (100-150)
- Try: More observations (1000+)
- Try: Lower observation noise

**Issue**: All parameters diverge
- Try: Reduce param_noise_std
- Try: Verify data generation is correct
- Try: Check that initial guesses aren't too far away

**Issue**: Very slow computation
- This is expected for 75 ensemble members × 500 observations
- Try: Smaller ensemble (50) or fewer observations (250) for testing

## Mathematical Note

The state augmentation transforms the problem from:

*Nonlinear dynamics estimation:* Find X(t) given Y(t)

to:

*Joint state-parameter estimation:* Find [X(t); θ] given Y(t)

This is significantly harder because the parameter space grows exponentially with dimensionality, and observability must be established in the augmented space.
