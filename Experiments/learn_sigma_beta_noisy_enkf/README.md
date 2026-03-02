# Learn Sigma and Beta Parameters from Noisy Observations using EnKF

## Overview

This experiment extends the `learn_sigma_perfect_enkf` experiment in two key ways:

1. **Multiple Parameters**: Learns **both sigma and beta** parameters of the Lorentz63 system (instead of just sigma)
2. **Realistic Observations**: Uses **noisy observations** instead of perfect/noise-free observations

## Key Differences from `learn_sigma_perfect_enkf`

| Aspect | learn_sigma_perfect_enkf | learn_sigma_beta_noisy_enkf |
|--------|--------------------------|----------------------------|
| Parameters learned | Sigma only | Sigma + Beta |
| Observation quality | Perfect (noiseless) | Noisy (std = 0.5) |
| Parameter learning difficulty | Easier (single param) | Harder (2 params, noisy data) |
| Real-world relevance | Low | High |

## Configuration

The experiment is configured in `config.yaml`:

```yaml
parameters:
  truth:
    sigma: 10.0
    rho: 28.0           # Fixed (not learned)
    beta: 2.6667

  initial_guess:
    sigma: 5.0          # Wrong initial guess
    rho: 28.0           # Correct
    beta: 1.5           # Wrong initial guess

enkf:
  ensemble_size: 50
  process_noise_std: 0.01
  param_noise_std: 0.2
  obs_noise_std: 0.5    # Observation noise level
```

## Experiment Goals

1. **Parameter Identifiability**: Can we learn two parameters from noisy observations?
2. **Convergence Speed**: How does noise affect convergence compared to perfect observations?
3. **Ensemble Consistency**: Do ensemble members converge to true parameters?
4. **Parameter Coupling**: Are sigma and beta learned independently or are they coupled?

## Running the Experiment

```bash
cd Experiments/learn_sigma_beta_noisy_enkf
python learn_sigma_beta_noisy_enkf.py
```

## Expected Results

The experiment will output:
- **sigma_beta_convergence.png**: Parameter convergence plots for both sigma and beta
- **trajectory_fit.png**: Ensemble mean predictions vs noisy observations
- **parameter_evolution.png**: Evolution of all ensemble members for sigma and beta
- **trajectory_comparison.png**: Trajectories using initial, final, and true parameters
- **results.pt**: PyTorch file with complete results for analysis

## Key Metrics

The experiment prints:
- Final sigma estimate and relative error
- Final beta estimate and relative error
- Ensemble spread (standard deviation) for each parameter
- Per-component errors

## Interpretation

### Success Indicators
- Both parameters converge toward true values
- Ensemble spread decreases over time
- Final predictions improve toward observations
- Relative errors < 5% for both sigma and beta

### Challenges
- Noisy observations reduce observability
- Two-parameter learning is harder than single-parameter
- Ensemble may diverge if initialization or noise parameters are poor

## Parameter Sensitivity

- **ensemble_size**: 50 is reasonable; try 30-100 for different regimes
- **param_noise_std**: Controls initial ensemble spread; larger values = wider initial ensemble
- **obs_noise_std**: Should match actual observation noise level; we use 0.5 as a moderate value
- **process_noise_std**: Controls model uncertainty; affects how much parameters can adapt

## Related Experiments

- `learn_sigma_perfect_enkf`: Single-parameter learning from perfect observations
- `learn_sigma_perfect`: NeuralODE equivalent (batch learning, perfect observations)
- `obs_dt_sweep`: Parameter learning across different observation frequencies

## Implementation Notes

1. **Fixed Rho**: The rho parameter is held constant at 28.0 throughout. Its ensemble noise is set to 0.
2. **Observation Noise**: Gaussian noise with std=0.5 is added to observations during data loading
3. **State Augmentation**: Both state (x, y, z) and parameters (sigma, rho, beta) are estimated jointly
4. **Sequential Estimation**: The EnKF updates the state and parameters at each observation time

## Future Extensions

- Learn all three parameters (sigma, rho, beta)
- Try different observation noise levels to assess sensitivity
- Compare with variational methods (4DVar)
- Analyze parameter correlations in the ensemble
