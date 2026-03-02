# Experiment: Learn Sigma Parameter with Perfect Observations

## Overview

This is the first experiment in the EnKF_PPE project. It tests the **NeuralODE** model's ability to learn the `sigma` parameter of the Lorentz63 system from **perfectly observed** (noise-free) data.

## Motivation

The NeuralODE uses **automatic differentiation** through a continuous ODE solver (adjoint method) to estimate parameters. This experiment answers the question:

**Can the NeuralODE recover a parameter value when:**
- ✅ The dynamics model is **exactly correct** (true Lorentz63)
- ✅ The observations are **noise-free** (perfect data)
- ❌ The initial parameter guess is **very wrong** (sigma=5.0 vs true sigma=10.0)

A positive result would indicate that automatic differentiation and the adjoint method are working correctly for this system.

## Experiment Setup

### True System
- **System:** Lorentz63 with σ=10.0, ρ=28.0, β=2.6667
- **Data:** 500 time steps at dt=0.1
- **Initial condition:** (x₀, y₀, z₀) = (1.0, 1.0, 1.0)

### Learning Configuration
- **Observable:** Full state (x, y, z) - **full observation**
- **Noise:** None - **perfect observations**
- **Learnable parameter:** σ (sigma) only
- **Fixed parameters:** ρ=28.0, β=8/3
- **Initial guess:** σ₀ = 5.0 (50% error from true value)

### NeuralODE Configuration
- **Epochs:** 100
- **Learning rate:** 0.01 (Adam optimizer)
- **Gradient clipping:** max norm = 1.0
- **Model time step:** 0.01 (10 steps per observation)
- **Observation interval:** 0.1

## Expected Results

### Success Criteria
1. **Sigma converges** to the true value of 10.0
2. **Final predictions** match observations closely (near-zero MSE)
3. **Convergence** occurs within 100 epochs

### Why This Matters
- **Perfect conditions test** the core algorithm reliability
- **No noise** eliminates observation uncertainty
- **True dynamics** isolate parameter learning capability
- **Single learnable parameter** simplifies interpretation

## Running the Experiment

### Prerequisites
```bash
# Ensure data is generated
python Data/Lorentz63/generate_data.py --steps 500 --dt 0.1
```

### Execute
```bash
# Option 1: Direct Python
python Experiments/learn_sigma_perfect/learn_sigma_perfect.py

# Option 2: Bash script
bash Experiments/learn_sigma_perfect/run.sh
```

## Outputs

The experiment saves results to `Experiments/learn_sigma_perfect/results/`:

### Files
- **`results.pt`** - PyTorch checkpoint containing:
  - `X_hist` - State trajectory history (T, Epochs, 3)
  - `theta_hist` - Parameter history (Epochs, 3)
  - `true_params` - True parameters (3,)
  - `initial_params` - Initial guesses (3,)

### Plots

1. **`sigma_convergence.png`**
   - Sigma value vs optimization epoch
   - Shows convergence trajectory from initial guess to true value

2. **`trajectory_fit.png`**
   - Three subplots (x, y, z components)
   - Red lines: Observations
   - Blue dashed lines: Final model predictions
   - Measures goodness-of-fit

3. **`parameter_evolution.png`**
   - Three subplots (σ, ρ, β)
   - Shows evolution of all three parameters
   - Green dotted line: initial guess
   - Red dashed line: true value
   - Blue line: estimated value during training

4. **`training_loss.png`**
   - MSE loss vs epoch (log scale)
   - Should decrease monotonically

## Interpretation Guide

### What to Look For

**Sigma Convergence Plot:**
- Initial σ should be 5.0
- Should smoothly increase toward 10.0
- Final value should be ≈ 10.0 (< 0.1 error)
- Convergence pattern indicates optimization behavior

**Trajectory Fit Plot:**
- Red and blue lines should **overlap closely**
- Any visible gaps = model mismatch
- Perfect overlap = parameters recovered correctly

**Training Loss:**
- Should **monotonically decrease** (with Adam optimizer)
- Final loss should be very small (< 1e-4 or better)
- Plateauing indicates convergence

**Parameter Evolution:**
- Rho and beta should **remain constant** (not learned)
- Sigma should **move monotonically** from 5.0 to 10.0

### Potential Issues

| Issue | Likely Cause |
|-------|--------------|
| Sigma doesn't move | Optimization not running, check learning rate |
| Sigma oscillates wildly | Learning rate too high, gradient explosion |
| Sigma stops before reaching 10.0 | Local minimum, may need more epochs |
| Predictions don't fit observations | Dynamics function incorrect, check trans_fn |
| Loss increases at end | Numerical instability, check gradient clipping |

## Next Steps

After validating this experiment, you can:

1. **Add observation noise** - see how much noise the method can tolerate
2. **Learn multiple parameters** - try learning ρ or β as well
3. **Wrong dynamics model** - see what happens with incorrect trans_fn
4. **Sparse observations** - reduce observation density
5. **Ensemble methods** - compare with State Augmented EnKF

## References

- **NeuralODE Paper:** Chen et al., "Neural Ordinary Differential Equations" (NIPS 2018)
- **Adjoint Method:** Pontryagin's maximum principle for sensitivity analysis
- **torchdiffeq:** https://github.com/rtqichen/torchdiffeq

## Contact

For questions about this experiment, see the main README.md or contact the project maintainers.
