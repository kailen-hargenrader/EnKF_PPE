# Quick Start Guide

## Overview

This is a new experiment that learns **both sigma and beta parameters** from **noisy observations** using the State Augmented Ensemble Kalman Filter (EnKF).

## Before Running

Make sure you have:
1. Generated the required Lorentz63 data
2. Installed the `enkf_ppe` package
3. Set up your Python environment with required dependencies

### Step 1: Generate Data (if needed)

```bash
cd Data/Lorentz63
python generate_data.py
cd ../..
```

This creates the data files needed for the experiment.

### Step 2: Install Package (if needed)

```bash
# From the root directory
pip install -e .
# or use uv (if available)
uv pip install -e .
```

## Running the Experiment

### Option A: From Command Line (Recommended)

```bash
cd Experiments/learn_sigma_beta_noisy_enkf
python learn_sigma_beta_noisy_enkf.py
```

### Option B: Using Shell Script

```bash
cd Experiments/learn_sigma_beta_noisy_enkf
bash run.sh
```

### Option C: From Python

```python
import os
import sys

# Add project root to path
sys.path.insert(0, '.')

# Run experiment
os.chdir('Experiments/learn_sigma_beta_noisy_enkf')
exec(open('learn_sigma_beta_noisy_enkf.py').read())
```

## Expected Output

When you run the experiment, you should see:

1. **Console output** showing progress:
   ```
   ======================================================================
   EXPERIMENT: Learn Sigma and Beta Parameters using State Augmented EnKF
   ======================================================================
   
   [1/4] Loading data...
   Added Gaussian noise (std=0.5) to observations
   
   [2/4] Setting up dynamics...
   [3/4] Setting up State Augmented EnKF (ensemble size: 50)...
   [4/4] Running State Augmented EnKF...
   
   ======================================================================
   RESULTS
   ======================================================================
   Final sigma (ensemble mean): X.XXXX
   Final beta (ensemble mean):  X.XXXX
   ...
   ```

2. **Output files** in the same directory:
   - `sigma_beta_convergence.png` - Parameter convergence plots
   - `trajectory_fit.png` - Ensemble predictions vs observations
   - `parameter_evolution.png` - Evolution of all ensemble members
   - `trajectory_comparison.png` - Trajectory comparison
   - `results.pt` - Complete results as PyTorch file

## What to Look For

### Success Indicators

✓ **Sigma converges** toward 10.0 (true value)
✓ **Beta converges** toward 2.6667 (true value)
✓ **Ensemble spread decreases** over time
✓ **Error messages** about convergence are ~1-5%

### Warnings

⚠ If parameters don't converge:
- Check that data file exists at `Data/Lorentz63/sigma10.0000_rho28.0000_beta2.6667_dt0.1000.pt`
- Ensure `enkf_ppe` package is installed
- Try running the simpler `learn_sigma_perfect_enkf` first to debug

## Configuration

To modify the experiment, edit `config.yaml`:

```yaml
data:
  n_obs: 500           # Increase for more observations
  noise_std: 0.5       # Change observation noise level

enkf:
  ensemble_size: 50    # Increase for better convergence
  obs_noise_std: 0.5   # Must match data noise_std
  param_noise_std: 0.2 # Increase for wider initial spread
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'enkf_ppe'"

**Solution**: Install the package
```bash
cd ..  # Go to project root
pip install -e .
```

### Issue: "FileNotFoundError: Data file not found"

**Solution**: Generate the data
```bash
cd Data/Lorentz63
python generate_data.py
cd ../..
```

### Issue: Script runs but produces no plots

**Solution**: Check that matplotlib is installed
```bash
pip install matplotlib
```

### Issue: Very slow performance

**Solution**: This is normal for CPU. Expected runtime:
- Small ensemble (30 members): 5-10 minutes
- Medium ensemble (50 members): 15-30 minutes
- Large ensemble (100 members): 30-60 minutes

To speed up testing:
```yaml
data:
  n_obs: 100           # Use fewer observations initially
enkf:
  ensemble_size: 30    # Use smaller ensemble
```

## Next Steps

After running successfully:

1. **Examine the plots**:
   - Do sigma and beta converge together?
   - Does observation noise hurt convergence?
   - How do ensemble spreads compare?

2. **Compare with baseline**:
   - Run `learn_sigma_perfect_enkf` and compare results
   - See how observation noise affects convergence

3. **Experiment with configurations**:
   - Try different ensemble sizes
   - Try different noise levels
   - See how parameters are affected

4. **Advanced analysis**:
   - Load `results.pt` and analyze the ensemble trajectories
   - Compute parameter correlations
   - Study convergence rates

## File Structure

```
learn_sigma_beta_noisy_enkf/
├── config.yaml                    # Configuration file
├── learn_sigma_beta_noisy_enkf.py # Main experiment
├── QUICKSTART.md                  # This file
├── README.md                      # Detailed documentation
├── COMPARISON.md                  # Comparison with other experiments
├── IMPLEMENTATION_SUMMARY.md      # Technical implementation details
├── run.sh                         # Shell script to run
│
└── [Output files after running]
    ├── sigma_beta_convergence.png
    ├── trajectory_fit.png
    ├── parameter_evolution.png
    ├── trajectory_comparison.png
    └── results.pt
```

## Key Differences from learn_sigma_perfect_enkf

| Feature | Perfect Sigma | Sigma & Beta Noisy |
|---------|---------------|-------------------|
| Parameters learned | 1 (sigma) | 2 (sigma, beta) |
| Observation noise | None | Gaussian (σ=0.5) |
| Learning difficulty | Easy | Medium |
| Convergence time | Fast (~100 obs) | Slower (~250 obs) |
| Expected error | <1% | 1-5% |

## Getting Help

If you encounter issues:

1. Check the detailed README.md
2. Review IMPLEMENTATION_SUMMARY.md for technical details
3. See COMPARISON.md for understanding experiment differences
4. Compare with working experiments like learn_sigma_perfect_enkf

## Interpreting Results

### Results file (results.pt)

```python
import torch

results = torch.load('results.pt')

# Access results
sigma_history = results['sigma_hist_mean']  # Shape: (T,)
beta_history = results['beta_hist_mean']    # Shape: (T,)
state_history = results['X_hist']           # Shape: (T, N, 3)
theta_history = results['theta_hist']       # Shape: (T, N, 3)

# Compute convergence
final_sigma = sigma_history[-1]
final_beta = beta_history[-1]
```

### Example analysis

```python
import matplotlib.pyplot as plt

sigma_mean = results['sigma_hist_mean']
sigma_std = results['sigma_hist_std']
true_sigma = results['true_params'][0, 0]

plt.figure()
plt.plot(sigma_mean, label='Estimated')
plt.axhline(true_sigma, color='r', linestyle='--', label='True')
plt.fill_between(range(len(sigma_mean)), 
                 sigma_mean - sigma_std,
                 sigma_mean + sigma_std,
                 alpha=0.3)
plt.legend()
plt.show()
```

## Final Notes

- This experiment is designed for **research and validation** purposes
- It demonstrates multi-parameter learning from noisy observations
- Results should be compared with baseline experiments
- The code is well-documented for extension and modification
