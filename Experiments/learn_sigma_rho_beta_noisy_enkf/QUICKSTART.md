# Quick Start Guide

## TL;DR

```bash
cd Experiments/learn_sigma_rho_beta_noisy_enkf
python learn_sigma_rho_beta_noisy_enkf.py
```

Runs in 30-60 minutes. Outputs 4 PNG plots + results.pt.

## What This Experiment Does

Learns **all three parameters** (sigma, rho, beta) of the Lorentz63 chaotic system from **noisy observations** using ensemble filtering.

**Difficulty**: Very High (3 parameters to estimate simultaneously)

## Before Running

1. **Data exists**: Run this once if needed
   ```bash
   cd Data/Lorentz63
   python generate_data.py
   cd ../..
   ```

2. **Package installed**: Run this once if needed
   ```bash
   pip install -e .
   ```

3. **You're in the right place**:
   ```bash
   cd Experiments/learn_sigma_rho_beta_noisy_enkf
   ```

## Running

```bash
python learn_sigma_rho_beta_noisy_enkf.py
```

Or with shell script:
```bash
bash run.sh
```

## What to Expect

### Console Output
```
======================================================================
EXPERIMENT: Learn Sigma, Rho, and Beta Parameters using State Augmented EnKF
======================================================================

[1/4] Loading data...
[2/4] Setting up dynamics...
[3/4] Setting up State Augmented EnKF (ensemble size: 75)...
[4/4] Running State Augmented EnKF...

======================================================================
RESULTS
======================================================================
Final sigma (ensemble mean): 9.XXXX
Final rho (ensemble mean):   27.XXXX
Final beta (ensemble mean):  2.XXXX
```

### Output Files (in same directory)
- `sigma_rho_beta_convergence.png` - Parameter evolution
- `trajectory_fit.png` - Predictions vs observations
- `parameter_evolution.png` - All ensemble members
- `trajectory_comparison.png` - Trajectory comparison
- `results.pt` - Raw data for analysis

### Runtime
- **CPU**: 30-60 minutes
- **Memory**: ~500 MB
- **Disk**: ~100 MB for plots + 50 MB for results.pt

## Success Criteria

✓ **Good**: All parameters converge with <5% error
⚠ **OK**: Sigma & beta converge, rho less certain
✗ **Bad**: Multiple parameters diverge

## Understanding Results

### Look at the three convergence plots:

**Parameter 1 (Sigma)**
- Should converge from 5.0 → 10.0
- If it does: ✓ Sigma is learnable

**Parameter 2 (Rho)**
- Should converge from 20.0 → 28.0
- If it does: ✓ Rho is learnable
- If it doesn't: ⚠ Rho may not be observable

**Parameter 3 (Beta)**
- Should converge from 1.5 → 2.6667
- If it does: ✓ Beta is learnable

### Look at trajectory plot:
- Does the "Final" trajectory (blue dashed) match observations better than "Initial" (orange)?
- If yes: ✓ Learning worked

## If Something Goes Wrong

**"ModuleNotFoundError: enkf_ppe"**
```bash
cd ..  # Go to project root
pip install -e .
```

**"FileNotFoundError: Data file not found"**
```bash
cd Data/Lorentz63 && python generate_data.py && cd ../..
```

**"Very slow or hanging"**
- This is normal. 75 ensemble × 500 obs takes 30-60 min
- Try smaller for testing:
  ```yaml
  # In config.yaml
  data:
    n_obs: 100           # Instead of 500
  enkf:
    ensemble_size: 30    # Instead of 75
  ```

**"Parameters don't converge"**
- Try larger ensemble:
  ```yaml
  enkf:
    ensemble_size: 100
  ```
- Try more observations:
  ```yaml
  data:
    n_obs: 1000
  ```

## Modifying Configuration

Edit `config.yaml`:

```yaml
# Observation noise level (default 0.5)
data:
  noise_std: 0.2  # Lower = easier learning

# Ensemble size (default 75)
enkf:
  ensemble_size: 100  # Larger = better accuracy, slower

# Number of observations (default 500)
data:
  n_obs: 1000  # More = better parameter estimation
```

## After Running

### View Plots
Open the four PNG files to see:
1. All three parameters converging
2. How well predictions match observations
3. Individual ensemble member trajectories
4. Trajectory comparison

### Analyze Results
```python
import torch
results = torch.load('results.pt')

# Get final estimates
sigma = results['sigma_hist_mean'][-1]
rho = results['rho_hist_mean'][-1]
beta = results['beta_hist_mean'][-1]

# Get ground truth
true_sigma, true_rho, true_beta = results['true_params'][0]

# Compute errors
print(f"Sigma error: {abs(sigma - true_sigma)/true_sigma*100:.2f}%")
print(f"Rho error: {abs(rho - true_rho)/true_rho*100:.2f}%")
print(f"Beta error: {abs(beta - true_beta)/true_beta*100:.2f}%")
```

## Key Differences from Simpler Experiments

| Experiment | Parameters | Noise | Difficulty | Time |
|------------|-----------|-------|-----------|------|
| learn_sigma_perfect | 1 | None | Low | 5 min |
| learn_sigma_beta_noisy | 2 | Yes | Medium | 15 min |
| **learn_sigma_rho_beta_noisy** | **3** | **Yes** | **Very High** | **30-60 min** |

This is the most challenging parameter learning setup!

## What This Tests

✓ Can EnKF handle high-dimensional parameter spaces?
✓ How many observations are needed for 3-parameter learning?
✓ Is rho observable from noisy observations?
✓ Does noise prevent convergence?

## Next Steps

1. **Run it** with default settings
2. **Examine plots** - What converges and what doesn't?
3. **Try variations** - Edit config.yaml and re-run
4. **Compare with baselines** - Run learn_sigma_beta_noisy_enkf for comparison
5. **Analyze results** - Load results.pt and compute statistics

## Help

- **How to run**: This section (QUICKSTART.md)
- **What it does**: README.md
- **Technical details**: IMPLEMENTATION_SUMMARY.md
- **Comparison**: COMPARISON.md

## File Structure

```
learn_sigma_rho_beta_noisy_enkf/
├── config.yaml                           # Modify this for settings
├── learn_sigma_rho_beta_noisy_enkf.py   # Run this
├── QUICKSTART.md                         # This file
├── README.md                             # Detailed docs
└── [Output after running]
    ├── sigma_rho_beta_convergence.png
    ├── trajectory_fit.png
    ├── parameter_evolution.png
    ├── trajectory_comparison.png
    └── results.pt
```

Good luck! 🚀
