# Comparison: learn_sigma_rho_beta_noisy_enkf with Previous Experiments

## Complete Experiment Series

```
Difficulty Progression:
┌─────────────────────────────────────────────────────────────┐
│ Easiest                                              Hardest │
├─────────────────────────────────────────────────────────────┤
│ σ only   │ σ+β      │ σ+β      │ σ+ρ+β    │ All dyn   │
│ perfect  │ perfect  │ noisy    │ noisy    │ param     │
│ ─────────┴──────────┴──────────┴──────────┴─────────────│
│ σ_perfect σ_β_perf σ_β_noisy σ_ρ_β_noisy (future)    │
│                                  ↑                      │
│                            YOU ARE HERE                │
└─────────────────────────────────────────────────────────────┘
```

## Feature Comparison Matrix

| Feature | σ_perfect | σβ_perfect | σβ_noisy | **σρβ_noisy** |
|---------|-----------|-----------|----------|--------------|
| **Parameters** | | | | |
| Sigma | ✓ Learned | ✓ Learned | ✓ Learned | ✓ Learned |
| Rho | Fixed (correct) | Fixed (correct) | Fixed (correct) | **✓ Learned** |
| Beta | Fixed (correct) | ✓ Learned | ✓ Learned | ✓ Learned |
| **Observations** | | | | |
| Noise Level | None | None | Moderate (0.5) | Moderate (0.5) |
| SNR | ∞ | ∞ | Low | Low |
| Realism | Low | Low | Medium | **High** |
| **Ensemble** | | | | |
| Size | 50 | 50 | 50 | **75** |
| Spread | Small | Small | Medium | **Large** |
| **Difficulty** | | | | |
| Rating | ⭐ | ⭐⭐ | ⭐⭐⭐ | **⭐⭐⭐⭐** |
| Learn Speed | Very fast | Fast | Slow | **Very slow** |
| Parameter Identifiability | High | High | Medium | **Low** |

## Expected Convergence Times

### Convergence Speed Ranking

```
Time to 95% convergence:

Easiest (σ_perfect):
├── ~100 observations
└── ~10 seconds

Medium (σβ_noisy):
├── ~250 observations
└── ~20 seconds

Hard (σρβ_noisy):
├── ~400-500 observations
└── 30-60 minutes

Very Hard (3+ params + lower SNR):
├── >1000 observations
└── >2 hours
```

## Parameter-by-Parameter Analysis

### Sigma (Easy to Learn)

| Experiment | Learning | Error | Time |
|-----------|----------|-------|------|
| σ_perfect | Easy | <1% | Fast |
| σβ_perfect | Easy | <1% | Fast |
| σβ_noisy | Easy | 1-2% | Medium |
| **σρβ_noisy** | **Easy** | **1-3%** | **Fast** |

Sigma controls the "size" of the attractor and is observable from state magnitudes.
**Expected**: Should converge well even with 3 parameters.

### Rho (Medium to Hard to Learn)

| Experiment | Learning | Error | Time |
|-----------|----------|-------|------|
| σ_perfect | N/A (fixed) | N/A | N/A |
| σβ_perfect | N/A (fixed) | N/A | N/A |
| σβ_noisy | N/A (fixed) | N/A | N/A |
| **σρβ_noisy** | **Hard** | **5-15%** | **Very slow** |

Rho is the Rayleigh number controlling transition to chaos. It may be weakly observable.
**Expected**: Hardest parameter to estimate. May not converge well.

### Beta (Medium to Learn)

| Experiment | Learning | Error | Time |
|-----------|----------|-------|------|
| σ_perfect | N/A (fixed) | N/A | N/A |
| σβ_perfect | Medium | <1% | Medium |
| σβ_noisy | Medium | 2-3% | Medium |
| **σρβ_noisy** | **Medium** | **2-5%** | **Medium** |

Beta controls geometric aspects and return rates.
**Expected**: Should converge moderately well.

## Computational Costs

### Runtime Comparison

```
Experiment              Ensemble  Obs   Time (minutes)
────────────────────────────────────────────────────
σ_perfect              50         500   1-2
σβ_perfect            50         500   2-3
σβ_noisy              50         500   5-10
σρβ_noisy             75         500   30-60
```

### Memory Comparison

```
Experiment              Memory (MB)
────────────────────────────────────
σ_perfect              ~100
σβ_perfect            ~100
σβ_noisy              ~150
σρβ_noisy             ~300-500
```

### CPU vs GPU

```
Hardware    σ_perfect  σβ_noisy  σρβ_noisy
──────────────────────────────────────────
CPU         2 min      10 min    45 min
GPU         <1 min     2 min     5-10 min
(if available)
```

## Identifiability Analysis

### Observable Dimensions

```
Observable from full-state measurements [x, y, z]:
─────────────────────────────────────────────────

Sigma:    ✓✓✓ Highly observable
          - Controls attractor size
          - Directly affects |state|

Rho:      ✓✗ Weakly observable (depending on SNR)
          - Controls bifurcation behavior
          - Observable in long-term statistics
          - May not be identifiable from short, noisy data

Beta:     ✓✓ Moderately observable
          - Controls oscillation rates
          - Observable from frequency analysis
```

### Observable Subspace Dimension

```
σ_perfect:   1D parameter space
σβ_perfect:  2D parameter space
σρβ_noisy:   3D parameter space (3D state + 3D parameters = 6D total)
```

The higher the parameter space dimension, the harder to explore it with limited ensemble members.

## Challenge Progression

### σ_perfect (Baseline)
```
Challenge: Learn one parameter from perfect data
Difficulty: ⭐ (Very Easy)
Key challenge: None - almost trivial
→ Establishes that EnKF parameter learning works
```

### σβ_noisy (Moderate)
```
Challenge: Learn two parameters from noisy data
Difficulty: ⭐⭐⭐ (Medium)
Key challenge: Noise + multiple parameters
→ Tests whether noise degrades convergence
```

### σρβ_noisy (Advanced)
```
Challenge: Learn three parameters from noisy data
Difficulty: ⭐⭐⭐⭐ (Very Hard)
Key challenges:
1. Three parameters (curse of dimensionality)
2. Observation noise (reduces signal)
3. Rho weak observability (hardest to estimate)
4. Ensemble collapse risk (not enough particles)
→ Tests limits of practical parameter estimation
```

## Success Scenarios

### Scenario 1: All Parameters Converge

```
Final errors:
  Sigma:  1.2%  ✓ Excellent
  Rho:    2.1%  ✓ Excellent
  Beta:   1.8%  ✓ Excellent

Interpretation:
  → All three parameters are identifiable
  → Ensemble size is adequate
  → Observation information is sufficient
  → This would be remarkable result!
```

### Scenario 2: Sigma and Beta Converge, Rho Diverges

```
Final errors:
  Sigma:  1.1%  ✓ Good
  Rho:    18%   ✗ Poor
  Beta:   1.9%  ✓ Good

Interpretation:
  → Rho is not well-identifiable
  → Likely scenario (expected outcome)
  → System shows "2-parameter observability"
  → Comparable to learning only σ and β
```

### Scenario 3: All Parameters Diverge

```
Final errors:
  Sigma:  8%    ✗ Poor
  Rho:    12%   ✗ Poor
  Beta:   9%    ✗ Poor

Interpretation:
  → Ensemble too small OR ensemble collapse
  → Try: Larger ensemble (100-150 members)
  → Or: More observations (1000+)
  → Or: Reduce observation noise
```

## Key Research Questions

This experiment answers:

| Question | How | Answer Type |
|----------|-----|-------------|
| Are 3 params identifiable? | Compare errors | Yes/No |
| Which param is hardest? | Compare convergence times | Rho > Beta > Sigma |
| How much data needed? | Observe learning curves | ~400-500 obs |
| Can we handle noise? | Compare σβ_noisy vs σρβ_noisy | Yes/Maybe/No |
| Is rho observable? | Check rho convergence | Yes/No/Partially |

## Recommended Analysis Workflow

### Step 1: Establish Baseline
```bash
# Run all three experiments to establish baseline
cd ../learn_sigma_perfect_enkf && python learn_sigma_perfect_enkf.py
cd ../learn_sigma_beta_noisy_enkf && python learn_sigma_beta_noisy_enkf.py
cd learn_sigma_rho_beta_noisy_enkf && python learn_sigma_rho_beta_noisy_enkf.py
```

### Step 2: Compare Results
```python
# Load and compare results
import torch

results_σβ = torch.load('../learn_sigma_beta_noisy_enkf/results.pt')
results_σρβ = torch.load('results.pt')

# Compare sigma convergence
σβ_sigma_error = abs(results_σβ['sigma_hist_mean'][-1] - 10.0) / 10 * 100
σρβ_sigma_error = abs(results_σρβ['sigma_hist_mean'][-1] - 10.0) / 10 * 100

print(f"Sigma error (2-param): {σβ_sigma_error:.2f}%")
print(f"Sigma error (3-param): {σρβ_sigma_error:.2f}%")
# Should be similar (sigma should learn well in both)
```

### Step 3: Analyze Parameter Coupling
```python
# Check if parameters learned together or separately
sigma_hist = results_σρβ['sigma_hist_mean']
rho_hist = results_σρβ['rho_hist_mean']
beta_hist = results_σρβ['beta_hist_mean']

# Plot convergence speed comparison
import matplotlib.pyplot as plt
plt.plot(sigma_hist / 10, label='Sigma (normalized)')
plt.plot(rho_hist / 28, label='Rho (normalized)')
plt.plot(beta_hist / 2.6667, label='Beta (normalized)')
plt.legend()
plt.show()
# Faster rise = faster convergence
```

## Modification Ideas

### To Make It Easier
1. Reduce observation noise: `obs_noise_std: 0.1`
2. Use more observations: `n_obs: 1000`
3. Better initial guess for rho: `rho: 25.0` instead of `20.0`
4. Larger ensemble: `ensemble_size: 100`

### To Make It Harder
1. Increase observation noise: `obs_noise_std: 1.0`
2. Worse initial guesses: `rho: 10.0` (much farther)
3. Smaller ensemble: `ensemble_size: 50`
4. Fewer observations: `n_obs: 200`

## Extending the Series

### Next Experiments to Try

1. **All parameters + observation location uncertainty**
   - Learn what is observable from different observation sets

2. **All parameters + time-varying parameters**
   - Test parameter estimation with drifting truth

3. **All parameters + model error**
   - Add small errors to dynamics model

4. **Comparison with variational methods (4DVar)**
   - Compare EnKF with optimization-based DA

## Summary Table

| Aspect | σ_perfect | σβ_noisy | **σρβ_noisy** |
|--------|-----------|----------|--------------|
| Parameters | 1 | 2 | **3** |
| Difficulty | Easy | Medium | **Hard** |
| Convergence time | ~100 obs | ~250 obs | **~500 obs** |
| Expected error | <1% | 1-5% | **2-10%** |
| Ensemble size | 50 | 50 | **75** |
| Runtime | 2 min | 10 min | **45 min** |
| Real-world relevance | Low | Medium | **High** |
| Parameter identifiability | All high | All medium | **Mixed** |

**Key takeaway**: This experiment tests the practical limits of ensemble-based parameter estimation with realistic noise and multiple unknowns. Results should inform whether 3-parameter learning is feasible for your applications.
