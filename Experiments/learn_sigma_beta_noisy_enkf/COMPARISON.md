# Comparison: learn_sigma_perfect_enkf vs learn_sigma_beta_noisy_enkf

## Side-by-Side Comparison

### Parameters

| Parameter | learn_sigma_perfect_enkf | learn_sigma_beta_noisy_enkf |
|-----------|--------------------------|----------------------------|
| Sigma     | Learned (wrong initial) | Learned (wrong initial) |
| Rho       | Observed (correct) | Fixed (correct) |
| Beta      | Observed (correct) | Learned (wrong initial) |

### Observations

| Aspect | learn_sigma_perfect_enkf | learn_sigma_beta_noisy_enkf |
|--------|--------------------------|----------------------------|
| Noise Level | None (σ = 0.0) | Gaussian (σ = 0.5) |
| Observation Type | Noiseless/Perfect | Realistic/Noisy |
| Signal-to-Noise | Infinite | Low (observability challenge) |

### EnKF Configuration

| Setting | learn_sigma_perfect_enkf | learn_sigma_beta_noisy_enkf |
|---------|--------------------------|----------------------------|
| Ensemble Size | 50 | 50 |
| Process Noise | 0.01 | 0.01 |
| Parameter Noise | [0.5, 0.1, 0.1] | [0.5, 0.0, 0.3] |
| Obs Noise | 0.0 | 0.5 |

### Difficulty Levels

```
Difficulty Progression:
┌────────────────────────────────────────────────────────────┐
│ Easiest                                            Hardest │
├────────────────────────────────────────────────────────────┤
│  σ = 0   │  1 param │  2 params │  σ = 0.5  │  3 params  │
│          │ perfect  │ perfect   │ (2 params)│  noisy     │
│ ─────────┴──────────┴───────────┴───────────┴────────────│
│ σ_perfect σ_beta_   σ_beta_    σ_beta_    β_rho_        │
│ _noiseless perfect_ noisy_     noisy_     noisy_        │
│           noiseless  ← you are here                      │
└────────────────────────────────────────────────────────────┘
```

## Expected Convergence Rates

### Sigma Parameter
```
learn_sigma_perfect_enkf:
  Time to 95% convergence: ~50-100 observations
  Final error: ~0.5%

learn_sigma_beta_noisy_enkf:
  Time to 95% convergence: ~150-250 observations
  Final error: ~2-5%
  Slower due to: Noise + learning 2 parameters simultaneously
```

### Beta Parameter
```
learn_sigma_beta_noisy_enkf:
  Time to 95% convergence: ~150-250 observations
  Final error: ~2-5%
  Similar to sigma (coupling in observation space)
```

## Interpretation Guide

### What Each Experiment Tests

**learn_sigma_perfect_enkf** answers:
- Can EnKF learn parameters at all?
- How fast can it converge with ideal observations?
- What is the baseline convergence behavior?

**learn_sigma_beta_noisy_enkf** answers:
- Can EnKF handle multiple parameters?
- Can it work with realistic noisy data?
- How does observation noise limit parameter identifiability?
- Are sigma and beta identifiable from noisy observations?

## Key Metrics Comparison

### Convergence Metrics

If both experiments run successfully, compare:

```python
# From learn_sigma_perfect_enkf results
sigma_perfect_final_error = abs(sigma_final - 10.0) / 10.0

# From learn_sigma_beta_noisy_enkf results
sigma_noisy_final_error = abs(sigma_final - 10.0) / 10.0
beta_noisy_final_error = abs(beta_final - 2.6667) / 2.6667

# Expected ratio
error_ratio = sigma_noisy_final_error / sigma_perfect_final_error
# Should be roughly 2-5x (noisiness penalty)
```

### Ensemble Spread

```python
# Smaller spread = more confident parameter estimates
sigma_spread_perfect = theta_hist_perfect[-1, :, 0].std()
sigma_spread_noisy = theta_hist_noisy[-1, :, 0].std()

# Noisy version should have larger spread (less confident)
# ratio = sigma_spread_noisy / sigma_spread_perfect
# Should be roughly 1.5-2.0x larger
```

## Similarity Matrix

How different are the experiments?

```
Dimension               Weight  Similarity
────────────────────────────────────────
Same EnKF algorithm       40%    ✓ (100%)
Same data source          20%    ✓ (100%)
Same ensemble setup       15%    ~ (70%)
Same learning objectives  15%    ~ (50%)
Same noise conditions     10%    ✗ (0%)
────────────────────────────────────────
Overall similarity: ~75%
```

## When to Use Each Experiment

### Use learn_sigma_perfect_enkf when:
- Validating that EnKF parameter learning works at all
- Benchmarking convergence under ideal conditions
- Understanding baseline performance with perfect data
- Teaching/learning how state augmentation works

### Use learn_sigma_beta_noisy_enkf when:
- Testing realistic parameter estimation scenarios
- Assessing robustness to observation noise
- Understanding multi-parameter learning challenges
- Evaluating ensemble filtering in practice
- Comparing with real measurement systems

## Hypothetical Results Analysis

### Scenario 1: Both Parameters Converge Well
```
Final sigma error: 1.3%
Final beta error: 1.6%
→ Both parameters are identifiable from noisy observations
→ The system provides sufficient information for 2-parameter learning
→ Proceed to 3-parameter learning or real data assimilation
```

### Scenario 2: Sigma Converges, Beta Doesn't
```
Final sigma error: 1.2%
Final beta error: 8.5%
→ Beta is not well-identifiable from these observations
→ Either: (a) observation noise too high, (b) beta doesn't affect
   the observed variables as much, or (c) need more information
→ Try: Higher ensemble size, or learn sigma only with this noise level
```

### Scenario 3: Both Parameters Diverge
```
Final sigma error: 15%+
Final beta error: 20%+
→ 2-parameter learning fails with this noise level
→ Either: (a) ensemble size too small, (b) noise too high,
   or (c) parameters too weakly coupled to observations
→ Try: Increase ensemble size, reduce observation noise, or
   go back to 1-parameter learning
```

## Running Both Experiments for Comparison

```bash
# 1. Run the perfect observation version
cd Experiments/learn_sigma_perfect_enkf
python learn_sigma_perfect_enkf.py
cp results.pt ../learn_sigma_beta_noisy_enkf/results_perfect.pt

# 2. Run the noisy observation version
cd ../learn_sigma_beta_noisy_enkf
python learn_sigma_beta_noisy_enkf.py

# 3. Compare results with both results files
# (You can create a comparison script to load and analyze both)
```

## Potential Extensions

From **learn_sigma_beta_noisy_enkf**, you could extend to:

1. **learn_sigma_beta_rho_noisy_enkf**: Learn all 3 parameters
   - Highest difficulty level
   - Tests full parameter identifiability
   - Most realistic scenario

2. **learn_sigma_variable_noise_enkf**: Vary noise level
   - Systematic study of noise robustness
   - Find minimum SNR for parameter learning
   - Identify noise tolerance thresholds

3. **learn_sigma_beta_adaptive_ensemble_enkf**: Adaptive ensemble sizes
   - Dynamically adjust ensemble size
   - Focus computation on learning phases
   - Optimize for speed vs accuracy

## Summary Table

| Aspect | Perfect (Sigma Only) | Noisy (Sigma + Beta) |
|--------|----------------------|----------------------|
| Challenge Level | Low | High |
| Parameter Count | 1 | 2 |
| Observation Noise | No | Yes (σ=0.5) |
| Learning Time | Fast (~100 obs) | Slower (~250 obs) |
| Typical Errors | <1% | 1-5% |
| Real-world Relevance | Low | High |
| Good For | Validation/Baseline | Realistic Testing |
