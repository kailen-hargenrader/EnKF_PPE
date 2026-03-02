#!/bin/bash

# Run the learn_sigma_rho_beta_noisy_enkf experiment
# This learns all three (sigma, rho, beta) parameters from noisy observations using State Augmented EnKF

cd "$(dirname "$0")"

echo "=========================================="
echo "Running: Learn All Three Parameters from Noisy Observations (EnKF)"
echo "=========================================="
echo ""
echo "Note: This experiment learns SIGMA, RHO, and BETA parameters."
echo "Expected runtime: 30-60 minutes"
echo ""

python learn_sigma_rho_beta_noisy_enkf.py

echo ""
echo "=========================================="
echo "Experiment complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - sigma_rho_beta_convergence.png: All 3 parameters convergence"
echo "  - trajectory_fit.png: Ensemble mean vs observations"
echo "  - parameter_evolution.png: All ensemble member evolution (3 params)"
echo "  - trajectory_comparison.png: Trajectory comparison"
echo "  - results.pt: Complete results as PyTorch file"
echo ""
echo "Analysis suggestions:"
echo "  1. Check convergence of all three parameters"
echo "  2. Compare with learn_sigma_beta_noisy_enkf results"
echo "  3. Analyze which parameters are more observable"
echo ""
