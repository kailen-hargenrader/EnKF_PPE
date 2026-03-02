#!/bin/bash

# Run the learn_sigma_beta_noisy_enkf experiment
# This learns both sigma and beta parameters from noisy observations using State Augmented EnKF

cd "$(dirname "$0")"

echo "=========================================="
echo "Running: Learn Sigma & Beta from Noisy Observations (EnKF)"
echo "=========================================="
echo ""

python learn_sigma_beta_noisy_enkf.py

echo ""
echo "=========================================="
echo "Experiment complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - sigma_beta_convergence.png: Parameter convergence plots"
echo "  - trajectory_fit.png: Ensemble mean vs observations"
echo "  - parameter_evolution.png: All ensemble member evolution"
echo "  - trajectory_comparison.png: Trajectory comparison"
echo "  - results.pt: Complete results as PyTorch file"
echo ""
