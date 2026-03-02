#!/bin/bash
# Run the learn_sigma_perfect experiment

cd "$(dirname "$0")"

echo "Starting learn_sigma_perfect experiment..."
echo "==========================================="

python learn_sigma_perfect.py

echo ""
echo "Experiment completed!"
echo "Check results/ directory for outputs."
