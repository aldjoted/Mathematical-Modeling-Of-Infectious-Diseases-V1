"""Visualizes MCMC parameter samples by generating and saving histograms.

This script serves as an entry point to plot histograms for MCMC (Markov Chain
Monte Carlo) simulation results. It adjusts the Python path to ensure correct
module imports from within the project structure and then calls the
`parameters_histograms` function from the `mcmc_sample_histograms` module.
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_scripts_dir = os.path.dirname(current_dir)
project_root_dir = os.path.dirname(project_scripts_dir)

# Ensure the project root directory is in sys.path for module resolution.
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

try:
    from scripts.DataVisualization.mcmc_sample_histograms import parameters_histograms
except ModuleNotFoundError:
    print("Attempting fallback import for parameters_histograms...")
    try:
        from mcmc_sample_histograms import parameters_histograms
    except ImportError as e:
        print(f"Error: Could not import 'parameters_histograms'. Ensure the files are in the correct directories.")
        print(f"Details: {e}")
        print(f"Current sys.path: {sys.path}")
        sys.exit(1)


if __name__ == '__main__':
    print("Starting MCMC sample histogram plotting...")
    parameters_histograms()
    print("Histogram plotting finished.")