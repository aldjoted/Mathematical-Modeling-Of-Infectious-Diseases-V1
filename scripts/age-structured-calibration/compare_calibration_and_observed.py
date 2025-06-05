import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import numpy as np
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)
from scripts.utils.FileUtils import FileUtils


class CalibrationComparison:
    def __init__(self):
        # Age labels to match CSV column naming (with "I" prefix for simulated)
        self.age_labels = ["0_30", "30_60", "60_80", "80_plus"]
        self.display_labels = ["0-30", "30-60", "60-80", "80+"]  # For plot titles
        self.colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.rcParams.update({'font.size': 12})
        
    def load_data(self, observed_file, simulated_file):
        """Load both observed and simulated data"""
        self.observed_data = pd.read_csv(observed_file)
        self.simulated_data = pd.read_csv(simulated_file)
        # Optional sanity-check
        # print("Observed columns:", self.observed_data.columns.tolist())
        # print("Simulated columns:", self.simulated_data.columns.tolist())
        
    def plot_comparison(self, output_dir):
        """Create comparison plots for each age group"""
        time_points = range(len(self.observed_data))
        
        # Create subplots for each age group
        _, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (age_group, display_label, color) in enumerate(zip(
            self.age_labels, self.display_labels, self.colors)):
            ax = axes[idx]
            
            # Plot observed data
            obs_col = f"new_confirmed_{age_group}"
            ax.scatter(time_points, self.observed_data[obs_col], 
                       alpha=0.5, color=color, label='Observed', s=30)
            
            # Plot simulated data (with 'I' prefix)
            sim_col = f"simulated_I_{age_group}"
            ax.plot(time_points, self.simulated_data[sim_col], 
                    color=color, linestyle='--', label='Simulated')
            
            ax.set_title(f'Age Group: {display_label}')
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('New Confirmed Cases')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/calibration_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_simulated_only(self, output_dir):
        """Create a single plot with all simulated trajectories"""
        time_points = range(len(self.simulated_data))
        
        plt.figure(figsize=(12, 8))
        
        for age_group, display_label, color in zip(
            self.age_labels, self.display_labels, self.colors):
            sim_col = f"simulated_I_{age_group}"
            plt.plot(time_points, self.simulated_data[sim_col], 
                     color=color, label=f'Age {display_label}')
        
        plt.title('Simulated Cases by Age Group')
        plt.xlabel('Time (days)')
        plt.ylabel('Number of Cases')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add total cases across all groups
        sim_cols = [f"simulated_I_{ag}" for ag in self.age_labels]
        total_cases = self.simulated_data[sim_cols].sum(axis=1)
        plt.plot(time_points, total_cases, 'k--', label='Total', linewidth=2)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/simulated_cases.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    # Initialize utilities and comparison class
    file_utils = FileUtils()
    project_root = file_utils.get_project_root()
    comparison = CalibrationComparison()
    
    # Define file paths
    observed_data_path = f"{project_root}/data/processed/processed_data.csv"
    simulated_data_path = f"{project_root}/data/calibration_output/simulated_incidence_best_fit.csv"
    output_dir = f"{project_root}/data/visualizations"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data and generate plots
    comparison.load_data(observed_data_path, simulated_data_path)
    comparison.plot_comparison(output_dir)
    comparison.plot_simulated_only(output_dir)

if __name__ == "__main__":
    main()
