"""Plots SEPAIHRD model simulation dynamics from CSV files.

This script reads simulation output for baseline and intervention scenarios,
then generates and saves plots for each compartment, stratified by age group
and total population, to a specified output directory.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import seaborn as sns

COMPARTMENTS = ['S', 'E', 'P', 'A', 'I', 'H', 'ICU', 'R', 'D']
LINE_STYLES = ['-', '--', '-.', ':']


def plot_simulation_dynamics(csv_filepath: str, output_dir: str, num_age_classes: int = 4, file_prefix: str = "", use_log_scale: bool = False):
    """Plots SEPAIHRD simulation dynamics from a CSV file.

    Reads simulation results and generates plots for each compartment,
    stratified by age group and total population. Plots are saved to the
    specified output directory. If the CSV file is not found, or 'Time' column
    is missing, an error message is printed and the function returns.

    Args:
        csv_filepath (str): Path to the input CSV file. Must contain a 'Time'
                            column and data columns for compartments (e.g., 'S0').
        output_dir (str): Directory to save the generated plots.
        num_age_classes (int, optional): Number of age classes. Defaults to 4.
        file_prefix (str, optional): Prefix for output plot filenames.
                                     Defaults to "".
        use_log_scale (bool, optional): Whether to use a logarithmic y-axis.
                                        Defaults to False.
    """
    if not os.path.exists(csv_filepath):
        print(f"Error: CSV file not found at {csv_filepath}")
        return

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    try:
        df = pd.read_csv(csv_filepath)
    except Exception as e:
        print(f"Error reading CSV file {csv_filepath}: {e}")
        return

    if 'Time' not in df.columns:
        print(f"Error: 'Time' column not found in {csv_filepath}")
        return

    time_points = df['Time']
    palette = sns.color_palette("husl", num_age_classes)

    for compartment_prefix in COMPARTMENTS:
        plt.figure(figsize=(14, 9))

        age_group_cols = []
        for i in range(num_age_classes):
            col_name = f"{compartment_prefix}{i}"
            if col_name in df.columns:
                age_group_cols.append(col_name)
                plt.plot(time_points, df[col_name],
                         label=f'{compartment_prefix} Age Group {i}',
                         color=palette[i % len(palette)],
                         linestyle=LINE_STYLES[i % len(LINE_STYLES)],
                         alpha=0.8, linewidth=1.5)
            else:
                print(f"Warning: Column {col_name} not found in {csv_filepath}. Skipping for {compartment_prefix} plot.")

        if not age_group_cols:
            print(f"No data columns found for compartment {compartment_prefix}. Skipping plot.")
            plt.close()
            continue

        df_compartment_sum = df[age_group_cols].sum(axis=1)
        plt.plot(time_points, df_compartment_sum, label=f'Total {compartment_prefix}', color='black', linestyle='--', linewidth=2.5)

        title_prefix_str = file_prefix.replace("_", " ").capitalize().strip()
        plt.title(f'{title_prefix_str} {compartment_prefix} Dynamics Over Time', fontsize=18, fontweight='bold')
        plt.xlabel('Time (days)', fontsize=15)
        plt.ylabel('Number of Individuals' + (' (Log Scale)' if use_log_scale else ''), fontsize=15)

        if use_log_scale:
            plt.yscale('log')
            plt.ylim(bottom=max(0.1, df_compartment_sum[df_compartment_sum > 0].min() / 10 if df_compartment_sum[df_compartment_sum > 0].any() else 0.1))

        plt.legend(fontsize=11, title=f'{compartment_prefix} Groups', title_fontsize='13')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout(pad=1.5)

        plot_filename = f"{file_prefix}{compartment_prefix.lower()}_dynamics{'_log' if use_log_scale else ''}.png"
        plot_filepath = os.path.join(output_dir, plot_filename)
        try:
            plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {plot_filepath}")
        except Exception as e:
            print(f"Error saving plot {plot_filepath}: {e}")
        plt.close()


def main():
    """Parses command-line arguments and plots simulation dynamics.

    Handles argument parsing for input CSV files (baseline and intervention),
    output directory, number of age classes, and y-axis scale.
    It then calls `plot_simulation_dynamics` for both baseline and
    intervention data if the respective files exist.
    """
    parser = argparse.ArgumentParser(description="Plot SEPAIHRD model simulation dynamics from CSV files.")
    parser.add_argument("--baseline_csv", type=str,
                        default="data/output/sepaihrd_age_baseline_results.csv",
                        help="Path to the baseline simulation results CSV file.")
    parser.add_argument("--intervention_csv", type=str,
                        default="data/output/sepaihrd_age_intervention_results.csv",
                        help="Path to the intervention simulation results CSV file.")
    parser.add_argument("--output_dir", type=str,
                        default="data/output/plots/simulation_dynamics",
                        help="Directory to save the generated plots.")
    parser.add_argument("--num_age_classes", type=int, default=4,
                        help="Number of age classes in the model.")
    parser.add_argument("--log_scale", action='store_true',
                        help="Use logarithmic scale for the y-axis.")

    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    baseline_csv_abs = os.path.join(project_root, args.baseline_csv)
    intervention_csv_abs = os.path.join(project_root, args.intervention_csv)
    output_dir_abs = os.path.join(project_root, args.output_dir)

    print(f"Project root detected as: {project_root}")
    print(f"Attempting to load baseline data from: {baseline_csv_abs}")
    print(f"Attempting to load intervention data from: {intervention_csv_abs}")
    print(f"Plots will be saved to: {output_dir_abs}")
    if args.log_scale:
        print("Using logarithmic scale for y-axis.")

    # Plot baseline simulation results if the CSV file exists.
    if os.path.exists(baseline_csv_abs):
        print("\nProcessing baseline simulation results...")
        plot_simulation_dynamics(baseline_csv_abs, output_dir_abs, args.num_age_classes,
                                 file_prefix="baseline_", use_log_scale=args.log_scale)
    else:
        print(f"\nWarning: Baseline CSV file not found at {baseline_csv_abs}. Skipping baseline plots.")

    # Plot intervention simulation results if the CSV file exists.
    if os.path.exists(intervention_csv_abs):
        print("\nProcessing intervention simulation results...")
        plot_simulation_dynamics(intervention_csv_abs, output_dir_abs, args.num_age_classes,
                                 file_prefix="intervention_", use_log_scale=args.log_scale)
    else:
        print(f"\nWarning: Intervention CSV file not found at {intervention_csv_abs}. Skipping intervention plots.")

    print("\nScript finished.")


if __name__ == "__main__":
    main()