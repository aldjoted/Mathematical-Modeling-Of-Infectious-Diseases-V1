"""
Post-calibration analysis and visualization script for SEPAIHRD model outputs
(Updated for memory-optimized C++ PostCalibrationAnalyser)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Define age group labels globally
AGE_GROUPS = ['0-30', '30-60', '60-80', '80+']

# Define NPI periods with string dates for robust plotting
NPI_PERIODS_DEF = [
    ("2020-03-01", "2020-03-14", 'Baseline (CoMix)', 'lightgray', 0.2),
    ("2020-03-15", "2020-05-03", 'Strict Lockdown', 'lightcoral', 0.2),
    ("2020-05-04", "2020-05-24", 'De-esc. Ph 0-1', 'palegoldenrod', 0.3),
    ("2020-05-25", "2020-06-20", 'De-esc. Ph 2-early 3', 'lightblue', 0.3),
    ("2020-06-21", "2020-08-31", '"New Normality" / Summer', 'lightgreen', 0.2),
    ("2020-09-01", "2020-10-24", 'Autumn Resurgence', 'sandybrown', 0.3),
    ("2020-10-25", "2020-12-26", 'Second State of Alarm', 'plum', 0.3),
    ("2020-12-27", "2020-12-31", 'End of Year / Vax Start', 'lightsteelblue', 0.3)
]

def add_npi_shading(ax, periods=NPI_PERIODS_DEF):
    """Adds NPI period shading to an Axes object based on dates."""
    y_min, y_max = ax.get_ylim()
    for start_str, end_str, label, color, alpha in periods:
        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)
        ax.axvspan(start_date, end_date, alpha=alpha, color=color)
    ax.set_ylim(y_min, y_max)

class SEPAIHRDAnalyzer:
    def __init__(self, output_dir_base, start_date_str):
        self.output_dir_base = Path(output_dir_base)
        self.figures_dir = self.output_dir_base / "PostCalibrationFigures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.start_date = pd.to_datetime(start_date_str)

    def _get_filepath(self, *subpaths):
        return self.output_dir_base / Path(*subpaths)

    def _load_csv(self, *subpaths, check_time=True, **kwargs):
        filepath = self._get_filepath(*subpaths)
        if filepath.exists():
            try:
                df = pd.read_csv(filepath, **kwargs)
                if check_time and 'time' in df.columns:
                    df['date'] = self.start_date + pd.to_timedelta(df['time'], unit='D')
                return df
            except pd.errors.EmptyDataError:
                print(f"Warning: {filepath} is empty.")
                return None
            except Exception as e:
                print(f"Warning: Could not load {filepath}. Error: {e}")
                return None
        else:
            print(f"Warning: File not found - {filepath}")
            return None
        
    def _format_date_axis(self, ax):
        """Helper to format date axes consistently."""
        locator = mdates.MonthLocator(interval=2)
        formatter = mdates.DateFormatter('%b \'%y')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    def plot_posterior_predictive_checks(self):
        """Plot daily and cumulative incidence with observed data and model uncertainty."""
        data_types = ["daily_hospitalizations", "daily_icu_admissions", "daily_deaths"]
        
        for dtype in data_types:
            fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
            
            # --- Daily Data ---
            median_df = self._load_csv("posterior_predictive", f"{dtype}_median.csv")
            lower95_df = self._load_csv("posterior_predictive", f"{dtype}_lower95.csv")
            upper95_df = self._load_csv("posterior_predictive", f"{dtype}_upper95.csv")
            observed_df = self._load_csv("posterior_predictive", f"{dtype}_observed.csv")

            ax = axes[0]
            if all(df is not None for df in [median_df, lower95_df, upper95_df]):
                median_total = median_df.filter(like='_age').sum(axis=1)
                lower95_total = lower95_df.filter(like='_age').sum(axis=1)
                upper95_total = upper95_df.filter(like='_age').sum(axis=1)
                
                ax.plot(median_df['date'], median_total, label='Model Median', color='navy', lw=2)
                ax.fill_between(median_df['date'], lower95_total, upper95_total, alpha=0.3, color='cornflowerblue', label='Model 95% CrI')

            if observed_df is not None:
                observed_total = observed_df.filter(like='_age').sum(axis=1)
                ax.plot(observed_df['date'], observed_total, 'o', label='Observed Data', color='red', markersize=2, alpha=0.7)

            ax.set_title(f'Posterior Predictive Check: Daily {dtype.replace("_", " ").title()}', fontsize=14)
            ax.set_ylabel('Count', fontsize=12)
            ax.legend()
            add_npi_shading(ax, periods=NPI_PERIODS_DEF)
            
            # --- Cumulative Data ---
            cum_dtype = f"cumulative_{dtype.split('_', 1)[1]}"
            median_cum_df = self._load_csv("posterior_predictive", f"{cum_dtype}_median.csv")
            lower95_cum_df = self._load_csv("posterior_predictive", f"{cum_dtype}_lower95.csv")
            upper95_cum_df = self._load_csv("posterior_predictive", f"{cum_dtype}_upper95.csv")
            observed_cum_df = self._load_csv("posterior_predictive", f"{cum_dtype}_observed.csv")

            ax = axes[1]
            if all(df is not None for df in [median_cum_df, lower95_cum_df, upper95_cum_df]):
                median_cum_total = median_cum_df.filter(like='_age').sum(axis=1)
                lower95_cum_total = lower95_cum_df.filter(like='_age').sum(axis=1)
                upper95_cum_total = upper95_cum_df.filter(like='_age').sum(axis=1)

                ax.plot(median_cum_df['date'], median_cum_total, label='Model Median', color='darkgreen', lw=2)
                ax.fill_between(median_cum_df['date'], lower95_cum_total, upper95_cum_total, alpha=0.3, color='lightgreen', label='Model 95% CrI')

            if observed_cum_df is not None:
                observed_cum_total = observed_cum_df.filter(like='_age').sum(axis=1)
                ax.plot(observed_cum_df['date'], observed_cum_total, 'o', label='Observed Data', color='sienna', markersize=2, alpha=0.7)
            
            ax.set_title(f'Posterior Predictive Check: {cum_dtype.replace("_", " ").title()}', fontsize=14)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Cumulative Count', fontsize=12)
            ax.legend()
            add_npi_shading(ax, periods=NPI_PERIODS_DEF)
            self._format_date_axis(ax)

            plt.tight_layout()
            plt.savefig(self.figures_dir / f"ppc_{dtype}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Plotted PPC for {dtype}")

    def plot_age_specific_severity_metrics_bar(self):
        """Plot IFR, IHR, IICUR by age group using aggregated MCMC summary."""
        summary_df = self._load_csv("mcmc_aggregated", "metrics_summary.csv", check_time=False, index_col=0)
        if summary_df is None:
            print("Aggregated MCMC scalar metrics summary not found.")
            return
        
        metrics_to_plot = {
            "IFR": ("Infection Fatality Rate", "darkred"),
            "IHR": ("Infection Hospitalization Rate", "darkblue"),
            "IICUR": ("Infection ICU Admission Rate", "darkgreen")
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)

        for i, (metric_prefix, (metric_title, color)) in enumerate(metrics_to_plot.items()):
            ax = axes[i]
            
            medians = [summary_df.loc[f'{metric_prefix}_age_{j}']['median'] * 100 for j in range(len(AGE_GROUPS))]
            lower_errors = [(summary_df.loc[f'{metric_prefix}_age_{j}']['median'] - summary_df.loc[f'{metric_prefix}_age_{j}']['q025']) * 100 for j in range(len(AGE_GROUPS))]
            upper_errors = [(summary_df.loc[f'{metric_prefix}_age_{j}']['q975'] - summary_df.loc[f'{metric_prefix}_age_{j}']['median']) * 100 for j in range(len(AGE_GROUPS))]
            
            ax.bar(AGE_GROUPS, medians, yerr=[lower_errors, upper_errors], color=color, alpha=0.7, capsize=5)
            ax.set_ylabel(f'{metric_prefix} (%)', fontsize=12)
            ax.set_title(metric_title, fontsize=14)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.figures_dir / "age_specific_severity_metrics_bar_CI.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Plotted age-specific severity metrics with CIs.")
        
    def plot_parameter_posteriors_kde(self):
        """Plot KDE of key parameter posteriors."""
        samples_df = self._load_csv("parameter_posteriors", "posterior_samples.csv", check_time=False)
        if samples_df is None: return
        
        params_to_plot = [p for p in samples_df.columns if p not in ['sample_id', 'objective_value']]
        num_params = len(params_to_plot)
        cols = 5
        rows = int(np.ceil(num_params / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        axes = axes.flatten()

        for i, param_name in enumerate(params_to_plot):
            ax = axes[i]
            sns.kdeplot(samples_df[param_name], ax=ax, fill=True, linewidth=1.5, color='crimson')
            
            median_val = samples_df[param_name].median()
            ax.axvline(median_val, color='black', linestyle='--', label=f'Median: {median_val:.3g}')
            
            ax.set_title(f'Posterior: {param_name}', fontsize=10)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.legend(fontsize=8)

        for j in range(i + 1, len(axes)): fig.delaxes(axes[j])

        plt.tight_layout()
        plt.suptitle('Key Parameter Posterior Distributions', fontsize=16, y=1.02)
        plt.savefig(self.figures_dir / "parameter_posteriors_kde.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Plotted KDEs for key parameter posteriors.")

    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("\n--- Starting Python Post-Calibration Analysis ---")
        
        self.plot_posterior_predictive_checks()
        self.plot_age_specific_severity_metrics_bar()
        self.plot_parameter_posteriors_kde()
        
        # Les autres plots dépendent de fichiers qui ne sont plus générés par la version optimisée
        # Vous pouvez les réactiver si vous modifiez la sortie C++ pour créer ces fichiers agrégés
        # self.plot_reproduction_number_with_ci() 
        # self.plot_seroprevalence_trajectory()
        # self.plot_hidden_compartments_aggregated()
        # self.plot_scenario_trajectory_comparison()
        # self.plot_scenario_summary_bars()
        
        # self.generate_html_report()
        
        print("\n--- Python Analysis Complete! ---")
        print(f"Figures saved to: {self.figures_dir}")

def main():
    parser = argparse.ArgumentParser(
        description='Analyze SEPAIHRD model post-calibration outputs (Python Script)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    try:
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent.parent
        default_output_dir = project_root / 'data' / 'output'
    except NameError:
        project_root = Path.cwd()
        default_output_dir = project_root / 'data' / 'output'
        print(f"Warning: '__file__' not found. Assuming project root is current directory: {project_root}")

    parser.add_argument(
        '--output-dir', 
        type=str,
        default=str(default_output_dir),
        help='Path to the base output directory from C++ PostCalibrationAnalyser'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2020-03-01',
        help='The start date of the simulation (YYYY-MM-DD), corresponding to time=0.'
    )
    
    args = parser.parse_args()
    output_dir_to_analyze = Path(args.output_dir)

    print(f"--- Python Analysis Runner ---")
    print(f"Analyzing data from: {output_dir_to_analyze.resolve()}")
    print(f"Using simulation start date: {args.start_date}")

    try:
        analyzer = SEPAIHRDAnalyzer(output_dir_to_analyze, args.start_date)
        analyzer.run_full_analysis()
    except FileNotFoundError as e:
        print(f"\nCritical Error: {e}")
        print("Please ensure the C++ simulation has been run and the output directory exists.")
        print(f"Expected directory structure: {output_dir_to_analyze}/posterior_predictive, etc.")
        return 1
    except Exception as e:
        print(f"\nAn unexpected error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    main()