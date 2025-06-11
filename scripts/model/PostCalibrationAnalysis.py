"""
Post-calibration analysis and visualization script for SEPAIHRD model outputs
(Updated to use aggregated trajectory data from memory-optimized C++ PostCalibrationAnalyser)
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
                median_total = median_df.filter(like='age_').sum(axis=1)
                lower95_total = lower95_df.filter(like='age_').sum(axis=1)
                upper95_total = upper95_df.filter(like='age_').sum(axis=1)

                ax.plot(median_df['date'], median_total, label='Model Median', color='navy', lw=2)
                ax.fill_between(median_df['date'], lower95_total, upper95_total, alpha=0.3, color='cornflowerblue', label='Model 95% CrI')

            if observed_df is not None:
                observed_total = observed_df.filter(like='age_').sum(axis=1)
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
                median_cum_total = median_cum_df.filter(like='age_').sum(axis=1)
                lower95_cum_total = lower95_cum_df.filter(like='age_').sum(axis=1)
                upper95_cum_total = upper95_cum_df.filter(like='age_').sum(axis=1)

                ax.plot(median_cum_df['date'], median_cum_total, label='Model Median', color='darkgreen', lw=2)
                ax.fill_between(median_cum_df['date'], lower95_cum_total, upper95_cum_total, alpha=0.3, color='lightgreen', label='Model 95% CrI')

            if observed_cum_df is not None:
                observed_cum_total = observed_cum_df.filter(like='age_').sum(axis=1)
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
        
        params_to_plot = [p for p in samples_df.columns if p not in ['sample_index', 'objective_value']]
        num_params = len(params_to_plot)
        cols = 5
        rows = int(np.ceil(num_params / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        axes = axes.flatten()

        for i, param_name in enumerate(params_to_plot):
            ax = axes[i]
            param_data = samples_df[param_name]
            
            if param_data.var() == 0:
                # Handle zero variance: plot a vertical line at the constant value
                const_val = param_data.iloc[0]
                ax.axvline(const_val, color='blue', linestyle='-', label=f'Constant: {const_val:.3g}')
                ax.text(0.5, 0.5, "Zero variance", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red')
                print(f"Warning: Parameter '{param_name}' has zero variance. Plotting as constant.")
            else:
                sns.kdeplot(param_data, ax=ax, fill=True, linewidth=1.5, color='crimson', warn_singular=False)
            
            median_val = param_data.median()
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

    def plot_reproduction_number_with_ci(self):
        """Plot time-varying reproduction number with uncertainty bands."""
        rt_df = self._load_csv("rt_trajectories", "Rt_aggregated_with_uncertainty.csv")
        if rt_df is None:
            print("Rt trajectory data not found.")
            return
        
        rt_df['date'] = self.start_date + pd.to_timedelta(rt_df['time'], unit='D')
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot median with uncertainty bands
        ax.plot(rt_df['date'], rt_df['median'], label='Median $R_t$', color='darkblue', lw=2)
        ax.fill_between(rt_df['date'], rt_df['q025'], rt_df['q975'], 
                       alpha=0.3, color='cornflowerblue', label='95% CrI')
        ax.fill_between(rt_df['date'], rt_df['q05'], rt_df['q95'], 
                       alpha=0.2, color='lightblue', label='90% CrI')
        
        # Add reference line at R=1
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='$R_t = 1$')
        
        ax.set_title('Time-Varying Reproduction Number $R_t$ with Uncertainty', fontsize=16)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('$R_t$', fontsize=14)
        ax.set_ylim(0, ax.get_ylim()[1])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        add_npi_shading(ax, periods=NPI_PERIODS_DEF)
        self._format_date_axis(ax)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "Rt_trajectory_with_uncertainty.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Plotted Rt trajectory with uncertainty bands.")

    def plot_seroprevalence_trajectory(self):
        """Plot seroprevalence trajectory with uncertainty."""
        sero_df = self._load_csv("seroprevalence", "seroprevalence_trajectory.csv")
        if sero_df is None:
            print("Seroprevalence trajectory data not found.")
            return
        
        sero_df['date'] = self.start_date + pd.to_timedelta(sero_df['time'], unit='D')
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot median with uncertainty bands (convert to percentage)
        ax.plot(sero_df['date'], sero_df['median'] * 100, label='Median Seroprevalence', 
               color='darkgreen', lw=2)
        ax.fill_between(sero_df['date'], sero_df['q025'] * 100, sero_df['q975'] * 100, 
                       alpha=0.3, color='lightgreen', label='95% CrI')
        
        # Add ENE-COVID data point
        ene_covid_date = pd.to_datetime('2020-05-04')
        ax.scatter([ene_covid_date], [4.8], color='red', s=100, marker='o', 
                  label='ENE-COVID (4.8%, 95% CI: 4.3-5.4%)', zorder=10)
        ax.errorbar([ene_covid_date], [4.8], yerr=[[0.5], [0.6]], color='red', 
                   fmt='none', capsize=5, capthick=2)
        
        ax.set_title('Seroprevalence Trajectory with Uncertainty', fontsize=16)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Seroprevalence (%)', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        add_npi_shading(ax, periods=NPI_PERIODS_DEF)
        self._format_date_axis(ax)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "seroprevalence_trajectory.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Plotted seroprevalence trajectory.")

    def plot_scenario_trajectory_comparison(self):
        """Plot scenario comparison for key trajectories."""
        # Load scenario comparison data
        scenario_df = self._load_csv("scenarios", "scenario_comparison.csv", check_time=False)
        if scenario_df is None:
            print("Scenario comparison data not found.")
            return
        
        # Create a figure with key metrics comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        scenarios = scenario_df['scenario'].tolist()
        metrics = [
            ('R0', 'Basic Reproduction Number $R_0$'),
            ('peak_hospital', 'Peak Hospital Occupancy'),
            ('peak_ICU', 'Peak ICU Occupancy'),
            ('total_deaths', 'Total Deaths')
        ]
        
        for i, (metric, title) in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            values = scenario_df[metric].tolist()
            colors = ['gray', 'red', 'blue'][:len(scenarios)]
            
            bars = ax.bar(scenarios, values, color=colors, alpha=0.7)
            ax.set_title(title, fontsize=14)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}' if val < 100 else f'{int(val)}',
                       ha='center', va='bottom')
        
        plt.tight_layout()
        plt.suptitle('Scenario Comparison: Key Epidemic Metrics', fontsize=16, y=1.02)
        plt.savefig(self.figures_dir / "scenario_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Plotted scenario comparison.")

    def plot_scenario_summary_bars(self):
        """Plot summary bars for all scenarios."""
        scenario_df = self._load_csv("scenarios", "scenario_comparison.csv", check_time=False)
        if scenario_df is None:
            print("Scenario data not found.")
            return
        
        # Create normalized comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Normalize metrics relative to baseline
        baseline_idx = scenario_df[scenario_df['scenario'] == 'baseline'].index[0]
        metrics_to_normalize = ['peak_hospital', 'peak_ICU', 'total_deaths', 'overall_attack_rate']
        
        normalized_data = []
        for idx, row in scenario_df.iterrows():
            if idx != baseline_idx:
                norm_values = []
                for metric in metrics_to_normalize:
                    baseline_val = scenario_df.loc[baseline_idx, metric]
                    if baseline_val > 0:
                        norm_val = (row[metric] - baseline_val) / baseline_val * 100
                    else:
                        norm_val = 0
                    norm_values.append(norm_val)
                normalized_data.append((row['scenario'], norm_values))
        
        # Plot grouped bars
        x = np.arange(len(metrics_to_normalize))
        width = 0.35
        
        for i, (scenario, values) in enumerate(normalized_data):
            offset = width * (i - 0.5)
            bars = ax.bar(x + offset, values, width, label=scenario.replace('_', ' ').title())
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', 
                       va='bottom' if height > 0 else 'top')
        
        ax.set_ylabel('Relative Change from Baseline (%)', fontsize=12)
        ax.set_title('Scenario Impact on Key Epidemic Metrics (Relative to Baseline)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_normalize])
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "scenario_impact_bars.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Plotted scenario impact bars.")

    def generate_html_report(self):
        """Generate a simple HTML report with all figures."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SEPAIHRD Post-Calibration Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }} /* Ensure CSS is static */
                h1, h2 {{ color: #333; }}
                .figure {{ margin: 30px 0; text-align: center; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                .description {{ margin: 10px 0; color: #666; }}
            </style>
        </head>
        <body>
            <h1>SEPAIHRD Model Post-Calibration Analysis Report</h1>
            <p>Generated on: {date}</p>
            
            <h2>1. Posterior Predictive Checks</h2>
            <div class="figure">
                <img src="PostCalibrationFigures/ppc_daily_hospitalizations.png">
                <p class="description">Daily and cumulative hospitalizations: Model predictions vs observed data</p>
            </div>
            <div class="figure">
                <img src="PostCalibrationFigures/ppc_daily_icu_admissions.png">
                <p class="description">Daily and cumulative ICU admissions: Model predictions vs observed data</p>
            </div>
            <div class="figure">
                <img src="PostCalibrationFigures/ppc_daily_deaths.png">
                <p class="description">Daily and cumulative deaths: Model predictions vs observed data</p>
            </div>
            
            <h2>2. Age-Specific Severity Metrics</h2>
            <div class="figure">
                <img src="PostCalibrationFigures/age_specific_severity_metrics_bar_CI.png">
                <p class="description">Age-stratified infection fatality, hospitalization, and ICU admission rates</p>
            </div>
            
            <h2>3. Parameter Posterior Distributions</h2>
            <div class="figure">
                <img src="PostCalibrationFigures/parameter_posteriors_kde.png">
                <p class="description">Posterior distributions of key model parameters</p>
            </div>
            
            <h2>4. Time-Varying Reproduction Number</h2>
            <div class="figure">
                <img src="PostCalibrationFigures/Rt_trajectory_with_uncertainty.png">
                <p class="description">Evolution of the effective reproduction number with uncertainty bands</p>
            </div>
            
            <h2>5. Seroprevalence Trajectory</h2>
            <div class="figure">
                <img src="PostCalibrationFigures/seroprevalence_trajectory.png">
                <p class="description">Model-predicted seroprevalence over time with ENE-COVID validation</p>
            </div>
            
            <h2>6. Scenario Analysis</h2>
            <div class="figure">
                <img src="PostCalibrationFigures/scenario_comparison.png">
                <p class="description">Comparison of key metrics across different intervention scenarios</p>
            </div>
            <div class="figure">
                <img src="PostCalibrationFigures/scenario_impact_bars.png">
                <p class="description">Relative impact of scenarios compared to baseline</p>
            </div>
        </body>
        </html>
        """.format(date=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        report_path = self.figures_dir / "analysis_report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        print(f"Generated HTML report: {report_path}")


    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("\n--- Starting Python Post-Calibration Analysis ---")
        
        # All functions are now active
        self.plot_posterior_predictive_checks()
        self.plot_age_specific_severity_metrics_bar()
        self.plot_parameter_posteriors_kde()
        self.plot_reproduction_number_with_ci()
        self.plot_seroprevalence_trajectory()
        self.plot_scenario_trajectory_comparison()
        self.plot_scenario_summary_bars()
        
        self.generate_html_report()
        
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