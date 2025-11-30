"""
Post-calibration analysis and visualization script for SEPAIHRD model outputs
ENHANCED VERSION:
- Added Small Multiples (Faceting) for age-stratified incidence.
- Enhanced uncertainty visualization (HDI intervals on KDEs).
- Improved aesthetic guidelines (Munzner's principles).
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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')

# --- Visual Style Configuration ---
plt.style.use('seaborn-v0_8-whitegrid')  # Whitegrid is often cleaner for reading values
sns.set_context("talk", font_scale=0.9)  # Improving readability for reports
PALETTE = sns.color_palette("deep")      # Standard, accessible palette

# Define age group labels globally
AGE_GROUPS = ['0-30', '30-60', '60-80', '80+']

# Define NPI periods with string dates for robust plotting (consolidated for clarity)
NPI_PERIODS_DEF = [
    ("2020-03-01", "2020-03-14", 'Baseline', 'lightgray', 0.3),
    ("2020-03-15", "2020-05-03", 'Lockdown', 'lightcoral', 0.15),
    ("2020-05-04", "2020-06-20", 'De-escalation', 'palegoldenrod', 0.2),
    ("2020-06-21", "2020-08-31", 'New Normality', 'lightgreen', 0.15),
    ("2020-09-01", "2020-10-24", 'Autumn Wave', 'sandybrown', 0.15),
    ("2020-10-25", "2020-12-26", '2nd Alarm', 'plum', 0.2),
]

def add_npi_shading(ax, periods=NPI_PERIODS_DEF):
    """Adds NPI period shading to an Axes object based on dates."""
    y_min, y_max = ax.get_ylim()
    for start_str, end_str, label, color, alpha in periods:
        try:
            start_date = pd.to_datetime(start_str)
            end_date = pd.to_datetime(end_str)
            ax.axvspan(start_date, end_date, alpha=alpha, color=color, label=None, zorder=0)
        except Exception:
            pass  # Skip shading if dates are out of range or invalid
    # Restore limits to avoid autoscaling based on shading
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
        formatter = mdates.DateFormatter('%b %y')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    def plot_posterior_predictive_checks(self):
        """Plot daily and cumulative incidence with observed data and model uncertainty."""
        data_types = ["daily_hospitalizations", "daily_icu_admissions", "daily_deaths"]
        
        for dtype in data_types:
            fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
            
            # --- Daily Data ---
            median_df = self._load_csv("posterior_predictive", f"{dtype}_median.csv")
            lower95_df = self._load_csv("posterior_predictive", f"{dtype}_lower95.csv")
            upper95_df = self._load_csv("posterior_predictive", f"{dtype}_upper95.csv")
            observed_df = self._load_csv("posterior_predictive", f"{dtype}_observed.csv")

            ax = axes[0]
            if all(df is not None for df in [median_df, lower95_df, upper95_df]):
                # Reduction: Aggregate items (ages) to totals
                median_total = median_df.filter(like='age_').sum(axis=1)
                lower95_total = lower95_df.filter(like='age_').sum(axis=1)
                upper95_total = upper95_df.filter(like='age_').sum(axis=1)

                # Mark: Area (Uncertainty Band), Channel: Position (Y), Color (Saturation)
                ax.fill_between(median_df['date'], lower95_total, upper95_total, 
                                alpha=0.4, color=PALETTE[0], label='Model 95% CrI', zorder=2)
                # Mark: Line (Median)
                ax.plot(median_df['date'], median_total, label='Model Median', color=PALETTE[0], lw=2.5, zorder=3)

            if observed_df is not None:
                observed_total = observed_df.filter(like='age_').sum(axis=1)
                # Mark: Points (Observed), Channel: Color (Hue - Red for contrast)
                ax.plot(observed_df['date'], observed_total, 'o', label='Observed', 
                        color='firebrick', markersize=4, alpha=0.8, zorder=4)

            ax.set_title(f'Daily {dtype.replace("_", " ").title()}', fontsize=16, pad=10)
            ax.set_ylabel('Count')
            ax.legend(loc='upper right', frameon=True)
            add_npi_shading(ax)
            ax.grid(True, which='major', linestyle='--', alpha=0.6)
            
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

                ax.fill_between(median_cum_df['date'], lower95_cum_total, upper95_cum_total, 
                                alpha=0.4, color=PALETTE[2], label='Model 95% CrI', zorder=2)
                ax.plot(median_cum_df['date'], median_cum_total, label='Model Median', color=PALETTE[2], lw=2.5, zorder=3)

            if observed_cum_df is not None:
                observed_cum_total = observed_cum_df.filter(like='age_').sum(axis=1)
                ax.plot(observed_cum_df['date'], observed_cum_total, 'o', label='Observed', 
                        color='firebrick', markersize=4, alpha=0.8, zorder=4)
            
            ax.set_title(f'Cumulative {cum_dtype.replace("_", " ").title()}', fontsize=16, pad=10)
            ax.set_ylabel('Cumulative Count')
            ax.legend(loc='lower right', frameon=True)
            add_npi_shading(ax)
            self._format_date_axis(ax)

            plt.tight_layout()
            plt.savefig(self.figures_dir / f"ppc_{dtype}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Plotted PPC for {dtype}")

    def plot_age_stratified_incidence(self):
        """
        Faceted Small Multiples for Age-Specific Trajectories.
        Why: Aggregating sums hides heterogeneity. Faceting enables detailed trend comparison.
        """
        data_types = ["daily_hospitalizations", "daily_deaths"]  # Key metrics for age differentiation
        
        for dtype in data_types:
            median_df = self._load_csv("posterior_predictive", f"{dtype}_median.csv")
            observed_df = self._load_csv("posterior_predictive", f"{dtype}_observed.csv")
            
            if median_df is None:
                continue

            # Identify available age columns
            age_cols = [c for c in median_df.columns if 'age_' in c]
            n_groups = len(age_cols)
            
            if n_groups == 0:
                continue

            # Create Small Multiples (Facet)
            fig, axes = plt.subplots(n_groups, 1, figsize=(12, 3 * n_groups), sharex=True)
            if n_groups == 1:
                axes = [axes]

            for i, col in enumerate(age_cols):
                ax = axes[i]
                # Label mapping
                label = AGE_GROUPS[i] if i < len(AGE_GROUPS) else col
                
                # Plot Model
                ax.plot(median_df['date'], median_df[col], color=PALETTE[0], lw=2, label='Model Median')
                
                # Plot Observed if available
                if observed_df is not None and col in observed_df.columns:
                    ax.plot(observed_df['date'], observed_df[col], '.', color='firebrick', 
                            markersize=3, alpha=0.6, label='Observed')

                ax.set_title(f"Age Group: {label}", fontsize=12, loc='left')
                ax.set_ylabel("Count")
                add_npi_shading(ax)
                
                if i == 0:
                    ax.legend(loc='upper right', frameon=True, fontsize='small')

            self._format_date_axis(axes[-1])
            plt.suptitle(f"Age-Stratified {dtype.replace('_', ' ').title()}", fontsize=16, y=1.01)
            plt.tight_layout()
            plt.savefig(self.figures_dir / f"age_stratified_{dtype}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Plotted Age-Stratified {dtype}")

    def plot_age_specific_severity_metrics_bar(self):
        """Plot IFR, IHR, IICUR by age group using aggregated MCMC summary."""
        summary_df = self._load_csv("mcmc_aggregated", "metrics_summary.csv", check_time=False, index_col=0)
        if summary_df is None:
            print("Aggregated MCMC scalar metrics summary not found.")
            return
        
        metrics_to_plot = {
            "IFR": ("Infection Fatality Rate", "darkred"),
            "IHR": ("Infection Hospitalization Rate", "darkblue"),
            "IICUR": ("ICU Admission Rate (given Hosp)", "darkgreen")
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for i, (metric_prefix, (metric_title, color)) in enumerate(metrics_to_plot.items()):
            ax = axes[i]
            
            medians = []
            errors = [[], []]  # lower, upper
            
            valid_groups = []
            for j in range(len(AGE_GROUPS)):
                key = f'{metric_prefix}_age_{j}'
                if key in summary_df.index:
                    median = summary_df.loc[key]['median'] * 100
                    low = summary_df.loc[key]['q025'] * 100
                    high = summary_df.loc[key]['q975'] * 100
                    
                    medians.append(median)
                    errors[0].append(max(0, median - low))
                    errors[1].append(max(0, high - median))
                    valid_groups.append(AGE_GROUPS[j])

            if medians:
                bars = ax.bar(valid_groups, medians, yerr=errors, color=color, alpha=0.6, capsize=5, edgecolor='black')
                ax.set_ylabel('Rate (%)')
                ax.set_title(metric_title)
                ax.grid(axis='y', linestyle='--', alpha=0.5)
                
                # Value Annotations
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 5), textcoords="offset points",
                                ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.figures_dir / "age_specific_severity_metrics_bar_CI.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Plotted age-specific severity metrics with CIs.")
        
    def plot_parameter_posteriors_kde(self):
        """
        Enhanced Plot: KDE of key parameter posteriors with Credible Interval annotations.
        Why: Explicitly showing the 95% interval aids in assessing parameter certainty.
        """
        samples_df = self._load_csv("parameter_posteriors", "posterior_samples.csv", check_time=False)
        if samples_df is None:
            return
        
        params_to_plot = [p for p in samples_df.columns if p not in ['sample_index', 'objective_value']]
        num_params = len(params_to_plot)
        cols = 4
        rows = int(np.ceil(num_params / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        axes = axes.flatten()

        for i, param_name in enumerate(params_to_plot):
            ax = axes[i]
            param_data = samples_df[param_name]
            
            if param_data.var() == 0:
                # Handle zero variance: plot a vertical line at the constant value
                ax.axvline(param_data.iloc[0], color='blue', linestyle='-')
                ax.set_title(f'{param_name} (Fixed)', fontsize=10)
            else:
                # Plot Density
                sns.kdeplot(param_data, ax=ax, fill=True, color=PALETTE[3], alpha=0.3, linewidth=1.5, warn_singular=False)
                
                # Calculate stats
                mean_val = param_data.mean()
                q025 = param_data.quantile(0.025)
                q975 = param_data.quantile(0.975)
                
                # Mark Mean
                ax.axvline(mean_val, color='k', linestyle='--', lw=1, label=f'Mean: {mean_val:.2g}')
                
                # Mark 95% Credible Interval
                ax.axvline(q025, color='k', linestyle=':', lw=1)
                ax.axvline(q975, color='k', linestyle=':', lw=1)
                
                ax.set_title(f'{param_name}', fontsize=11, fontweight='bold')
                ax.set_xlabel('')
                ax.set_yticks([])  # Remove y ticks for cleaner look (density magnitude often irrelevant)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.suptitle('Parameter Posterior Distributions (with 95% CI)', fontsize=16, y=1.02)
        plt.savefig(self.figures_dir / "parameter_posteriors_kde.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Plotted KDEs for key parameter posteriors.")

    def plot_reproduction_number_with_ci(self):
        """Plot time-varying reproduction number with uncertainty bands (fan chart style)."""
        rt_df = self._load_csv("rt_trajectories", "Rt_aggregated_with_uncertainty.csv")
        if rt_df is None:
            print("Rt trajectory data not found.")
            return
        
        rt_df['date'] = self.start_date + pd.to_timedelta(rt_df['time'], unit='D')
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Fan Chart style: 90% inside 95% (lighter)
        ax.fill_between(rt_df['date'], rt_df['q025'], rt_df['q975'], 
                        alpha=0.15, color='purple', label='95% CrI')
        ax.fill_between(rt_df['date'], rt_df['q05'], rt_df['q95'], 
                        alpha=0.25, color='purple', label='90% CrI')
        
        ax.plot(rt_df['date'], rt_df['median'], label='Median $R_t$', color='indigo', lw=2)
        
        # Threshold line
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.8, lw=1.5, label='Threshold ($R_t=1$)')
        
        ax.set_title('Effective Reproduction Number ($R_t$)', fontsize=16)
        ax.set_ylabel('$R_t$')
        ax.set_ylim(0, max(rt_df['q975'].max(), 2.5))  # Cap high values for readability
        ax.legend(loc='upper right', frameon=True)
        ax.grid(True, alpha=0.3)
        
        add_npi_shading(ax)
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
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot median with uncertainty bands (convert to percentage)
        ax.plot(sero_df['date'], sero_df['median'] * 100, label='Median Seroprevalence', 
                color='teal', lw=2)
        ax.fill_between(sero_df['date'], sero_df['q025'] * 100, sero_df['q975'] * 100, 
                        alpha=0.2, color='teal', label='95% CrI')
        
        # Add ENE-COVID data point
        ene_covid_date = pd.to_datetime('2020-05-04')
        ax.errorbar([ene_covid_date], [4.8], yerr=[[0.5], [0.6]], color='crimson', 
                    fmt='o', capsize=5, capthick=2, label='ENE-COVID Data', zorder=5)
        ax.text(ene_covid_date, 5.6, "ENE-COVID\nValidation", color='crimson', ha='center', fontsize=9)
        
        ax.set_title('Population Seroprevalence', fontsize=16)
        ax.set_ylabel('Seroprevalence (%)')
        ax.legend(loc='upper left', frameon=True)
        ax.grid(True, alpha=0.3)
        
        add_npi_shading(ax)
        self._format_date_axis(ax)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "seroprevalence_trajectory.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Plotted seroprevalence trajectory.")

    def plot_scenario_trajectory_comparison(self):
        """Plot scenario comparison for key trajectories."""
        scenario_df = self._load_csv("scenarios", "scenario_comparison.csv", check_time=False)
        if scenario_df is None:
            print("Scenario comparison data not found.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        scenarios = scenario_df['scenario'].tolist()
        metrics = [
            ('R0', '$R_0$'),
            ('peak_hospital', 'Peak Hosp.'),
            ('peak_ICU', 'Peak ICU'),
            ('total_deaths', 'Total Deaths')
        ]
        
        colors = sns.color_palette("husl", len(scenarios))
        
        for i, (metric, title) in enumerate(metrics):
            ax = axes[i]
            values = scenario_df[metric].tolist()
            
            bars = ax.bar(scenarios, values, color=colors, alpha=0.8, edgecolor='white')
            ax.set_title(title, fontsize=14)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}' if val < 10 else f'{int(val)}',
                        ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.suptitle('Scenario Analysis: Key Metrics', fontsize=16, y=1.02)
        plt.savefig(self.figures_dir / "scenario_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Plotted scenario comparison.")

    def plot_scenario_summary_bars(self):
        """Plot summary bars for all scenarios."""
        scenario_df = self._load_csv("scenarios", "scenario_comparison.csv", check_time=False)
        if scenario_df is None:
            print("Scenario data not found.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        baseline_idx = scenario_df[scenario_df['scenario'] == 'baseline'].index
        if len(baseline_idx) == 0:
            print("Warning: 'baseline' scenario not found.")
            return
        baseline_idx = baseline_idx[0]

        metrics_to_normalize = ['peak_hospital', 'peak_ICU', 'total_deaths', 'overall_attack_rate']
        
        # Dynamic width based on number of scenarios
        x = np.arange(len(metrics_to_normalize))
        num_non_baseline = len(scenario_df) - 1
        if num_non_baseline == 0:
            return
        width = 0.8 / num_non_baseline
        
        bar_idx = 0
        for idx, row in scenario_df.iterrows():
            if idx == baseline_idx:
                continue  # Skip baseline (it's 0%)
            
            norm_values = []
            for metric in metrics_to_normalize:
                base_val = scenario_df.loc[baseline_idx, metric]
                val = (row[metric] - base_val) / base_val * 100 if base_val > 0 else 0
                norm_values.append(val)
            
            offset = width * bar_idx - (width * (num_non_baseline - 1) / 2)
            ax.bar(x + offset, norm_values, width, label=row['scenario'].title())
            bar_idx += 1
        
        ax.set_ylabel('Change relative to Baseline (%)')
        ax.set_title('Scenario Impact Analysis', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').replace('overall ', '').title() for m in metrics_to_normalize])
        ax.legend(title="Scenario")
        ax.axhline(0, color='black', lw=1)
        ax.grid(axis='y', alpha=0.3)
        
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
            <title>SEPAIHRD Analysis</title>
            <style>
                body {{ font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 40px; max-width: 1200px; margin: auto; color: #333; }}
                h1 {{ border-bottom: 2px solid #eee; padding-bottom: 10px; color: #2c3e50; }}
                h2 {{ margin-top: 40px; color: #34495e; background: #f8f9fa; padding: 10px; border-left: 5px solid #3498db; }}
                .figure-container {{ display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }}
                .figure {{ margin: 20px 0; text-align: center; background: white; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); border-radius: 8px; }}
                img {{ max-width: 100%; height: auto; }}
                .description {{ margin-top: 10px; font-style: italic; color: #7f8c8d; }}
            </style>
        </head>
        <body>
            <h1>SEPAIHRD Model Post-Calibration Report</h1>
            <p>Generated: {date}</p>
            
            <h2>1. Model Fit (Posterior Predictive Checks)</h2>
            <div class="figure-container">
                <div class="figure"><img src="PostCalibrationFigures/ppc_daily_hospitalizations.png"><p class="description">Daily Hospitalizations</p></div>
                <div class="figure"><img src="PostCalibrationFigures/ppc_daily_deaths.png"><p class="description">Daily Deaths</p></div>
            </div>

            <h2>2. Age-Stratified Dynamics</h2>
            <p>Breakdown of dynamics by age group to identify high-risk populations.</p>
            <div class="figure-container">
                <div class="figure"><img src="PostCalibrationFigures/age_stratified_daily_hospitalizations.png"><p class="description">Hospitalizations by Age</p></div>
                <div class="figure"><img src="PostCalibrationFigures/age_stratified_daily_deaths.png"><p class="description">Deaths by Age</p></div>
                <div class="figure"><img src="PostCalibrationFigures/age_specific_severity_metrics_bar_CI.png"><p class="description">Severity Rates (IFR/IHR)</p></div>
            </div>
            
            <h2>3. Epidemiological Drivers</h2>
            <div class="figure-container">
                <div class="figure"><img src="PostCalibrationFigures/Rt_trajectory_with_uncertainty.png"><p class="description">Effective Reproduction Number</p></div>
                <div class="figure"><img src="PostCalibrationFigures/seroprevalence_trajectory.png"><p class="description">Seroprevalence</p></div>
            </div>

            <h2>4. Parameter Estimation</h2>
            <div class="figure"><img src="PostCalibrationFigures/parameter_posteriors_kde.png"><p class="description">Parameter Posteriors</p></div>

            <h2>5. Scenario Analysis</h2>
            <div class="figure-container">
                <div class="figure"><img src="PostCalibrationFigures/scenario_impact_bars.png"><p class="description">Relative Impact</p></div>
                <div class="figure"><img src="PostCalibrationFigures/scenario_comparison.png"><p class="description">Absolute Comparison</p></div>
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
        
        self.plot_posterior_predictive_checks()
        self.plot_age_stratified_incidence()  # New function for age-specific facets
        self.plot_age_specific_severity_metrics_bar()
        self.plot_parameter_posteriors_kde()
        self.plot_reproduction_number_with_ci()
        self.plot_seroprevalence_trajectory()
        self.plot_scenario_trajectory_comparison()
        self.plot_scenario_summary_bars()
        
        self.generate_html_report()
        
        print(f"\n--- Python Analysis Complete! Figures saved to: {self.figures_dir} ---")

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

    parser.add_argument('--output-dir', type=str, default=str(default_output_dir), help='Base output directory')
    parser.add_argument('--start-date', type=str, default='2020-03-01', help='Simulation start date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    output_dir_to_analyze = Path(args.output_dir)

    print(f"--- Analysis Runner ---")
    print(f"Target: {output_dir_to_analyze.resolve()}")
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
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    main()