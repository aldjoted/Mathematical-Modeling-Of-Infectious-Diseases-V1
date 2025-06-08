"""
Post-calibration analysis and visualization script for SEPAIHRD model outputs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import argparse
from scipy import stats
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib') # Suppress tight_layout warning

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define age group labels globally
AGE_GROUPS = ['0-30', '30-60', '60-80', '80+']
# Define NPI periods for consistent plotting (dates from project description)
# (Day 0 = March 1, 2020)
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

    def _load_csv(self, *subpaths, check_time=True):
        filepath = self._get_filepath(*subpaths)
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                if check_time and 'time' in df.columns:
                    # Convert 'time' (days from start) to datetime for better x-axis
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
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=2)) # Ticks tous les 2 mois
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=1)) # Ticks mineurs tous les mois
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y')) # Format "Mois Année"
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    def plot_posterior_predictive_checks(self):
        """Plot daily and cumulative incidence with observed data and model uncertainty."""
        data_types = ["daily_hospitalizations", "daily_icu_admissions", "daily_deaths"]
        cumulative_types = ["cumulative_hospitalizations", "cumulative_icu_admissions", "cumulative_deaths"]
        
        for i, (dtype, cum_dtype) in enumerate(zip(data_types, cumulative_types)):
            fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # --- Daily Data ---
            median_df = self._load_csv("posterior_predictive", f"{dtype}_median.csv")
            lower95_df = self._load_csv("posterior_predictive", f"{dtype}_lower95.csv")
            upper95_df = self._load_csv("posterior_predictive", f"{dtype}_upper95.csv")
            observed_df = self._load_csv("posterior_predictive", f"{dtype}_observed.csv")

            ax = axes[0]
            if median_df is not None and lower95_df is not None and upper95_df is not None:
                # Sum across age groups for overall plot
                median_total = median_df[[col for col in median_df.columns if '_age' in col]].sum(axis=1)
                lower95_total = lower95_df[[col for col in lower95_df.columns if '_age' in col]].sum(axis=1)
                upper95_total = upper95_df[[col for col in upper95_df.columns if '_age' in col]].sum(axis=1)
                
                ax.plot(median_df['date'], median_total, label='Model Median', color='navy')
                ax.fill_between(median_df['date'], lower95_total, upper95_total, alpha=0.3, color='cornflowerblue', label='Model 95% CrI')

            if observed_df is not None:
                observed_total = observed_df[[col for col in observed_df.columns if '_age' in col]].sum(axis=1)
                ax.plot(observed_df['date'], observed_total, 'o-', label='Observed Data', color='red', markersize=3, alpha=0.7)

            ax.set_title(f'Posterior Predictive Check: Daily {dtype.replace("_", " ").title()}', fontsize=14)
            ax.set_ylabel('Count', fontsize=12)
            ax.legend()
            add_npi_shading(ax, periods=NPI_PERIODS_DEF)
            self._format_date_axis(ax) 

            # --- Cumulative Data ---
            median_cum_df = self._load_csv("posterior_predictive", f"{cum_dtype}_median.csv")
            lower95_cum_df = self._load_csv("posterior_predictive", f"{cum_dtype}_lower95.csv")
            upper95_cum_df = self._load_csv("posterior_predictive", f"{cum_dtype}_upper95.csv")
            observed_cum_df = self._load_csv("posterior_predictive", f"{cum_dtype}_observed.csv")

            ax = axes[1]
            if median_cum_df is not None and lower95_cum_df is not None and upper95_cum_df is not None:
                median_cum_total = median_cum_df[[col for col in median_cum_df.columns if '_age' in col]].sum(axis=1)
                lower95_cum_total = lower95_cum_df[[col for col in lower95_cum_df.columns if '_age' in col]].sum(axis=1)
                upper95_cum_total = upper95_cum_df[[col for col in upper95_cum_df.columns if '_age' in col]].sum(axis=1)

                ax.plot(median_cum_df['date'], median_cum_total, label='Model Median', color='darkgreen')
                ax.fill_between(median_cum_df['date'], lower95_cum_total, upper95_cum_total, alpha=0.3, color='lightgreen', label='Model 95% CrI')

            if observed_cum_df is not None:
                observed_cum_total = observed_cum_df[[col for col in observed_cum_df.columns if '_age' in col]].sum(axis=1)
                ax.plot(observed_cum_df['date'], observed_cum_total, 'o-', label='Observed Data', color='sienna', markersize=3, alpha=0.7)
            
            ax.set_title(f'Posterior Predictive Check: Cumulative {cum_dtype.replace("_", " ").title()}', fontsize=14)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Cumulative Count', fontsize=12)
            ax.legend()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.xticks(rotation=45)
            add_npi_shading(ax, periods=NPI_PERIODS_DEF)
            self._format_date_axis(ax)

            plt.tight_layout()
            plt.savefig(self.figures_dir / f"ppc_{dtype}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Plotted PPC for {dtype}")

    def plot_reproduction_number_with_ci(self):
        """Plot Rt with credible intervals if available."""
        df = self._load_csv("rt_trajectories", "Rt_aggregated_with_uncertainty.csv")
        if df is None:
            print("Aggregated Rt data not found, cannot plot with CI.")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        
        if 'median' in df.columns and 'q025' in df.columns and 'q975' in df.columns:
            ax.plot(df['date'], df['median'], linewidth=2, color='darkblue', label='Median Rt')
            ax.fill_between(df['date'], df['q025'], df['q975'], alpha=0.3, color='cornflowerblue', label='95% CrI')
        elif 'median' in df.columns: # Fallback if only median is present
             ax.plot(df['date'], df['median'], linewidth=2, color='darkblue', label='Median Rt')
        elif 'Rt' in df.columns: # Fallback for older single run format
            ax.plot(df['date'], df['Rt'], linewidth=2, color='darkblue', label='Rt')
        else:
            print("Required columns (median/q025/q975 or Rt) not in Rt_aggregated_with_uncertainty.csv")
            plt.close(fig)
            return

        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Rt = 1')
        
        custom_legend_handles = [plt.Line2D([0], [0], color='darkblue', lw=2),
                                 plt.Rectangle((0,0),1,1,fc='cornflowerblue', alpha=0.3),
                                 plt.Line2D([0], [0], color='red', linestyle='--', alpha=0.7)]
        custom_legend_labels = ['Median Rt', '95% CrI', 'Rt = 1']

        # Add NPI period shading and extend legend
        y_min, y_max = ax.get_ylim()
        current_max_y = 0
        if 'median' in df.columns: current_max_y = max(current_max_y, df['median'].max())
        if 'Rt' in df.columns: current_max_y = max(current_max_y, df['Rt'].max())

        ax.set_ylim(0, max(3.5, current_max_y * 1.1)) # Adjust y-limit dynamically

        for start, end, label, color, alpha_val in NPI_PERIODS_DEF:
            ax.axvspan(pd.to_datetime(self.start_date + pd.to_timedelta(start, unit='D')), 
                       pd.to_datetime(self.start_date + pd.to_timedelta(end, unit='D')), 
                       alpha=alpha_val, color=color, label=f"{label} ({start}-{end}d)")
            custom_legend_handles.append(plt.Rectangle((0,0),1,1,fc=color, alpha=alpha_val))
            custom_legend_labels.append(f"{label}")

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Effective Reproduction Number (Rt)', fontsize=12)
        ax.set_title('Time-Varying Rt with 95% Credible Interval', fontsize=14)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45)
        
        # Create a single legend for all items
        ax.legend(custom_legend_handles, custom_legend_labels, loc='upper right', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "Rt_aggregated_with_CI.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Plotted Rt with CI.")

    def plot_seroprevalence_trajectory(self):
        """Plot seroprevalence trajectory with CIs and ENE-COVID validation point."""
        df = self._load_csv("seroprevalence", "seroprevalence_trajectory.csv")
        if df is None:
            print("Seroprevalence trajectory data not found.")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        if 'median' in df.columns and 'lower_95' in df.columns and 'upper_95' in df.columns:
            ax.plot(df['date'], df['median'] * 100, linewidth=2, color='purple', label='Model Median Seroprevalence')
            ax.fill_between(df['date'], df['lower_95'] * 100, df['upper_95'] * 100, alpha=0.3, color='orchid', label='Model 95% CrI')
        else:
            print("Required columns for seroprevalence CI not found.")
            plt.close(fig)
            return
            
        # ENE-COVID point from PDF (around day 64 - May 4th, 2020)
        # Prevalence of SARS-CoV-2 in Spain (ENE-COVID): a nationwide, population-based seroepidemiological study
        # Overall prevalence for first round (April 27-May 11): 5.0% (95% CI 4.7–5.3)
        # For simplicity, let's use the data provided in the PDF abstract (4.6-5.0%, CI 4.3-5.4%) which might be for a slightly different cut.
        # We will use day 64 (May 4th) as the target.
        ene_covid_day = 64 
        ene_covid_date = self.start_date + pd.to_timedelta(ene_covid_day, unit='D')
        ene_covid_mean_pdf = (0.046 + 0.050) / 2 # From PDF abstract text
        ene_covid_lower_pdf = 0.043
        ene_covid_upper_pdf = 0.054

        if 'ene_covid_point' in df.columns and not df['ene_covid_point'].isnull().all():
            # Use data from CSV if available (more robust)
            ene_row = df[~df['ene_covid_point'].isnull()].iloc[0]
            ax.errorbar(pd.to_datetime(ene_row['date']), ene_row['ene_covid_point'] * 100, 
                        yerr=[[(ene_row['ene_covid_point'] - ene_row['ene_covid_lower'])*100], 
                              [(ene_row['ene_covid_upper'] - ene_row['ene_covid_point'])*100]],
                        fmt='o', color='black', capsize=5, markersize=8, label='ENE-COVID (early May 2020)')
        else: # Fallback to hardcoded values
             ax.errorbar(ene_covid_date, ene_covid_mean_pdf * 100, 
                        yerr=[[(ene_covid_mean_pdf - ene_covid_lower_pdf)*100], 
                              [(ene_covid_upper_pdf - ene_covid_mean_pdf)*100]],
                        fmt='o', color='black', capsize=5, markersize=8, label='ENE-COVID (early May 2020)')


        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Seroprevalence (%)', fontsize=12)
        ax.set_title('Modelled Seroprevalence Trajectory vs ENE-COVID Study', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45)
        add_npi_shading(ax, periods=NPI_PERIODS_DEF)
        self._format_date_axis(ax) 

        plt.tight_layout()
        plt.savefig(self.figures_dir / "seroprevalence_trajectory_validation.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Plotted seroprevalence trajectory with ENE-COVID validation.")

    def plot_hidden_compartments_aggregated(self):
        """Plot aggregated hidden compartment dynamics (E, P, A, I, R) with CIs."""
        compartments = ["E", "P", "A", "I", "R"]
        titles = ["Exposed", "Presymptomatic", "Asymptomatic", "Symptomatic (Active)", "Recovered"]
        
        num_plots = len(compartments)
        fig, axes = plt.subplots(int(np.ceil(num_plots / 2)), 2, figsize=(15, 4 * np.ceil(num_plots / 2)), sharex=True)
        axes = axes.flatten()

        for i, (comp, title) in enumerate(zip(compartments, titles)):
            median_df = self._load_csv("mcmc_aggregated", f"{comp}_hidden_dynamics_median.csv")
            q025_df = self._load_csv("mcmc_aggregated", f"{comp}_hidden_dynamics_q025.csv")
            q975_df = self._load_csv("mcmc_aggregated", f"{comp}_hidden_dynamics_q975.csv")

            ax = axes[i]
            if median_df is not None and q025_df is not None and q975_df is not None:
                # Sum across age groups for total
                median_total = median_df[[col for col in median_df.columns if f'{comp}_age' in col]].sum(axis=1)
                q025_total = q025_df[[col for col in q025_df.columns if f'{comp}_age' in col]].sum(axis=1)
                q975_total = q975_df[[col for col in q975_df.columns if f'{comp}_age' in col]].sum(axis=1)

                ax.plot(median_df['date'], median_total, label=f'Median {title}', color=sns.color_palette()[i])
                ax.fill_between(median_df['date'], q025_total, q975_total, alpha=0.3, color=sns.color_palette()[i], label='95% CrI')
            else:
                ax.text(0.5, 0.5, f"Data for {comp} not found", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


            ax.set_title(f'Total {title} Individuals', fontsize=12)
            ax.set_ylabel('Number of Individuals', fontsize=10)
            if i >= num_plots - 2 : # xlabel for bottom row
                ax.set_xlabel('Date', fontsize=10)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            ax.legend(fontsize=8)
            add_npi_shading(ax, periods=[p for p in NPI_PERIODS_DEF if p[2] not in ['Baseline (CoMix)']]) # Slightly different shading for hidden dynamics


        # Remove any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.suptitle('Aggregated Hidden Compartment Dynamics (Median and 95% CrI)', fontsize=16, y=1.02)
        plt.savefig(self.figures_dir / "hidden_compartments_aggregated_CI.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Plotted aggregated hidden compartments with CI.")

    def plot_parameter_posteriors_kde(self, burn_in_frac=0.2):
        """Plot KDE of key parameter posteriors."""
        samples_df = self._load_csv("parameter_posteriors", "posterior_samples.csv", check_time=False)
        if samples_df is None:
            print("Posterior samples not found.")
            return

        # Apply burn-in
        num_samples = len(samples_df)
        samples_to_use = samples_df.iloc[int(num_samples * burn_in_frac):]

        # Select key parameters to plot (adjust as needed)
        params_to_plot_kde = ['beta', 'theta', 'sigma', 'gamma_p', 'gamma_A', 'gamma_I']
        # Add kappa parameters if they exist, assuming format kappa_1, kappa_2 etc.
        kappa_cols = [col for col in samples_to_use.columns if col.startswith('kappa_')]
        params_to_plot_kde.extend(kappa_cols)
        
        # Also add some age-specific parameters if they exist (e.g. p_0, h_0)
        # For simplicity, let's pick the first age group for p, h, icu, d_H, d_ICU
        age_specific_bases = ['p', 'h', 'icu', 'd_H', 'd_ICU']
        for base in age_specific_bases:
            if f'{base}_0' in samples_to_use.columns:
                params_to_plot_kde.append(f'{base}_0')
        
        actual_params_to_plot = [p for p in params_to_plot_kde if p in samples_to_use.columns]
        if not actual_params_to_plot:
            print("None of the specified parameters for KDE plot are in the samples file.")
            return

        num_params = len(actual_params_to_plot)
        cols = 3
        rows = int(np.ceil(num_params / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = axes.flatten()

        for i, param_name in enumerate(actual_params_to_plot):
            ax = axes[i]
            sns.kdeplot(samples_to_use[param_name], ax=ax, fill=True, linewidth=2)
            
            median_val = samples_to_use[param_name].median()
            mean_val = samples_to_use[param_name].mean()
            q025 = samples_to_use[param_name].quantile(0.025)
            q975 = samples_to_use[param_name].quantile(0.975)

            ax.axvline(median_val, color='red', linestyle='--', label=f'Median: {median_val:.3g}')
            ax.axvline(q025, color='gray', linestyle=':', alpha=0.7)
            ax.axvline(q975, color='gray', linestyle=':', alpha=0.7)
            
            ax.set_title(f'Posterior: {param_name}', fontsize=10)
            ax.set_xlabel('Parameter Value', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.legend(fontsize=8)

        for j in range(i + 1, len(axes)): # Hide unused subplots
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.suptitle('Key Parameter Posterior Distributions (KDE)', fontsize=16, y=1.02)
        plt.savefig(self.figures_dir / "parameter_posteriors_kde.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Plotted KDEs for key parameter posteriors.")

    def plot_age_specific_severity_metrics_bar(self):
        """Plot IFR, IHR, IICUR by age group using aggregated MCMC summary."""
        summary_df = self._load_csv("mcmc_aggregated", "scalar_metrics_summary.csv", check_time=False)
        if summary_df is None:
            print("Aggregated MCMC scalar metrics summary not found.")
            return
        
        metrics_to_plot = {
            "IFR": ("Infection Fatality Rate", "darkred"),
            "IHR": ("Infection Hospitalization Rate", "darkblue"),
            "IICUR": ("Infection ICU Admission Rate", "darkgreen")
        }
        
        num_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5), sharey=False)
        if num_metrics == 1: axes = [axes] # Ensure axes is iterable

        for i, (metric_prefix, (metric_title, color)) in enumerate(metrics_to_plot.items()):
            ax = axes[i]
            medians = []
            lower_cis = []
            upper_cis = []
            
            for age_idx in range(len(AGE_GROUPS)):
                metric_key = f"{metric_prefix}_age_{age_idx}"
                row = summary_df[summary_df['metric'] == metric_key]
                if not row.empty:
                    medians.append(row['median'].iloc[0] * 100) # Convert to percentage
                    lower_cis.append((row['median'].iloc[0] - row['q025'].iloc[0]) * 100)
                    upper_cis.append((row['q975'].iloc[0] - row['median'].iloc[0]) * 100)
                else:
                    medians.append(0)
                    lower_cis.append(0)
                    upper_cis.append(0)
            
            error_bars = [lower_cis, upper_cis]
            ax.bar(AGE_GROUPS, medians, yerr=error_bars, color=color, alpha=0.7, capsize=5, error_kw={'alpha':0.5})
            ax.set_ylabel(f'{metric_prefix} (%)', fontsize=12)
            ax.set_title(metric_title, fontsize=14)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.figures_dir / "age_specific_severity_metrics_bar_CI.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Plotted age-specific severity metrics with CIs.")

    def plot_scenario_trajectory_comparison(self):
        """Plot key trajectories (e.g., cumulative deaths, Rt) across scenarios."""
        scenarios_to_plot = {
            "cumulative_deaths": ("Cumulative Deaths", "Total Deaths"),
            "rt": ("Effective Reproduction Number (Rt)", "Rt")
        }

        for key, (title_suffix, y_label) in scenarios_to_plot.items():
            df = self._load_csv("scenarios", f"{key}_trajectories.csv")
            if df is None:
                print(f"Scenario trajectory file for {key} not found.")
                continue
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            scenario_cols = [col for col in df.columns if col not in ['time', 'date']]
            
            for scen_col in scenario_cols:
                label = scen_col.replace("_deaths", "").replace("_rt", "").replace("_", " ").title()
                linestyle = '--' if 'baseline' not in scen_col.lower() else '-'
                linewidth = 1.5 if 'baseline' not in scen_col.lower() else 2.5
                alpha = 0.8 if 'baseline' not in scen_col.lower() else 1.0
                color_idx = scenario_cols.index(scen_col) % len(sns.color_palette("Paired"))

                ax.plot(df['date'], df[scen_col], label=label, linestyle=linestyle, linewidth=linewidth, alpha=alpha, color=sns.color_palette("Paired")[color_idx])

            ax.set_title(f'Scenario Comparison: {title_suffix}', fontsize=14)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.xticks(rotation=45)
            add_npi_shading(ax, periods=NPI_PERIODS_DEF)
            self._format_date_axis(ax) 

            plt.tight_layout()
            plt.savefig(self.figures_dir / f"scenario_trajectories_{key}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Plotted scenario trajectory comparison for {key}.")

    def plot_scenario_summary_bars(self):
        """Plot bar chart summarizing key scalar metrics from scenario comparison."""
        df = self._load_csv("scenarios", "scenario_comparison_summary.csv", check_time=False)
        if df is None:
            print("Scenario comparison summary file not found.")
            return

        metrics_to_plot = ['total_cumulative_deaths', 'peak_hospital_occupancy', 'overall_attack_rate']
        
        # Calculate percentage change from baseline
        baseline_row = df[df['scenario_name'] == 'baseline']
        if baseline_row.empty:
            print("Baseline scenario not found in summary.")
            return
        baseline_row = baseline_row.iloc[0]

        df_plot = df[df['scenario_name'] != 'baseline'].copy()
        
        for metric in metrics_to_plot:
            if metric in df_plot.columns and metric in baseline_row:
                df_plot[f'{metric}_pct_change'] = (df_plot[metric] - baseline_row[metric]) / baseline_row[metric] * 100
            else:
                 df_plot[f'{metric}_pct_change'] = np.nan


        num_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 5 * num_metrics), sharex=True)
        if num_metrics == 1: axes = [axes]

        scenarios_names = df_plot['scenario_name'].tolist()

        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            pct_change_col = f'{metric}_pct_change'
            if pct_change_col not in df_plot.columns or df_plot[pct_change_col].isnull().all():
                ax.text(0.5, 0.5, f"Data for {metric} not available", ha='center', va='center', transform=ax.transAxes)
                continue

            values = df_plot[pct_change_col].tolist()
            colors = ['forestgreen' if v < 0 else 'firebrick' for v in values]
            
            bars = ax.bar(scenarios_names, values, color=colors, alpha=0.8)
            ax.axhline(0, color='black', linewidth=0.8)
            
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}%', 
                        va='bottom' if yval >=0 else 'top', ha='center', fontsize=9)

            ax.set_ylabel('Change from Baseline (%)', fontsize=10)
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.xticks(rotation=45, ha="right")
        plt.suptitle('Scenario Analysis: Impact on Key Outcomes', fontsize=16, y=1.0)
        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle
        plt.savefig(self.figures_dir / "scenario_summary_bars.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Plotted scenario summary bars.")


    def generate_html_report(self):
        """Generate a comprehensive HTML report with all plots."""
        current_date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_items = [
            ("Posterior Predictive Checks: Daily Hospitalizations", "ppc_daily_hospitalizations.png"),
            ("Posterior Predictive Checks: Daily ICU Admissions", "ppc_daily_icu_admissions.png"),
            ("Posterior Predictive Checks: Daily Deaths", "ppc_daily_deaths.png"),
            ("Time-Varying Rt (Aggregated)", "Rt_aggregated_with_CI.png"),
            ("Seroprevalence Trajectory vs ENE-COVID", "seroprevalence_trajectory_validation.png"),
            ("Aggregated Hidden Compartment Dynamics", "hidden_compartments_aggregated_CI.png"),
            ("Key Parameter Posterior Distributions (KDE)", "parameter_posteriors_kde.png"),
            ("Age-Specific Severity Metrics (Bar)", "age_specific_severity_metrics_bar_CI.png"),
            ("Scenario Summary (Bars)", "scenario_summary_bars.png"),
            ("Scenario Trajectories: Cumulative Deaths", "scenario_trajectories_cumulative_deaths.png"),
            ("Scenario Trajectories: Rt", "scenario_trajectories_rt.png"),
        ]
        
        html_body_content = ""
        for i, (title, img_file) in enumerate(report_items):
            if (self.figures_dir / img_file).exists():
                html_body_content += f"""
                <h2>{i+1}. {title}</h2>
                <img src="python_figures/{img_file}" alt="{title}">
                """
            else:
                 html_body_content += f"""
                <h2>{i+1}. {title}</h2>
                <p style="color:red;">Image python_figures/{img_file} not found.</p>
                """


        html_full = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SEPAIHRD Model Analysis Report (Python)</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; line-height: 1.6; color: #333; }}
                .container {{ max-width: 1000px; margin: auto; background: #f9f9f9; padding: 20px; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; text-align: center; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 35px; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }}
                img {{ max-width: 100%; height: auto; margin: 15px 0; border: 1px solid #ddd; border-radius: 4px; padding: 5px; background: white; }}
                p {{ margin-bottom: 10px; }}
                .timestamp {{ text-align: center; font-style: italic; color: #7f8c8d; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>SEPAIHRD Model Post-Calibration Analysis Report</h1>
                <p class="timestamp">Generated: {current_date}</p>
                {html_body_content}
            </div>
        </body>
        </html>
        """
        
        report_path = self.output_dir_base / "python_analysis_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_full)
        
        print(f"Python analysis report generated: {report_path}")

    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("Starting Python post-calibration analysis...")
        
        self.plot_posterior_predictive_checks()
        self.plot_reproduction_number_with_ci()
        self.plot_seroprevalence_trajectory()
        self.plot_hidden_compartments_aggregated()
        self.plot_parameter_posteriors_kde()
        self.plot_age_specific_severity_metrics_bar()
        self.plot_scenario_trajectory_comparison()
        self.plot_scenario_summary_bars()
        
        self.generate_html_report()
        
        print("Python analysis complete! Report and figures are in:", self.output_dir_base)

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

    print(f"--- Starting Analysis ---")
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
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    main()