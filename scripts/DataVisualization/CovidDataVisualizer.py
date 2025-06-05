"""Visualizes COVID-19 data, focusing on trends and comparisons.

This module provides the `CovidDataVisualizer` class, which is responsible
for loading processed COVID-19 data, generating various types of plots
(time series, grouped time series, stacked area charts), and saving them
as image files. It includes functionalities for calculating rates,
growth rates, and handling age-stratified data.
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns
import os
from pathlib import Path

# Simplified FileUtils for standalone execution.
# This class is used here to avoid dependency on a larger utils
# module when this script might be run in a more isolated environment.
class FileUtils:
    """A utility class for basic file system operations."""

    def ensure_directory_exists(self, path: str) -> bool:
        """Ensures that a directory exists, creating it if necessary.

        Args:
            path: The path to the directory.

        Returns:
            True if the directory exists or was created successfully, False otherwise.
        """
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"Error creating directory {path}: {e}")
            return False

# --- Image Settings ---
# Apply a global theme for consistent plot aesthetics.
sns.set_theme(style="whitegrid")

# Define standard font sizes for various plot elements.
FONTSIZE_TITLE = 18
FONTSIZE_LABEL = 14
FONTSIZE_TICKS = 12
FONTSIZE_LEGEND = 12
FONTSIZE_LEGEND_TITLE = 13

# Update matplotlib's runtime configuration for plots.
plt.rcParams.update({
    'font.size': FONTSIZE_TICKS,  # Default font size for text elements
    'axes.titlesize': FONTSIZE_TITLE,  # Font size for axes titles
    'axes.labelsize': FONTSIZE_LABEL,  # Font size for x and y labels
    'xtick.labelsize': FONTSIZE_TICKS,  # Font size for x-axis tick labels
    'ytick.labelsize': FONTSIZE_TICKS,  # Font size for y-axis tick labels
    'legend.fontsize': FONTSIZE_LEGEND,  # Font size for legend text
    'legend.title_fontsize': FONTSIZE_LEGEND_TITLE,  # Font size for legend title
    'figure.titlesize': FONTSIZE_TITLE + 2,  # Font size for the figure's suptitle
    'figure.figsize': (12, 7),  # Default figure size
    'lines.linewidth': 2.0,  # Default line width for plot lines
    'lines.markersize': 6,  # Default marker size
    'savefig.dpi': 300,  # Resolution for saved figures
    'savefig.bbox': 'tight',  # Adjust plot to fit tightly in the saved image
    'axes.titleweight': 'bold',  # Weight of the axes title font
    'axes.labelweight': 'normal',  # Weight of the axes label font
    'legend.frameon': False,  # Remove the frame around the legend for a cleaner look
})


class CovidDataVisualizer:
    """Visualizes COVID-19 data with a focus on clarity and consistency.

    This class handles the generation of various plots from COVID-19 data,
    including time series of absolute numbers, rates, and growth rates.
    It supports age-stratified visualizations and provides options for
    rolling averages. Plots are saved as PNG files by default.

    Attributes:
        age_group_suffixes (list[str]): Suffixes used for age-group-specific columns.
        age_group_labels (list[str]): Display labels for age groups.
        palette (sns.color_palette): Default color palette for plots.
        age_palette (sns.color_palette): Color palette for age-stratified plots.
        LABEL_DATE_AXIS (str): Standard label for the date axis.
        LABEL_AGE_GROUP (str): Standard label for age group legends/titles.
        LABEL_CASES (str): Standard label for case counts.
        LABEL_DEATHS (str): Standard label for death counts.
        LABEL_HOSPITALIZATIONS (str): Standard label for hospitalization counts.
        LABEL_ICU_PATIENTS (str): Standard label for ICU patient counts.
        LABEL_CFR (str): Standard label for Case Fatality Rate.
        LABEL_GROWTH_RATE (str): Standard label for growth rates.
        LABEL_HOSP_RATE (str): Standard label for Hospitalization Rate.
        LABEL_ICU_RATE (str): Standard label for ICU Admission Rate.
        file_utils (FileUtils): An instance of FileUtils for directory operations.
        default_output_dir (str): The default directory to save visualizations.
        fontsize_title (int): Font size for plot titles.
        fontsize_label (int): Font size for axis labels.
        fontsize_ticks (int): Font size for tick labels.
        fontsize_legend (int): Font size for legend text.
        fontsize_legend_title (int): Font size for legend titles.
    """

    def __init__(self, default_output_dir="visualizations_high_quality"):
        """Initializes the CovidDataVisualizer with output settings.

        Args:
            default_output_dir (str, optional): The directory where plots will be
                saved by default. Defaults to "visualizations_high_quality".
        """
        self.age_group_suffixes = ["0_30", "30_60", "60_80", "80_plus"]
        self.age_group_labels = ["0-30 years", "30-60 years", "60-80 years", "80+ years"]
        self.palette = sns.color_palette("tab10", 10)
        self.age_palette = sns.color_palette("viridis", len(self.age_group_labels))

        # Constants for plot labels to ensure consistency.
        self.LABEL_DATE_AXIS = "Date"
        self.LABEL_AGE_GROUP = "Age Group"
        self.LABEL_CASES = "Number of Cases"
        self.LABEL_DEATHS = "Number of Deaths"
        self.LABEL_HOSPITALIZATIONS = "Number of Hospitalizations"
        self.LABEL_ICU_PATIENTS = "Number of ICU Patients"
        self.LABEL_CFR = "Case Fatality Rate (%)"
        self.LABEL_GROWTH_RATE = "Growth Rate (%)"
        self.LABEL_HOSP_RATE = "Hospitalization Rate (%)"
        self.LABEL_ICU_RATE = "ICU Admission Rate (%)"

        self.file_utils = FileUtils()
        self.default_output_dir = default_output_dir
        self.file_utils.ensure_directory_exists(self.default_output_dir)
        
        # Store font sizes for direct access if needed, though rcParams is primary.
        self.fontsize_title = FONTSIZE_TITLE
        self.fontsize_label = FONTSIZE_LABEL
        self.fontsize_ticks = FONTSIZE_TICKS
        self.fontsize_legend = FONTSIZE_LEGEND
        self.fontsize_legend_title = FONTSIZE_LEGEND_TITLE


    def _safe_divide(self, numerator_series: pd.Series, denominator_series: pd.Series, scale: float = 1.0) -> pd.Series:
        """Performs division, handling division by zero by returning NaN.

        Args:
            numerator_series (pd.Series): The numerator.
            denominator_series (pd.Series): The denominator.
            scale (float, optional): A scaling factor for the result. Defaults to 1.0.

        Returns:
            pd.Series: The result of the division, or NaN where division by zero occurred.
        """
        denominator_safe = denominator_series.replace(0, np.nan)
        return (numerator_series / denominator_safe * scale)

    def load_data(self, filename: str) -> pd.DataFrame | None:
        """Loads and preprocesses data from a CSV file.

        The 'date' column is parsed and set as the DataFrame index.
        Numeric columns are attempted to be converted to numeric types.

        Args:
            filename (str): The path to the CSV file.

        Returns:
            pd.DataFrame | None: The loaded DataFrame with 'date' as index,
                                 or None if loading fails.
        """
        try:
            df = pd.read_csv(filename, parse_dates=['date'])
            if 'date' not in df.columns:
                print("Error: 'date' column not found in CSV.")
                return None
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            print(f"Data loaded successfully. {len(df)} data points from {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}.")
            # Attempt to convert object columns to numeric where appropriate.
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except ValueError:
                        # If conversion fails, keep the column as is and warn the user.
                        print(f"Warning: Column '{col}' could not be converted to numeric and remains type {df[col].dtype}.")
                        pass 
            return df
        except Exception as e:
            print(f"Error loading data from {filename}: {e}")
            return None

    def _apply_date_axis_formatting(self, ax: plt.Axes):
        """Applies standard date formatting to the x-axis of a plot.

        Args:
            ax (plt.Axes): The matplotlib Axes object to format.
        """
        ax.set_xlabel(self.LABEL_DATE_AXIS)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        plt.setp(ax.get_xticklabels(), rotation=0, ha='right')

    def _save_plot(self, fig: plt.Figure, filename_base: str, output_dir: str | None = None, formats: list | None = None):
        """Saves a matplotlib Figure to one or more file formats.

        Ensures the output directory exists before saving. Defaults to PNG format.

        Args:
            fig (plt.Figure): The matplotlib Figure to save.
            filename_base (str): The base name for the output file (without extension).
            output_dir (str, optional): The directory to save the plot.
                Defaults to `self.default_output_dir`.
            formats (list, optional): A list of file format extensions (e.g., ['png', 'pdf']).
                Defaults to ['png'].
        """
        if output_dir is None:
            output_dir = self.default_output_dir
        self.file_utils.ensure_directory_exists(output_dir)
        
        if formats is None:
            formats = ['png']  # Default to PNG as per class docstring

        for fmt in formats:
            path = Path(output_dir) / f"{filename_base}.{fmt}"
            fig.savefig(path)  # bbox_inches and dpi are handled by rcParams
            print(f"Saved plot: {path}")
        plt.close(fig)  # Close the figure to free up memory

    def plot_time_series(self, df: pd.DataFrame, y_col: str, title: str, ylabel: str,
                         filename_base: str, output_dir: str | None = None,
                         color: str | None = None, rolling_window: int | None = None,
                         secondary_y_col: str | None = None, secondary_ylabel: str | None = None,
                         secondary_color: str | None = None,
                         h_line: float | None = None, h_line_label: str | None = None):
        """Plots a single time series, optionally with a rolling average and a secondary y-axis.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            y_col (str): Column name for the primary y-axis data.
            title (str): Title of the plot.
            ylabel (str): Label for the primary y-axis.
            filename_base (str): Base filename for the saved plot.
            output_dir (str, optional): Directory to save the plot. Defaults to `self.default_output_dir`.
            color (str, optional): Color for the primary time series. Defaults to palette color.
            rolling_window (int, optional): Window size for rolling average. Defaults to None.
            secondary_y_col (str, optional): Column name for the secondary y-axis data. Defaults to None.
            secondary_ylabel (str, optional): Label for the secondary y-axis. Defaults to None.
            secondary_color (str, optional): Color for the secondary time series. Defaults to palette color.
            h_line (float, optional): Value for a horizontal reference line. Defaults to None.
            h_line_label (str, optional): Label for the horizontal reference line. Defaults to None.
        """
        if y_col not in df.columns:
            print(f"Warning: Column '{y_col}' not found for plot '{title}'. Skipping.")
            return
        if secondary_y_col and secondary_y_col not in df.columns:
            print(f"Warning: Secondary column '{secondary_y_col}' not found for plot '{title}'. Skipping secondary axis.")
            secondary_y_col = None

        fig, ax1 = plt.subplots()

        data_to_plot = df[y_col].copy()
        current_title = title
        if rolling_window:
            data_to_plot = data_to_plot.rolling(window=rolling_window, center=True, min_periods=1).mean()
            current_title += f" ({rolling_window}-day Rolling Avg)"

        sns.lineplot(x=data_to_plot.index, y=data_to_plot.values, ax=ax1, color=color or self.palette[0], label=y_col)
        ax1.set_ylabel(ylabel, color=color or self.palette[0])
        ax1.tick_params(axis='y', labelcolor=color or self.palette[0])

        if secondary_y_col:
            ax2 = ax1.twinx()
            secondary_data_to_plot = df[secondary_y_col].copy()
            if rolling_window:
                secondary_data_to_plot = secondary_data_to_plot.rolling(window=rolling_window, center=True, min_periods=1).mean()
            
            sns.lineplot(x=secondary_data_to_plot.index, y=secondary_data_to_plot.values, ax=ax2, color=secondary_color or self.palette[1],
                         label=secondary_y_col, linestyle='--')
            ax2.set_ylabel(secondary_ylabel, color=secondary_color or self.palette[1])
            ax2.tick_params(axis='y', labelcolor=secondary_color or self.palette[1])
            
            # Combine legends from both axes
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(0.01, 0.99))
        else:
            ax1.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99))
            
        if h_line is not None:
            ax1.axhline(h_line, color='grey', linestyle=':', linewidth=1.5, label=h_line_label or f'Reference: {h_line}')
            # Re-generate legend to include h_line if it was added
            handles_all, labels_all = ax1.get_legend_handles_labels()
            if secondary_y_col and 'ax2' in locals(): 
                lines2, labels2 = ax2.get_legend_handles_labels()
                handles_all.extend(lines2) # Ensure secondary axis lines are also in legend
                labels_all.extend(labels2)
            ax1.legend(handles_all, labels_all, loc='upper left', bbox_to_anchor=(0.01, 0.99))


        ax1.set_title(current_title)
        plt.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.7)
        self._apply_date_axis_formatting(ax1)
        self._save_plot(fig, filename_base, output_dir)

    def plot_grouped_time_series(self, df: pd.DataFrame,
                                 cols_to_plot: list[str], 
                                 legend_labels: list[str],
                                 title: str, ylabel: str, filename_base: str,
                                 output_dir: str | None = None,
                                 rolling_window: int | None = None,
                                 palette: list[str] | None = None):
        """Plots multiple time series on the same axes, typically for comparison.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            cols_to_plot (list[str]): List of column names to plot.
            legend_labels (list[str]): List of labels for the legend, corresponding to `cols_to_plot`.
            title (str): Title of the plot.
            ylabel (str): Label for the y-axis.
            filename_base (str): Base filename for the saved plot.
            output_dir (str, optional): Directory to save the plot. Defaults to `self.default_output_dir`.
            rolling_window (int, optional): Window size for rolling average. Defaults to None.
            palette (list[str], optional): Color palette for the lines. Defaults to class palettes.
        """
        plot_data_dict = {}
        current_title = title
        
        valid_cols_to_plot = []
        valid_legend_labels = []

        for i, col_name in enumerate(cols_to_plot):
            if col_name not in df.columns:
                print(f"Warning: Column '{col_name}' not found in DataFrame. Skipping for plot '{title}'.")
                continue
            valid_cols_to_plot.append(col_name)
            valid_legend_labels.append(legend_labels[i])
            series = df[col_name].copy()
            if rolling_window:
                series = series.rolling(window=rolling_window, center=True, min_periods=1).mean()
            plot_data_dict[legend_labels[i]] = series # Use legend label as key for DataFrame
            
        if not plot_data_dict or not valid_cols_to_plot:
            # Create a plot indicating no data if all specified columns are missing or result in no data
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data to plot (columns missing or all NaN)", fontsize=14, ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{title} (No Data)")
            self._apply_date_axis_formatting(ax)
            self._save_plot(fig, filename_base + "_no_data", output_dir)
            return
            
        plot_df = pd.DataFrame(plot_data_dict, index=df.index) 
        
        if plot_df.isnull().all().all():
            # Create a plot indicating no data if all values are NaN after processing
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data to plot (all values are NaN after processing)", fontsize=14, ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{title} (No Data)")
            self._apply_date_axis_formatting(ax)
            self._save_plot(fig, filename_base + "_nan_data", output_dir)
            return

        if rolling_window:
            current_title += f" ({rolling_window}-day Rolling Avg)"
        
        fig, ax = plt.subplots()
        # Determine palette based on title content or use provided/default.
        current_palette = palette or (self.age_palette if ("Age Group" in title or "Age-Stratified" in title) else self.palette)
        
        for i, col_label in enumerate(valid_legend_labels): # Iterate using the validated legend labels
            sns.lineplot(x=plot_df.index, y=plot_df[col_label], ax=ax, label=col_label, 
                         color=current_palette[i % len(current_palette)])

        ax.set_title(current_title)
        ax.set_ylabel(ylabel)
        ax.legend(title=(self.LABEL_AGE_GROUP if ("Age Group" in title or "Age-Stratified" in title) else "Metric"), 
                  loc='upper left', bbox_to_anchor=(0.01, 0.99))

        plt.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.7)
        self._apply_date_axis_formatting(ax)
        self._save_plot(fig, filename_base, output_dir)

    def plot_stacked_area_chart(self, df: pd.DataFrame,
                                cols_to_plot: list[str], legend_labels: list[str],
                                title: str, ylabel: str, filename_base: str,
                                output_dir: str | None = None,
                                rolling_window: int | None = None,
                                palette: list[str] | None = None):
        """Plots a stacked area chart for multiple time series.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            cols_to_plot (list[str]): List of column names to plot.
            legend_labels (list[str]): List of labels for the legend.
            title (str): Title of the plot.
            ylabel (str): Label for the y-axis.
            filename_base (str): Base filename for the saved plot.
            output_dir (str, optional): Directory to save the plot. Defaults to `self.default_output_dir`.
            rolling_window (int, optional): Window size for rolling average. Defaults to None.
            palette (list[str], optional): Color palette for the areas. Defaults to class palettes.
        """
        plot_data_dict = {}
        current_title = title
        
        valid_cols_to_plot = []
        valid_legend_labels = []

        for i, col_name in enumerate(cols_to_plot):
            if col_name not in df.columns:
                print(f"Warning: Column '{col_name}' not found. Skipping for '{title}'.")
                continue
            valid_cols_to_plot.append(col_name)
            valid_legend_labels.append(legend_labels[i])
            series = df[col_name].copy()
            if rolling_window:
                series = series.rolling(window=rolling_window, center=True, min_periods=1).mean().fillna(0) # Fill NaN for stacking
            else:
                series = series.fillna(0) # Fill NaN for stacking
            plot_data_dict[legend_labels[i]] = series

        if not plot_data_dict or not valid_cols_to_plot:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data to plot (columns missing or all NaN)", fontsize=14, ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{title} (No Data)")
            self._apply_date_axis_formatting(ax)
            self._save_plot(fig, filename_base + "_no_data", output_dir)
            return

        plot_df = pd.DataFrame(plot_data_dict, index=df.index)

        # Check if all data is zero or NaN, which would make a stacked plot meaningless or empty.
        if plot_df.isnull().all().all() or (plot_df.sum().sum() == 0 and not plot_df.empty): 
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data available for this plot (all values zero or NaN)", fontsize=14, ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{title} (No Data)")
            self._apply_date_axis_formatting(ax)
            self._save_plot(fig, filename_base + "_empty_data", output_dir)
            return

        if rolling_window:
            current_title += f" ({rolling_window}-day Rolling Avg)"

        fig, ax = plt.subplots()
        current_palette = palette or (self.age_palette if ("Age Group" in title or "Age-Stratified" in title) else self.palette)

        # Use the DataFrame's plot.area method for stacking.
        plot_df.plot.area(ax=ax, stacked=True, alpha=0.85,
                            color=current_palette[:len(plot_df.columns)], # Ensure palette matches number of columns
                            linewidth=0.5)
        
        ax.set_title(current_title)
        ax.set_ylabel(ylabel)
        ax.legend(title=(self.LABEL_AGE_GROUP if ("Age Group" in title or "Age-Stratified" in title) else "Metric"), 
                  labels=valid_legend_labels, loc='upper left', bbox_to_anchor=(0.01, 0.99))
        plt.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.7) # Grid only on y-axis for area plots
        self._apply_date_axis_formatting(ax)
        self._save_plot(fig, filename_base, output_dir)

    def add_rate_columns(self, df: pd.DataFrame, rate_definitions: dict) -> pd.DataFrame:
        """Calculates and adds new rate columns to the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            rate_definitions (dict): A dictionary defining the rates to calculate.
                Example:
                {
                    "cfr_overall": {"num": "deceased", "den": "confirmed", "scale": 100},
                    "cfr_by_age": {"num": "deceased", "den": "confirmed", "scale": 100, "age_stratified": True}
                }

        Returns:
            pd.DataFrame: The DataFrame with added rate columns.
        """
        df_out = df.copy()
        for rate_name_base, params in rate_definitions.items():
            num_col_base, den_col_base = params["num"], params["den"]
            scale = params.get("scale", 1.0)
            age_stratified = params.get("age_stratified", False)

            if age_stratified:
                for suffix in self.age_group_suffixes:
                    num_col, den_col = f"{num_col_base}_{suffix}", f"{den_col_base}_{suffix}"
                    out_col = f"{rate_name_base}_{suffix}"
                    if num_col in df_out and den_col in df_out:
                        df_out[out_col] = self._safe_divide(df_out[num_col], df_out[den_col], scale).fillna(np.nan)
                    else:
                        df_out[out_col] = np.nan # Ensure column exists even if data is missing
            else:
                if num_col_base in df_out and den_col_base in df_out:
                    df_out[rate_name_base] = self._safe_divide(df_out[num_col_base], df_out[den_col_base], scale).fillna(np.nan)
                else:
                    df_out[rate_name_base] = np.nan # Ensure column exists
        return df_out
        
    def add_growth_rate_columns(self, df: pd.DataFrame, metric_col_base: str, period: int = 7,
                                age_stratified: bool = False) -> pd.DataFrame:
        """Calculates and adds growth rate columns for specified metrics.

        Growth rate is calculated as percentage change over a defined period.

        Args:
            df (pd.DataFrame): The input DataFrame.
            metric_col_base (str): The base name of the metric column (e.g., "new_confirmed").
            period (int, optional): The period (in days) over which to calculate growth. Defaults to 7.
            age_stratified (bool, optional): Whether to calculate for age-stratified versions
                of the metric. Defaults to False.

        Returns:
            pd.DataFrame: The DataFrame with added growth rate columns.
        """
        df_out = df.copy()
        
        def calculate_and_add_growth(series: pd.Series, output_col_name_base: str):
            """Helper to calculate growth and add to df_out."""
            # Calculate percentage change over the period.
            growth_series = series.pct_change(periods=period) * 100
            # Replace infinite values (from division by zero if previous value was 0) with NaN.
            df_out[f"{output_col_name_base}_growth_rate_{period}d"] = growth_series.replace([np.inf, -np.inf], np.nan)

        if age_stratified:
            for suffix in self.age_group_suffixes:
                metric_col = f"{metric_col_base}_{suffix}"
                if metric_col in df.columns: calculate_and_add_growth(df[metric_col], metric_col)
        else: 
            if metric_col_base in df.columns: calculate_and_add_growth(df[metric_col_base], metric_col_base)
        return df_out
        
    def run_all_visualizations(self, input_file: str, output_dir: str | None = None) -> bool:
        """Generates a comprehensive set of COVID-19 visualizations.

        This method loads data, calculates necessary metrics (rates, growth rates),
        and then calls various plotting methods to create and save visualizations
        for overall trends, age-stratified data, and a summary dashboard.

        Args:
            input_file (str): Path to the processed CSV data file.
            output_dir (str, optional): Directory to save the visualizations.
                Defaults to `self.default_output_dir`.

        Returns:
            bool: True if visualizations were generated successfully, False otherwise.
        """
        df_orig = self.load_data(input_file)
        if df_orig is None or df_orig.index.empty:
            print("Failed to load data. Aborting visualizations.")
            return False
        
        output_dir = output_dir or self.default_output_dir
        df = df_orig.copy() 
        # Define standard rolling window sizes for consistency.
        rolling_short, rolling_medium, rolling_long = 7, 14, 28

        print("\n--- Calculating Rate Metrics ---")
        rate_defs = {
            "cfr_overall_cumulative": {"num": "cumulative_deceased", "den": "cumulative_confirmed", "scale": 100},
            "cfr_by_age_cumulative": {"num": "cumulative_deceased", "den": "cumulative_confirmed", "scale": 100, "age_stratified": True},
            "hosp_rate_vs_confirmed_overall": {"num": "new_hospitalized_patients", "den": "new_confirmed", "scale": 100},
            "icu_rate_vs_hospitalized_overall": {"num": "new_intensive_care_patients", "den": "new_hospitalized_patients", "scale": 100},
            "hosp_rate_vs_confirmed_by_age": {"num": "new_hospitalized_patients", "den": "new_confirmed", "scale": 100, "age_stratified": True},
            "icu_rate_vs_hospitalized_by_age": {"num": "new_intensive_care_patients", "den": "new_hospitalized_patients", "scale": 100, "age_stratified": True},
        }
        df = self.add_rate_columns(df, rate_defs)

        print("\n--- Calculating Growth Rate Metrics ---")
        growth_period = 7 # Standard period for growth rate calculation
        base_metrics_for_growth = ["new_confirmed", "new_deceased", "new_hospitalized_patients", "new_intensive_care_patients"]
        for metric in base_metrics_for_growth:
            df = self.add_growth_rate_columns(df, metric, period=growth_period)
            df = self.add_growth_rate_columns(df, metric, period=growth_period, age_stratified=True)

        print("\n--- Generating High Quality Plots (No Per Capita) ---")

        # Plotting overall metrics (new cases, deaths, etc.)
        self.plot_time_series(df, 'new_confirmed', 'New Confirmed Cases', self.LABEL_CASES,
                              'ts_new_confirmed_rolled', output_dir, color=self.palette[0], rolling_window=rolling_short)
        self.plot_time_series(df, 'new_deceased', 'New Deceased', self.LABEL_DEATHS,
                              'ts_new_deceased_rolled', output_dir, color=self.palette[1], rolling_window=rolling_short)
        self.plot_time_series(df, 'new_hospitalized_patients', 'New Hospitalized Patients', self.LABEL_HOSPITALIZATIONS,
                              'ts_new_hospitalized_rolled', output_dir, color=self.palette[2], rolling_window=rolling_short)
        self.plot_time_series(df, 'new_intensive_care_patients', 'New ICU Patients', self.LABEL_ICU_PATIENTS,
                              'ts_new_icu_rolled', output_dir, color=self.palette[3], rolling_window=rolling_short)

        # Plotting overall cumulative metrics
        self.plot_time_series(df, 'cumulative_confirmed', 'Cumulative Confirmed Cases', self.LABEL_CASES,
                              'ts_cumulative_confirmed', output_dir, color=self.palette[0])
        self.plot_time_series(df, 'cumulative_deceased', 'Cumulative Deceased', self.LABEL_DEATHS,
                              'ts_cumulative_deceased', output_dir, color=self.palette[1])
        self.plot_time_series(df, 'cumulative_hospitalized_patients', 'Cumulative Hospitalized Patients', self.LABEL_HOSPITALIZATIONS,
                              'ts_cumulative_hospitalized', output_dir, color=self.palette[2])
        self.plot_time_series(df, 'cumulative_intensive_care_patients', 'Cumulative ICU Patients', self.LABEL_ICU_PATIENTS,
                              'ts_cumulative_icu', output_dir, color=self.palette[3])

        # Plotting age-stratified new metrics (grouped and stacked)
        age_metrics_new = [
            ("new_confirmed", self.LABEL_CASES, "New Confirmed Cases"),
            ("new_deceased", self.LABEL_DEATHS, "New Deceased"),
            ("new_hospitalized_patients", self.LABEL_HOSPITALIZATIONS, "New Hospitalized Patients"),
            ("new_intensive_care_patients", self.LABEL_ICU_PATIENTS, "New ICU Patients")
        ]
        for base, label, title_prefix in age_metrics_new:
            cols = [f"{base}_{s}" for s in self.age_group_suffixes]
            self.plot_grouped_time_series(df, cols, self.age_group_labels, f'{title_prefix} by Age Group', label,
                                          f'grouped_{base}_by_age_abs_rolled', output_dir, rolling_window=rolling_short)
            self.plot_stacked_area_chart(df, cols, self.age_group_labels, f'Stacked {title_prefix} by Age Group', label,
                                         f'stacked_{base}_by_age_rolled', output_dir, rolling_window=rolling_short)

        # Plotting age-stratified cumulative metrics (grouped and stacked)
        age_metrics_cumulative = [
            ("cumulative_confirmed", self.LABEL_CASES, "Cumulative Confirmed Cases"),
            ("cumulative_deceased", self.LABEL_DEATHS, "Cumulative Deceased"),
            ("cumulative_hospitalized_patients", self.LABEL_HOSPITALIZATIONS, "Cumulative Hospitalized Patients"),
            ("cumulative_intensive_care_patients", self.LABEL_ICU_PATIENTS, "Cumulative ICU Patients")
        ]
        for base, label, title_prefix in age_metrics_cumulative:
            cols = [f"{base}_{s}" for s in self.age_group_suffixes]
            self.plot_grouped_time_series(df, cols, self.age_group_labels, f'{title_prefix} by Age Group', label,
                                          f'grouped_{base}_by_age', output_dir) # No rolling for cumulative totals
            self.plot_stacked_area_chart(df, cols, self.age_group_labels, f'Stacked {title_prefix} by Age Group', label,
                                         f'stacked_{base}_by_age', output_dir) # No rolling for cumulative totals

        # Plotting various rates (CFR, hospitalization rate, ICU rate)
        self.plot_time_series(df, 'cfr_overall_cumulative', 'Overall Case Fatality Rate (Cumulative)', self.LABEL_CFR,
                              'ts_cfr_overall_cumulative_rolled', output_dir, color=self.palette[0], rolling_window=rolling_long)
        cfr_age_cols = [f"cfr_by_age_cumulative_{s}" for s in self.age_group_suffixes]
        self.plot_grouped_time_series(df, cfr_age_cols, self.age_group_labels, 'Age-Stratified Case Fatality Rate (Cumulative)', self.LABEL_CFR,
                                      'grouped_cfr_by_age_cumulative_rolled', output_dir, rolling_window=rolling_long)

        self.plot_time_series(df, 'hosp_rate_vs_confirmed_overall', 'Overall Hospitalization Rate (vs. Confirmed)', self.LABEL_HOSP_RATE,
                              'ts_hosp_rate_overall_rolled', output_dir, color=self.palette[1], rolling_window=rolling_medium)
        hosp_rate_age_cols = [f"hosp_rate_vs_confirmed_by_age_{s}" for s in self.age_group_suffixes]
        self.plot_grouped_time_series(df, hosp_rate_age_cols, self.age_group_labels, 'Age-Stratified Hospitalization Rate (vs. Confirmed)', self.LABEL_HOSP_RATE,
                                      'grouped_hosp_rate_by_age_rolled', output_dir, rolling_window=rolling_medium)

        self.plot_time_series(df, 'icu_rate_vs_hospitalized_overall', 'Overall ICU Admission Rate (vs. Hospitalized)', self.LABEL_ICU_RATE,
                              'ts_icu_rate_overall_rolled', output_dir, color=self.palette[2], rolling_window=rolling_medium)
        icu_rate_age_cols = [f"icu_rate_vs_hospitalized_by_age_{s}" for s in self.age_group_suffixes]
        self.plot_grouped_time_series(df, icu_rate_age_cols, self.age_group_labels, 'Age-Stratified ICU Rate (vs. Hospitalized)', self.LABEL_ICU_RATE,
                                      'grouped_icu_rate_by_age_rolled', output_dir, rolling_window=rolling_medium)

        # Define metrics for growth rate plots
        overall_growth_metrics = [ # (base_col, label_col_unused_here, title_prefix)
            ("new_confirmed", self.LABEL_CASES, "New Confirmed Cases"),
            ("new_deceased", self.LABEL_DEATHS, "New Deceased"),
            ("new_hospitalized_patients", self.LABEL_HOSPITALIZATIONS, "New Hospitalized Patients"),
            ("new_intensive_care_patients", self.LABEL_ICU_PATIENTS, "New ICU Patients")
        ]
        age_growth_metrics = [ # (base_col, title_prefix)
            ("new_confirmed", "New Confirmed Cases"),
            ("new_deceased", "New Deceased"),
            ("new_hospitalized_patients", "New Hospitalized Patients"),
            ("new_intensive_care_patients", "New ICU Patients")
        ]

        # Plotting overall growth rates
        for i, (base, _, title_prefix) in enumerate(overall_growth_metrics):
            col_name = f"{base}_growth_rate_{growth_period}d"
            self.plot_time_series(df, col_name, f'{growth_period}-day Growth Rate of {title_prefix}', self.LABEL_GROWTH_RATE, 
                                  f'ts_{base}_growth_rate_rolled', output_dir,
                                  color=self.palette[i % len(self.palette)], rolling_window=rolling_short, h_line=0, h_line_label="Stable")

        # Plotting age-stratified growth rates
        for base, title_prefix in age_growth_metrics:
            cols = [f"{base}_{s}_growth_rate_{growth_period}d" for s in self.age_group_suffixes]
            self.plot_grouped_time_series(df, cols, self.age_group_labels, f'{growth_period}-day Growth Rate of {title_prefix} by Age Group', self.LABEL_GROWTH_RATE,
                                          f'grouped_{base}_growth_by_age_rolled', output_dir, rolling_window=rolling_short)
        
        # Create a dashboard with key metrics
        fig_dash, axes = plt.subplots(2, 2, figsize=(15, 10)) # Adjusted figsize for better layout
        fig_dash.suptitle(f'COVID-19 Key Metrics ({rolling_short}-day Rolling Averages)', fontsize=FONTSIZE_TITLE + 2)
        
        metrics_for_dashboard = [
            ('new_confirmed', axes[0,0], 'New Confirmed Cases', self.palette[0]), # Using specific label for clarity
            ('new_deceased', axes[0,1], 'New Deceased', self.palette[1]),
            ('new_hospitalized_patients', axes[1,0], 'New Hospitalized Patients', self.palette[2]),
            ('cfr_overall_cumulative', axes[1,1], 'Overall CFR (%)', self.palette[3]) # CFR is cumulative
        ]

        for metric, ax_item, ylabel, color in metrics_for_dashboard:
            data_series = df[metric] if metric in df.columns else pd.Series(dtype='float64') # Ensure series exists
            if not data_series.empty:
                # Apply rolling average appropriate for the metric type
                rolling_dash = rolling_short if "new_" in metric else (rolling_long if "cfr" in metric else rolling_medium)
                sns.lineplot(data=data_series.rolling(rolling_dash, center=True, min_periods=1).mean(), 
                             ax=ax_item, color=color)
            else:
                # Display message if data is missing for a subplot
                ax_item.text(0.5, 0.5, f"Data for\\n'{metric}'\\nnot available", ha='center', va='center', transform=ax_item.transAxes)

            ax_item.set_title(metric.replace('_', ' ').title()) # Auto-generate title from metric name
            ax_item.set_ylabel(ylabel)
            self._apply_date_axis_formatting(ax_item)
            ax_item.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.7)

        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to prevent suptitle overlap
        self._save_plot(fig_dash, "dashboard_key_metrics", output_dir)
        print("\n--- All visualizations generated. ---")
        return True


if __name__ == '__main__':
    # This section is for standalone testing or direct execution of the visualizer.
    # It demonstrates how to use the CovidDataVisualizer class.
    
    print("Running CovidDataVisualizer standalone demo...")
    
    # Determine project root relative to this script file for robust pathing.
    # Assumes script is in "scripts/DataVisualization/"
    current_script_path = Path(__file__).resolve()
    project_root_demo = current_script_path.parent.parent.parent 
    
    # Define a dummy input file path and output directory for the demo.
    # In a real scenario, this input file would be generated by a data processing script.
    demo_input_file = project_root_demo / "data" / "processed" / "processed_data.csv" # Example path
    demo_output_dir = project_root_demo / "data" / "visualizations_demo_output"

    print(f"Project root (for demo): {project_root_demo}")
    print(f"Demo input file: {demo_input_file}")
    print(f"Demo output directory: {demo_output_dir}")

    # Create a dummy CSV file for the demo if it doesn't exist.
    # This ensures the script can run even without actual processed data.
    file_util_demo = FileUtils()
    file_util_demo.ensure_directory_exists(demo_input_file.parent)
    if not demo_input_file.exists():
        print(f"Creating dummy data file for demo: {demo_input_file}")
        # Create a minimal DataFrame that matches expected structure.
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        
        age_suffixes = ["0_30", "30_60", "60_80", "80_plus"]
        data_columns = {
            'date': dates,
            'new_confirmed': np.random.randint(50, 200, size=len(dates)),
            'new_deceased': np.random.randint(0, 10, size=len(dates)),
            'new_hospitalized_patients': np.random.randint(5, 30, size=len(dates)),
            'new_intensive_care_patients': np.random.randint(0, 5, size=len(dates)),
            'cumulative_confirmed': np.random.randint(500, 2000, size=len(dates)).cumsum(),
            'cumulative_deceased': np.random.randint(10, 100, size=len(dates)).cumsum(),
            'cumulative_hospitalized_patients': np.random.randint(50, 300, size=len(dates)).cumsum(),
            'cumulative_intensive_care_patients': np.random.randint(10, 50, size=len(dates)).cumsum(),
        }
        for suffix in age_suffixes:
            data_columns[f'new_confirmed_{suffix}'] = np.random.randint(10, 50, size=len(dates))
            data_columns[f'new_deceased_{suffix}'] = np.random.randint(0, 3, size=len(dates))
            data_columns[f'new_hospitalized_patients_{suffix}'] = np.random.randint(1, 8, size=len(dates))
            data_columns[f'new_intensive_care_patients_{suffix}'] = np.random.randint(0, 2, size=len(dates))
            data_columns[f'cumulative_confirmed_{suffix}'] = np.random.randint(100, 500, size=len(dates)).cumsum()
            data_columns[f'cumulative_deceased_{suffix}'] = np.random.randint(2, 20, size=len(dates)).cumsum()
            data_columns[f'cumulative_hospitalized_patients_{suffix}'] = np.random.randint(10, 80, size=len(dates)).cumsum()
            data_columns[f'cumulative_intensive_care_patients_{suffix}'] = np.random.randint(2, 10, size=len(dates)).cumsum()

        dummy_df = pd.DataFrame(data_columns)
        dummy_df.to_csv(demo_input_file, index=False)
        print(f"Dummy data file created at {demo_input_file}")

    # Initialize the visualizer with the demo output directory.
    visualizer_demo = CovidDataVisualizer(default_output_dir=str(demo_output_dir))
    
    # Run all visualizations using the demo input file.
    success = visualizer_demo.run_all_visualizations(input_file=str(demo_input_file))

    if success:
        print(f"\nStandalone demo completed successfully. Plots saved to: {demo_output_dir}")
    else:
        print("\nStandalone demo encountered issues.")