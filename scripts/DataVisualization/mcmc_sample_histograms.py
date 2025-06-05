"""Generates and saves histograms for MCMC parameter samples.

This script defines utility functions and classes for file system operations
and plotting. It loads MCMC (Markov Chain Monte Carlo) simulation data from a
CSV file, then generates and saves histograms for specified parameters.
The plots are styled using seaborn and matplotlib.
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


class FileUtils:
    """A utility class for file and directory manipulations.

    Provides methods to reliably determine the project's root directory
    and ensure that specific directories exist, creating them if necessary.
    """

    def get_project_root(self) -> str:
        """Finds the project root directory.

        This method assumes a specific project structure where this script is
        located at `<project_root>/scripts/DataVisualization/`.
        It navigates three levels up from the script's current file path to
        determine the project root. It also performs a basic validation by
        checking for the presence of common project markers like 'data', 'src'
        directories, and a 'CMakeLists.txt' file.

        Returns:
            str: The absolute path to the project root directory.
                 Prints a warning if the determined path does not seem to be a
                 valid project root based on the presence of marker files/dirs.
        """
        script_file_path = Path(__file__).resolve()
        # Assumes script is in <project_root>/scripts/DataVisualization/
        project_root_candidate = script_file_path.parent.parent.parent
        # Basic check for project markers
        if (project_root_candidate / "data").is_dir() and \
           (project_root_candidate / "src").is_dir() and \
           (project_root_candidate / "CMakeLists.txt").is_file():
            return str(project_root_candidate)
        else:
            print(f"Warning: The determined project root '{project_root_candidate}' "
                  "does not appear to contain expected markers (e.g., 'data', 'src', 'CMakeLists.txt'). "
                  "The path might be incorrect if the directory structure has changed from the expectation.")
            return str(project_root_candidate)

    def ensure_directory_exists(self, path: str) -> bool:
        """Ensures that a directory exists at the given path.

        If the directory (or any of its parent directories) does not exist,
        it attempts to create them.

        Args:
            path (str): The path of the directory to check and create.

        Returns:
            bool: True if the directory exists or was successfully created,
                  False otherwise.
        """
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"Error creating directory {path}: {e}")
            return False


sns.set_theme(style="whitegrid")

# Define global font sizes for consistent plot styling.
FONTSIZE_TITLE = 18
FONTSIZE_LABEL = 14
FONTSIZE_TICKS = 12
FONTSIZE_LEGEND = 12
FONTSIZE_SUPTITLE = 20

# Update matplotlib's runtime configuration for plot appearance.
plt.rcParams.update({
    'font.size': FONTSIZE_TICKS,
    'axes.titlesize': FONTSIZE_TITLE,
    'axes.labelsize': FONTSIZE_LABEL,
    'xtick.labelsize': FONTSIZE_TICKS,
    'ytick.labelsize': FONTSIZE_TICKS,
    'legend.fontsize': FONTSIZE_LEGEND,
    'figure.titlesize': FONTSIZE_SUPTITLE,
    'figure.figsize': (10, 6),  # Default figure size
    'lines.linewidth': 2.0,     # Default line width
    'savefig.dpi': 300,         # Resolution for saved figures
    'savefig.bbox': 'tight',    # Ensure saved figures are not clipped
    'axes.titleweight': 'bold',
    'axes.labelweight': 'normal',
    'axes.edgecolor': 'black',  # Color of plot borders
    'axes.linewidth': 1.0,      # Line width of plot borders
    'patch.edgecolor': 'black', # Edge color for patches (e.g., histogram bars)
    'patch.linewidth': 0.5,     # Line width for patch edges
})


def save_plot_multiformat(fig, filename_base: str, output_dir: str, formats: list = None):
    """Saves a matplotlib figure in multiple specified formats.

    Args:
        fig (matplotlib.figure.Figure): The figure object to save.
        filename_base (str): The base name for the output file (without extension).
        output_dir (str): The directory where the plot files will be saved.
                          This directory will be created if it doesn't exist.
        formats (list, optional): A list of strings representing the desired
                                  file formats (e.g., ['png', 'pdf', 'svg']).
                                  Defaults to ['png'].
    """
    if formats is None:
        formats = ['png']

    Path(output_dir).mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    for fmt in formats:
        path = Path(output_dir) / f"{filename_base}.{fmt}"
        try:
            fig.savefig(path)
            print(f"Saved plot: {path}")
        except Exception as e:
            print(f"Error saving plot {path}: {e}")


def parameters_histograms():
    """Loads MCMC samples and plots publication-quality histograms for selected columns.

    This function performs the following steps:
    1. Determines the project root directory to locate data and output paths.
    2. Defines the input CSV file containing MCMC samples and the output directory for plots.
    3. Ensures the output directory exists.
    4. Loads the MCMC samples from the CSV file.
    5. Specifies a list of parameters (columns) for which to generate histograms.
    6. Filters this list to include only columns actually present in the loaded data.
    7. Iterates through the valid columns:
        a. Skips columns with all NaN values or no variance.
        b. Creates a histogram with a Kernel Density Estimate (KDE) overlay.
        c. Customizes plot titles and labels, using LaTeX for Greek letters.
        d. Saves each histogram in multiple formats (defaulting to PNG).
    8. Generates a special `displot` for the 'beta' parameter if present and valid.
    9. Prints completion messages and the path to the saved plots.
    """
    file_utils = FileUtils()
    try:
        project_root = file_utils.get_project_root()
    except Exception as e:
        print(f"Could not determine project root: {e}. Defaulting to relative paths.")
        project_root = "."  # Fallback to current directory as project root

    # Define paths using pathlib for robustness
    data_file_name = "mcmc_samples_20250522_212909.csv"
    data_dir = Path(project_root) / "data" / "mcmc_samples"
    data_file_path = data_dir / data_file_name

    output_dir = Path(project_root) / "plots" / "histogramsParametersSamples"
    file_utils.ensure_directory_exists(str(output_dir))

    # Check if data file exists
    if not data_file_path.exists():
        print(f"Error: Data file not found at {data_file_path}")
        return

    try:
        df = pd.read_csv(data_file_path, comment='/') # Allow comments in CSV
    except Exception as e:
        print(f"Error loading CSV file '{data_file_path}': {e}")
        return

    # Columns for which histograms will be plotted
    columns_to_plot = [
        'objective_value', 'beta', 'theta', 'sigma', 'gamma_p', 'gamma_A',
        'gamma_I', 'gamma_H', 'gamma_ICU', 'p_0', 'p_1', 'p_2', 'p_3',
        'h_0', 'h_1', 'h_2', 'h_3', 'icu_0', 'icu_1', 'icu_2', 'icu_3',
        'd_H_0', 'd_H_1', 'd_H_2', 'd_H_3', 'd_ICU_0', 'd_ICU_1', 'd_ICU_2',
        'd_ICU_3', 'kappa_2', 'kappa_3', 'kappa_4', 'kappa_5', 'kappa_6', 'kappa_7', 'kappa_8'
    ]

    # Filter to only columns present in the DataFrame to avoid errors
    valid_columns_to_plot = [col for col in columns_to_plot if col in df.columns]
    if not valid_columns_to_plot:
        print("No valid columns selected or found in the DataFrame for plotting.")
        return

    print(f"Plotting histograms for: {', '.join(valid_columns_to_plot)}")

    # Define consistent colors for histogram and KDE plot
    hist_color = sns.color_palette("pastel")[0]
    kde_color = sns.color_palette("deep")[0]

    for column in valid_columns_to_plot:
        # Skip columns with no data or no variance
        if df[column].isnull().all():
            print(f"Skipping '{column}' as it contains only NaN values.")
            continue
        if df[column].nunique() <= 1:
            print(f"Skipping '{column}' as it has no variance or only one unique value.")
            continue

        plt.figure() # Create a new figure for each histogram

        try:
            sns.histplot(
                df[column].dropna(),  # Remove NaN values before plotting
                bins='auto',          # Automatically determine optimal number of bins
                kde=True,
                line_kws={'linewidth': plt.rcParams['lines.linewidth'] + 0.5, 'color': kde_color},
                color=hist_color,
                edgecolor=plt.rcParams['patch.edgecolor'],
                linewidth=plt.rcParams['patch.linewidth']
            )

            # Format column names for display, using LaTeX for Greek letters
            col_display_name = column.replace('_', r'\_') # Escape underscores for LaTeX
            if column == 'beta': col_display_name = r'$\beta$'
            elif column == 'theta': col_display_name = r'$\theta$'
            elif column == 'sigma': col_display_name = r'$\sigma$'
            elif column.startswith('gamma_'): col_display_name = fr'$\gamma_{{{column.split("_" )[1]}}}$'
            # Add more specific formatting for other parameters if needed

            plt.title(f'Distribution of {col_display_name}')
            plt.xlabel(col_display_name)
            plt.ylabel('Frequency')

            plt.grid(True, which='major', linestyle=':', linewidth=0.5, alpha=0.6)

            save_plot_multiformat(plt.gcf(), f'{column}_hist', str(output_dir))

        except Exception as e:
            print(f"Error plotting histogram for column '{column}': {e}")
        finally:
            plt.close()  # Close the figure to free memory

    print("Histogram plotting complete.")

    # Generate a special displot for 'beta' if it's a valid column
    if 'beta' in valid_columns_to_plot and not df['beta'].isnull().all() and df['beta'].nunique() > 1:
        print("\nGenerating special displot for 'beta'...")
        try:
            displot_fig = sns.displot(
                df['beta'].dropna(),
                kind="hist",
                kde=True,
                rug=True,  # Add a rug plot
                aspect=1.5,
                bins='auto',
                color=hist_color,
                line_kws={'linewidth': plt.rcParams['lines.linewidth'] + 0.5, 'color': kde_color},
                edgecolor=plt.rcParams['patch.edgecolor'],
                linewidth=plt.rcParams['patch.linewidth']
            )
            displot_fig.set_titles(r'Distribution of $\beta$')
            displot_fig.set_xlabels(r'$\beta$')
            displot_fig.set_ylabels('Density / Frequency')

            for ax in displot_fig.axes.flatten():
                ax.grid(True, which='major', linestyle=':', linewidth=0.5, alpha=0.6)

            save_plot_multiformat(displot_fig.fig, 'beta_displot', str(output_dir))
            plt.close(displot_fig.fig)
        except Exception as e:
            print(f"Could not generate displot for 'beta': {e}")

    print(f"\nPlots saved in: {output_dir}")