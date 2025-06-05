"""Main script to generate COVID-19 data visualizations.

This script initializes the necessary components, including determining the
project root, setting up file paths for input data and output visualizations,
and then uses the `CovidDataVisualizer` class to generate and save all plots.
"""

import sys
from pathlib import Path

# Determine the project root dynamically from the script's location.
# This allows the script to be run from different working directories.
PROJECT_ROOT_FROM_SCRIPT_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT_FROM_SCRIPT_PATH))

from CovidDataVisualizer import CovidDataVisualizer
# Assuming FileUtils is part of a utility module within the project structure.
from scripts.utils.FileUtils import FileUtils

# --- Main script logic ---

# 1. Initialize FileUtils and determine project root.
# FileUtils is expected to have a method `get_project_root()`.
# If it fails, a fallback mechanism uses the script's path.
main_file_utils = FileUtils()

try:
    project_root_str = main_file_utils.get_project_root()
    project_root = Path(project_root_str)
    print(f"Project root identified by FileUtils: {project_root}")
except Exception as e:
    print(f"Error getting project root from FileUtils: {e}")
    print(f"Falling back to project root derived from script path: {PROJECT_ROOT_FROM_SCRIPT_PATH}")
    project_root = PROJECT_ROOT_FROM_SCRIPT_PATH


# 2. Define input and output file paths using pathlib for OS-agnostic paths.
input_filename = "processed_data.csv"
input_path = project_root / "data" / "processed" / input_filename
output_dir = project_root / "data" / "visualizations"


# 3. Create CovidDataVisualizer object.
# The visualizer will handle the plotting logic.
visualizer = CovidDataVisualizer(default_output_dir=str(output_dir))
print(f"CovidDataVisualizer initialized. Default output directory: {visualizer.default_output_dir}")


# 4. Ensure output directory exists before attempting to save plots.
if main_file_utils.ensure_directory_exists(str(output_dir)):
    print(f"Output directory successfully ensured: {output_dir}")

    # 5. Run the visualizations using the specified input data.
    print(f"Attempting to run visualizations on: {input_filename}")
    print(f"Full input file path: {input_path}")
    success = visualizer.run_all_visualizations(input_file=str(input_path))
    
    if success:
        print("-" * 50)
        print("Data visualization process completed successfully.")
        print(f"All visualization outputs should be located in: {output_dir}")
        print("-" * 50)
    else:
        print("-" * 50)
        print("Data visualization process encountered issues. Please check the logs above for details.")
        print("-" * 50)
else:
    # Critical error if the output directory cannot be accessed or created.
    print(f"Critical Error: Failed to create or access the main output directory: {output_dir}")
    print("Please check path validity and write permissions. Visualization process cannot continue.")