import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from scripts.utils.FileUtils import FileUtils
from CovidDataProcessor import CovidDataProcessor

# Create a CovidDataProcessor object
processor = CovidDataProcessor()

# Create a FileUtils object
file_utils = FileUtils()

# Get project root directory
project_root = file_utils.get_project_root()

# Define input and output file paths
input_filename = "ES.csv"
output_filename = "processed_data.csv"
input_path = f"{project_root}/data/raw/{input_filename}"
output_path = f"{project_root}/data/processed/{output_filename}"

# Processed the data
success = processor.process_data(input_path, output_path, start_date='2020-01-01', end_date='2020-12-31')

if success:
    print("Data processing successful.")