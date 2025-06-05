# Scripts Directory

This directory contains a comprehensive collection of Python scripts for COVID-19 epidemiological modeling using the SEPAIHRD (Susceptible-Exposed-Pre-symptomatic-Asymptomatic-Infected-Hospitalized-ICU-Recovered-Deceased) model. The scripts support the entire modeling workflow from COVID-19 data preparation to MCMC model calibration and results visualization.

## Overview

The scripts are organized into specialized subdirectories, each serving a specific purpose in the COVID-19 modeling pipeline. These tools support age-structured epidemiological modeling with four age groups (0-30, 30-60, 60-80, 80+), Bayesian parameter estimation, and comprehensive visualization of model results.

## Directory Structure

### `age-structured-calibration/`
**Purpose**: Model calibration validation for age-structured SEPAIHRD models

This subdirectory contains scripts for:
- **Calibration Comparison** (`compare_calibration_and_observed.py`): Validates calibrated model results against observed COVID-19 data
- Generates side-by-side comparison plots for each age group (0-30, 30-60, 60-80, 80+)
- Visualizes model fit quality with scatter plots for observed data and line plots for simulated results
- Supports assessment of model performance across different age strata

**Key Scripts**:
- `compare_calibration_and_observed.py`: Main comparison and visualization script

### `data-processing/`
**Purpose**: COVID-19 data preprocessing and age-group aggregation

This subdirectory includes scripts for:
- **COVID-19 Data Processing** (`CovidDataProcessor.py`): Aggregates age-specific COVID-19 metrics into four age groups
- Processes confirmed cases, deaths, hospitalizations, and ICU admissions
- Handles population data aggregation for demographic analysis
- Data validation and format standardization for Spanish COVID-19 data (ES.csv)

**Core Classes**:
- `CovidDataProcessor`: Main data processing class with age group aggregation
- Aggregates metrics: confirmed cases, deaths, hospitalizations, ICU admissions
- Age group mapping: 0-30, 30-60, 60-80, 80+ years

**Key Scripts**:
- `CovidDataProcessor.py`: Core data processing functionality
- `main_data_processing.py`: Main execution script for data processing workflow

### `DataVisualization/`
**Purpose**: Comprehensive visualization of COVID-19 data and SEPAIHRD model results

This subdirectory contains scripts for:
- **COVID-19 Data Visualization** (`CovidDataVisualizer.py`): Extensive visualization toolkit (760+ lines)
- Time series plotting with trend analysis and growth rate calculations
- Age-stratified visualization with grouped time series and stacked area charts
- **MCMC Results Visualization**: Parameter posterior distribution plotting and convergence diagnostics
- **SEPAIHRD Model Dynamics**: Compartmental model visualization with NPI period annotations
- **Post-Calibration Analysis**: Comprehensive model output analysis and validation

**Visualization Types**:
- Time series plots with confidence intervals and trend analysis
- Age-stratified incidence, mortality, and hospitalization charts
- MCMC parameter histograms and posterior distributions
- SEPAIHRD compartment dynamics with intervention period shading
- Model calibration validation plots

**Key Scripts**:
- `CovidDataVisualizer.py`: Main COVID-19 data visualization class
- `PostCalibrationAnalysis.py`: SEPAIHRD model analysis and plotting (600+ lines)
- `mcmc_sample_histograms.py`: MCMC parameter visualization utilities
- `plot_sepaihrd_dynamics.py`: SEPAIHRD compartment dynamics plotting
- `main_data_visualization.py`: Main visualization workflow execution

### `model/`
**Purpose**: Post-calibration model analysis and validation

This subdirectory provides:
- **SEPAIHRD Model Analysis** (`PostCalibrationAnalysis.py`): Comprehensive post-calibration analysis
- Model output validation and statistical analysis
- Intervention period analysis with NPI (Non-Pharmaceutical Intervention) shading
- Age-structured model performance assessment
- Model prediction validation against historical data

**Core Features**:
- Statistical analysis of model outputs
- Intervention period visualization
- Age-stratified model validation
- Comprehensive reporting and figure generation

### `utils/`
**Purpose**: Shared utilities and helper functions

This subdirectory provides:
- **File System Utilities** (`FileUtils.py`): Project structure management and directory operations
- Project root detection based on standard directory markers (data/, include/, src/)
- Directory creation and validation utilities
- Cross-platform file path handling

**Core Utilities**:
- Project root auto-detection
- Directory management functions
- File system operation helpers

## Getting Started

### Prerequisites
- **Python 3.8+** with scientific computing libraries:
  - NumPy, SciPy, Pandas for data manipulation
  - Matplotlib, Seaborn for visualization
  - Pathlib for cross-platform file handling
- **Data Requirements**: Spanish COVID-19 data (ES.csv) in the expected format
- **Model Output**: SEPAIHRD model simulation results and MCMC samples

### Recommended Workflow
1. **Data Preparation**: Use `data-processing/main_data_processing.py` to process raw Spanish COVID-19 data
2. **Exploratory Analysis**: Run `DataVisualization/main_data_visualization.py` to generate COVID-19 trend visualizations
3. **Model Calibration**: Execute C++ SEPAIHRD model calibration (see main project documentation)
4. **Calibration Validation**: Use `age-structured-calibration/compare_calibration_and_observed.py` to validate model fit
5. **Results Analysis**: Apply `model/PostCalibrationAnalysis.py` for comprehensive model output analysis
6. **Parameter Visualization**: Generate MCMC parameter plots using `DataVisualization/mcmc_sample_histograms.py`

### Age Group Structure
All scripts use a consistent four-group age stratification:
- **0-30 years**: Young adults and children
- **30-60 years**: Middle-aged adults  
- **60-80 years**: Older adults
- **80+ years**: Elderly population

### Model Framework
The scripts support the **SEPAIHRD** compartmental model:
- **S**: Susceptible
- **E**: Exposed
- **P**: Pre-symptomatic
- **A**: Asymptomatic
- **I**: Infected (symptomatic)
- **H**: Hospitalized
- **ICU**: Intensive Care Unit
- **R**: Recovered
- **D**: Deceased

## Usage Guidelines

### File Structure and Paths
- Scripts use `FileUtils.get_project_root()` for automatic project root detection
- Input data expected in `data/raw/` (raw data) and `data/processed/` (processed data)
- Output visualizations saved to `data/visualizations/` and `data/output/`
- MCMC samples stored in `data/mcmc_samples/`

### Data Format Requirements
- **Raw COVID-19 data**: Spanish format (ES.csv) with age-specific columns
- **Processed data**: Age-aggregated format with four age groups
- **MCMC samples**: CSV format with parameter columns
- **Simulation results**: CSV format with time series and age-stratified compartments

### Execution Examples
```bash
# Process raw COVID-19 data
cd scripts/data-processing/
python main_data_processing.py

# Generate COVID-19 visualizations
cd ../DataVisualization/
python main_data_visualization.py

# Compare calibration results
cd ../age-structured-calibration/
python compare_calibration_and_observed.py

# Analyze SEPAIHRD model outputs
cd ../model/
python PostCalibrationAnalysis.py
```

### Configuration
- Age group definitions are hardcoded in processing scripts
- Visualization settings (colors, fonts, figure sizes) defined in individual scripts
- NPI (Non-Pharmaceutical Intervention) periods defined in `PostCalibrationAnalysis.py`
- File paths automatically detected using project structure markers

## Key Features

### COVID-19 Specific Implementation
- **Spanish COVID-19 Data Processing**: Specialized for ES.csv format with age-specific columns
- **Four Age Group Structure**: Consistent 0-30, 30-60, 60-80, 80+ stratification across all scripts
- **SEPAIHRD Model Support**: Full visualization and analysis support for 9-compartment model
- **NPI Period Analysis**: Built-in support for Spanish intervention periods (lockdowns, de-escalation phases)

### Advanced Visualization Capabilities
- **Time Series Analysis**: Growth rate calculation, trend analysis, and smoothing
- **Age-Stratified Plots**: Grouped time series, stacked area charts, and heat maps
- **MCMC Diagnostics**: Parameter histograms, posterior distributions, and convergence analysis  
- **Model Validation**: Side-by-side comparison of observed vs. simulated data
- **Publication Ready**: High-quality figures with customizable styling and export options

### Modular Design
- **Standalone Scripts**: Each script can be run independently with automatic path detection
- **Shared Utilities**: Common functions for file operations and project structure management
- **Flexible Input/Output**: Automatic directory detection and creation for results storage

## Contributing

When adding new scripts:
- Follow the four age group structure (0-30, 30-60, 60-80, 80+)
- Use `FileUtils.get_project_root()` for path detection
- Include comprehensive docstrings following the existing format
- Add visualization scripts to `DataVisualization/` subdirectory
- Follow seaborn/matplotlib styling conventions established in existing scripts
- Test with Spanish COVID-19 data format (ES.csv)

## Support

For questions about specific scripts or COVID-19 modeling approaches, refer to:
- **Script Documentation**: Comprehensive docstrings in each Python file
- **Model Documentation**: See main project README for SEPAIHRD model details
- **Data Format**: Spanish COVID-19 data specifications in `data-processing/` scripts
- **Visualization Examples**: Output samples in `data/visualizations/` directory

## Dependencies

### Required Python Packages
```python
# Core scientific computing
numpy
scipy
pandas

# Visualization
matplotlib
seaborn

# Utilities
pathlib
argparse
warnings
```

### Optional Packages
```python
# For enhanced statistical analysis
statsmodels

# For interactive plotting
plotly
```

Install dependencies with:
```bash
pip install numpy scipy pandas matplotlib seaborn
```