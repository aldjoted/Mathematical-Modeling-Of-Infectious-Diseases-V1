# Mathematical Modeling of SARS-CoV-2 Dynamics in Spain and General SIR Framework

This C++ project implements and calibrates an age-structured deterministic compartmental model (SEPAIHRD) to simulate the SARS-CoV-2 epidemic dynamics in Spain during 2020. It also includes a C++ framework with implementations of fundamental SIR (Susceptible-Infected-Recovered) models. The project focuses on Bayesian calibration techniques, assessment of Non-Pharmaceutical Intervention (NPI) impacts, and estimation of hidden epidemic dynamics, applying methodologies discussed in the SMATM128 course (UNamur).

## Table of Contents

*   [Features](#features)
*   [Models Implemented](#models-implemented)
*   [Technologies Used](#technologies-used)
*   [Prerequisites](#prerequisites)
*   [Installation](#installation)
*   [Project Structure](#project-structure)
*   [Usage](#usage)
    *   [Running Simulations](#running-simulations)
    *   [Input Data](#input-data)
    *   [Output Data](#output-data)
*   [Configuration](#configuration)
*   [Running Tests](#running-tests)
*   [Memory Checking](#memory-checking)
*   [Scripts](#scripts)
*   [Contributing](#contributing)
*   [License](#license)
*   [References](#references)
*   [Contact](#contact)

## Features

*   **SEPAIHRD Model:**
    *   Age-structured (4 age groups: 0-30, 30-60, 60-80, 80+) deterministic model.
    *   Estimates hidden dynamics (e.g., true prevalence).
    *   Incorporates piece-wise constant Non-Pharmaceutical Interventions (NPIs).
*   **Base SIR Framework:**
    *   Standard deterministic SIR model.
    *   Stochastic SIR model using the Gillespie algorithm or binomial chain method.
*   **Simulation Engine:**
    *   Flexible simulation engine for running models over time.
    *   Support for various ODE solver strategies (e.g., Dopri5, Cash-Karp, Fehlberg).
*   **Calibration Framework:**
    *   Bayesian calibration using MCMC (Metropolis-Hastings).
    *   Optimization algorithms (e.g., Particle Swarm Optimization, Hill Climbing) for finding initial parameters.
    *   Objective functions (e.g., Poisson likelihood).
*   **Modular Design:**
    *   Uses interfaces for models, ODE solvers, optimization algorithms, parameter managers, and objective functions, promoting extensibility.
    *   Includes utility classes for file operations, CSV parsing, data handling, and logging.
*   **Data Output:**
    *   Simulation results, calibration outputs (e.g., MCMC chains, summary statistics), and logs can be saved to CSV and text files.
*   **Testing:**
    *   Unit tests using Google Test framework.
*   **Memory Checking:**
    *   Integrated Valgrind support for memory leak detection.

## Models Implemented

1.  **Age-Structured SEPAIHRD Model:** The primary model for simulating SARS-CoV-2 dynamics.
    *   Implemented in `src/model/` and `include/model/`.
    *   Key classes: [`AgeSEPAIHRDModel`](include/model/AgeSEPAIHRDModel.hpp), [`AgeSEPAIHRDSimulator`](include/model/AgeSEPAIHRDsimulator.hpp), [`SEPAIHRDModelCalibration`](include/model/SEPAIHRDModelCalibration.hpp), [`PieceWiseConstantNPIStrategy`](include/model/PieceWiseConstantNPIStrategy.hpp).
2.  **Age-Structured SIR Model:** A generic age-structured SIR model.
    *   Implemented in `src/sir_age_structured/` and `include/sir_age_structured/`.
    *   Key classes: [`AgeSIRModel`](include/sir_age_structured/AgeSIRModel.hpp), [`ModelCalibrator`](include/sir_age_structured/ModelCalibrator.hpp).
3.  **Standard Deterministic SIR Model:**
    *   Implementation based on [`SIRModel.hpp`](include/base/SIRModel.hpp) and [`SIRModel.cpp`](src/base/SIRModel.cpp).
4.  **SIR Model with Population Dynamics (Vital Dynamics):**
    *   Implementation based on [`SIR_population_variable.hpp`](include/base/SIR_population_variable.hpp) and [`SIR_population_variable.cpp`](src/base/SIR_population_variable.cpp).
5.  **Stochastic SIR Model:**
    *   Implementation based on [`SIR_stochastic.hpp`](include/base/SIR_stochastic.hpp) and [`SIR_stochastic.cpp`](src/base/SIR_stochastic.cpp).

## Technologies Used

*   **Programming Language:** C++17
*   **Build System:** CMake (version 3.10 or higher)
*   **Major Libraries:**
    *   **Eigen3 (version 3.3+):** For linear algebra operations (matrices, vectors).
    *   **Google Test:** For unit testing.
    *   **GSL (GNU Scientific Library):** For numerical integration and random number generation.
    *   **Boost Libraries:** (Specifically `system` and headers).

## Prerequisites

*   **C++ Compiler:** A C++17 compatible compiler (e.g., GCC, Clang, MSVC).
*   **CMake:** Version 3.10 or newer.
*   **GSL (GNU Scientific Library):** Must be installed on your system.
*   **Eigen3 Library:** Must be installed or accessible by CMake.
*   **Boost Libraries:** (Specifically `system` and headers). Must be installed or accessible by CMake.
*   **Git:** For cloning the repository.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/aldjoted/Mathematical-Modeling-Of-Infectious-Diseases.git
    cd Mathematical-Modeling-Of-Infectious-Diseases
    ```

2.  **Create a build directory and navigate into it:**
    ```bash
    mkdir build
    cd build
    ```

3.  **Run CMake to configure the project:**
    ```bash
    cmake ..
    ```
    This will detect dependencies and generate Makefiles (or project files for other generators).

4.  **Compile the project:**
    ```bash
    make
    ```
    This will build all libraries and executables. Executables will be placed in the `build/bin/` directory.

## Project Structure

The project is organized as follows:

*   `CMakeLists.txt`: The main CMake build script for the project.
*   `Readme.md`: This file.
*   `build/`: Directory created during the build process. Contains compiled executables (in `build/bin/`) and other build artifacts.
*   `data/`: Contains input data, configuration files, and output results.
    *   `data/contacts.csv`: Example contact matrix data for age-structured models.
    *   `data/calibration/`: Contains data used for model calibration, such as observed epidemiological data (e.g., daily hospitalizations, ICU admissions, deaths, case counts) stratified by age groups for Spain. Specific file formats might include CSVs with time series data.
    *   `data/calibration_output/`: Stores results from model calibration, like MCMC samples (e.g., `mcmc_summary.csv`) and posterior predictive checks.
    *   `data/configuration/`: Contains parameter files for simulations and calibration (e.g., `initial_guess.txt`, `param_bounds.txt`, `proposal_sigmas.txt`, `params_to_calibrate.txt`, `pso_settings.txt`, `mcmc_settings.txt`, `hill_climbing_settings.txt`).
    *   `data/output/`: Default directory for simulation results (e.g., `sepaihrd_age_baseline_results.csv`, `sir_age_baseline_results.csv`).
    *   `data/processed/`: Contains data that has undergone pre-processing steps, ready for model input or analysis (e.g., `processed_data.csv`).
    *   `data/raw/`: Contains original, unaltered data obtained from various sources. This might include publicly available datasets on COVID-19 cases, hospitalizations, deaths, and demographic information for Spain.
*   `docs/`: Contains additional documentation. This may include Doxygen-generated API documentation, detailed model descriptions, calibration methodology explanations, or design documents outlining the software architecture.
*   `include/`: Contains header files (`.hpp`) for the C++ source code.
    *   `include/base/`: Header files for base SIR models.
    *   `include/exceptions/`: Custom exception classes.
    *   `include/model/`: Header files for the SEPAIHRD model and related components (parameters, objectives, optimizers).
    *   `include/sir_age_structured/`: Header files for the age-structured SIR model and related components.
    *   `include/utils/`: Utility class headers (e.g., [`FileUtils.hpp`](include/utils/FileUtils.hpp), [`Logger.hpp`](include/utils/Logger.hpp), CSV parsing).
*   `scripts/`: Contains utility scripts, possibly for data processing, visualization, or running batches of simulations. See [Scripts Section](#scripts) for more details.
    *   `scripts/age-structured-calibration/`: Scripts related to calibrating age-structured models.
    *   `scripts/data-processing/`: Scripts for processing data.
    *   `scripts/data-visualization/`: Scripts for visualizing results.
    *   `scripts/utils/`: Utility scripts.
*   `src/`: Contains C++ source code implementation files (`.cpp`).
    *   `src/base/`: Source files for base SIR models.
        *   `src/base/main/`: Main executable files for the base SIR models.
        *   `src/base/docs/`: Markdown documentation for the base models.
    *   `src/exceptions/`: Source files for custom exception classes.
    *   `src/model/`: Source files for the SEPAIHRD model, including its main executable (`main.cpp`), objectives, optimizers, and parameter management.
    *   `src/sir_age_structured/`: Source files for the age-structured SIR model, including its main executable (`main.cpp`), calibration demo, objectives, optimizers, etc.
    *   `src/utils/`: Utility class implementations.
*   `tests/`: Contains unit tests for the project, using the Google Test framework.
    *   `tests/model/`: Tests for the SEPAIHRD model.
    *   `tests/sir_age_structured/`: Tests for the age-structured SIR model.
    *   `tests/utils/`: Tests for utility classes.

## Usage

### Running Simulations

Executables are built into the `build/bin/` directory.

*   **SEPAIHRD Age-Structured Model:**
    The main executable for the SEPAIHRD model is `sepaihrd_age_structured_main`.
    ```bash
    cd build/bin
    ./sepaihrd_age_structured_main [options]
    ```
    Available options (see `src/model/main.cpp` for details):
    *   `--algorithm <name>` or `-a <name>`: Choose calibration algorithm.
        *   `pso` or `psomcmc`: Particle Swarm Optimization followed by MCMC (default).
        *   `hill` or `hillmcmc`: Hill Climbing followed by MCMC.
    *   `--help` or `-h`: Show help message.

*   **Age-Structured SIR Model:**
    Run the main simulation:
    ```bash
    cd build/bin
    ./sir_age_structured_main
    ```
    Run the calibration demo:
    ```bash
    cd build/bin
    ./sir_age_structured_calibration_demo
    ```

*   **Base SIR Models:**
    Executables for basic SIR models are also available:
    ```bash
    cd build/bin
    ./sir_model
    ./sir_pop_var
    ./sir_stochastic
    ```
    These typically run predefined scenarios or use parameters from configuration files or hardcoded values.

### Input Data

The models and calibration routines rely on various input files, primarily located in the `data/` directory:

*   **Contact Matrices:** (e.g., `data/contacts.csv`)
    *   CSV file representing the contact rates between different age groups.
    *   Dimensions: `num_age_classes` x `num_age_classes`.
*   **Epidemiological Data:** (e.g., `data/processed/processed_data.csv`)
    *   CSV file containing time series of observed data for calibration (e.g., daily new hospitalizations, ICU admissions, deaths, reported cases).
    *   Typically includes a 'date' column and data columns for different metrics, often stratified by age group.
    *   Used by `CalibrationData` class.
*   **Model & Calibration Configuration Files:** (located in `data/configuration/`)
    *   `initial_guess.txt`: Initial values for parameters to be calibrated. Format: one parameter per line with its value, or specific format for `FileUtils::readSEPAIHRDParameters`.
    *   `param_bounds.txt`: Lower and upper bounds for parameters during calibration. Format: `parameter_name lower_bound upper_bound` per line.
    *   `proposal_sigmas.txt`: Proposal standard deviations for MCMC sampling. Format: `parameter_name sigma_value` per line.
    *   `params_to_calibrate.txt`: A list of parameter names that the calibration algorithm should optimize. Format: one parameter name per line.
    *   `pso_settings.txt`, `mcmc_settings.txt`, `hill_climbing_settings.txt`: Key-value pairs for algorithm-specific settings (e.g., `num_particles value`, `num_iterations value`).
    *   General parameter files (e.g., for SEPAIHRD model): Text files with parameter names and their values (scalar or age-specific vectors). Lines starting with `#` are comments. See `FileUtils::readSEPAIHRDParameters` for parsing logic.

### Output Data

*   **Simulation Results:**
    *   Saved as CSV files in `data/output/` (e.g., `sepaihrd_age_baseline_results.csv`, `sepaihrd_age_final_calibrated_run.csv`, `sir_age_baseline_results.csv`).
    *   Typically include columns for time and the state of each compartment (S, E, P, A, I, H, R, D) for each age group.
*   **Calibration Outputs:**
    *   Stored in `data/calibration_output/`.
    *   `mcmc_summary.csv`: Contains summary statistics (mean, median, standard deviation, quantiles) for the posterior distributions of calibrated parameters and derived metrics.
    *   Other files might include MCMC trace plots (if generated by scripts), posterior predictive check data, etc.
*   **Log Files:**
    *   Application logs are printed to the console.
    *   File logging can be enabled via the [`Logger`](include/utils/Logger.hpp) class, typically to a file like `epidemic_model.log` in the execution directory.

## Configuration

Simulations and calibrations are configured through:

1.  **Command-line arguments:** As seen with `sepaihrd_age_structured_main` for selecting calibration algorithms.
2.  **Configuration Files:** Primarily text files located in `data/configuration/`. These files define:
    *   Model parameters (e.g., transmission rates, recovery rates, NPI effectiveness).
    *   Initial conditions for the model compartments.
    *   Parameters for calibration algorithms (e.g., number of MCMC iterations, PSO particle count).
    *   Which parameters to calibrate and their bounds.
    *   NPI strategy parameters (e.g., start/end dates of interventions, strength of interventions).
3.  **Source Code:** Some scenarios or default parameters might be hardcoded, especially for simpler models or demo applications.

Refer to the `main.cpp` files in `src/model/`, `src/sir_age_structured/`, and `src/base/main/` for specific configuration loading mechanisms. Utility functions in `src/utils/ReadCalibrationConfiguration.cpp` and `src/utils/FileUtils.cpp` handle parsing of many configuration files.

## Running Tests

The project uses Google Test for unit testing.

1.  Ensure the project is built (see [Installation](#installation)).
2.  From the `build` directory, run:
    ```bash
    make test
    ```
    or
    ```bash
    ctest
    ```
    This will execute all defined tests. Test executables (e.g., `utils_tests`, `model_tests`, `sir_age_structured_tests`) are also created in `build/bin/` (or a test-specific subdirectory within `build/`) and can be run individually.

## Memory Checking

Memory leak detection is integrated using Valgrind.

1.  Ensure the project is built.
2.  From the `build` directory, run `make memcheck_<executable_name>`. For example:
    ```bash
    make memcheck_sepaihrd_age_structured_main
    ```
    This will run the specified executable (e.g., `sepaihrd_age_structured_main`) under Valgrind with options to check for memory leaks. Other executables with memcheck targets include `sir_model`, `sir_pop_var`, `sir_stochastic`, `sir_age_structured_main`, and `sir_age_structured_calibration_demo`.

## Scripts

The `scripts/` directory contains utility scripts for various tasks. For detailed information, refer to `scripts/Readme.md`.

*   **`scripts/age-structured-calibration/`**: Scripts for calibrating age-structured models, potentially involving MCMC analysis or parameter fitting routines.
*   **`scripts/data-processing/`**: Scripts for cleaning raw data, transforming data formats, and other pre-processing steps necessary for model input.
*   **`scripts/data-visualization/`**: Scripts to generate plots, charts, or other graphical representations of data, model outputs, and calibration results.
*   **`scripts/utils/`**: Utility scripts providing helper functions or common functionalities used by other scripts in the project.

## Contributing
Alex Djousse Tedongmene, Assited by AI

## License

## References

*   Methodologies discussed in the SMATM128 course (UNamur).
* Particle Swarm Optimization Algorithm and Its Applications.
* COVID-19 Belgium: Extended SEIR-QD model with nursing homes and
long-term scenarios-based forecasts

## Contact

(TODO: )
