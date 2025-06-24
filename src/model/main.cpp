#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <iomanip>
#include <map>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>

// Model specific includes
#include "model/AgeSEPAIHRDModel.hpp"
#include "model/AgeSEPAIHRDsimulator.hpp"
#include "model/PieceWiseConstantNPIStrategy.hpp"
#include "model/parameters/SEPAIHRDParameterManager.hpp"
#include "model/parameters/SEPAIHRDParameters.hpp"
#include "model/objectives/SEPAIHRDObjectiveFunction.hpp"
#include "model/SEPAIHRDModelCalibration.hpp"
#include "model/optimizers/ParticleSwarmOptimizer.hpp"
#include "model/ReproductionNumberCalculator.hpp"
#include "sir_age_structured/optimizers/MetropolisHastingsSampler.hpp"
#include "sir_age_structured/ModelCalibrator.hpp"
#include "sir_age_structured/solvers/Dropri5SolverStrategy.hpp"
#include "sir_age_structured/SimulationResultProcessor.hpp"
#include "sir_age_structured/caching/SimulationCache.hpp"
#include "model/PostCalibrationAnalyser.hpp"

// Utility includes
#include "utils/FileUtils.hpp"
#include "utils/ReadContactMatrix.hpp"
#include "utils/GetCalibrationData.hpp"
#include "exceptions/Exceptions.hpp"
#include "utils/Logger.hpp"
#include "utils/ReadCalibrationConfiguration.hpp"

using namespace Eigen;
using namespace std;
using namespace epidemic;

void printUsage(const char* programName) {
    cout << "Usage: " << programName << " [--algorithm|-a <algorithm>] [--help|-h]" << endl;
    cout << "Options:" << endl;
    cout << "  --algorithm, -a <algorithm>  Choose calibration algorithm:" << endl;
    cout << "                              'pso' or 'psomcmc' for PSO + MCMC (default)" << endl;
    cout << "                              'hill' or 'hillmcmc' for Hill Climbing + MCMC" << endl;
    cout << "  --help, -h                  Show this help message" << endl;
}

enum class CalibrationAlgorithm {
    PSO_MCMC,
    HILL_MCMC
};

CalibrationAlgorithm parseAlgorithm(const string& algorithm) {
    string algo_lower = algorithm;
    transform(algo_lower.begin(), algo_lower.end(), algo_lower.begin(), ::tolower);
    
    if (algo_lower == "pso" || algo_lower == "psomcmc") {
        return CalibrationAlgorithm::PSO_MCMC;
    } else if (algo_lower == "hill" || algo_lower == "hillmcmc") {
        return CalibrationAlgorithm::HILL_MCMC;
    } else {
        throw invalid_argument("Unknown algorithm: " + algorithm + 
                             ". Valid options are: pso, psomcmc, hill, hillmcmc");
    }
}

std::shared_ptr<PiecewiseConstantNpiStrategy> createNpiStrategy(
    const SEPAIHRDParameters& params,
    const std::vector<std::string>& all_kappa_parameter_names,
    const std::map<std::string, std::pair<double, double>>& overall_param_bounds,
    int fixed_kappa_model_index = 0) {
    
    std::map<std::string, std::pair<double, double>> npi_specific_bounds;
    for(const auto& name : all_kappa_parameter_names) {
        if (name == "kappa_1") continue; // ne pas ajouté le kappa fixe
        auto it = overall_param_bounds.find(name);
        if (it != overall_param_bounds.end()) {
            npi_specific_bounds[name] = it->second;
        } else if (name == "kappa_1") {
            npi_specific_bounds[name] = {1.0, 1.0};
        }
    }

    double baseline_kappa_val = params.kappa_values.at(fixed_kappa_model_index);
    double baseline_end_time_val = params.kappa_end_times.at(fixed_kappa_model_index);
    bool is_baseline_fixed_val = true;

    std::vector<double> npi_end_times_after_baseline;
    std::vector<double> npi_values_after_baseline;
    std::vector<std::string> param_names_for_npi_values;
    
    if (params.kappa_end_times.size() > static_cast<size_t>(fixed_kappa_model_index + 1)) {
        npi_end_times_after_baseline.assign(
            params.kappa_end_times.begin() + fixed_kappa_model_index + 1, 
            params.kappa_end_times.end()
        );
        npi_values_after_baseline.assign(
            params.kappa_values.begin() + fixed_kappa_model_index + 1, 
            params.kappa_values.end()
        );
        param_names_for_npi_values.assign(
            all_kappa_parameter_names.begin() + fixed_kappa_model_index + 1, 
            all_kappa_parameter_names.end()
        );
    }

    return std::make_shared<PiecewiseConstantNpiStrategy>(
        npi_end_times_after_baseline,
        npi_values_after_baseline,
        npi_specific_bounds,
        baseline_kappa_val,
        baseline_end_time_val,
        is_baseline_fixed_val,
        param_names_for_npi_values
    );
}

void ensureOutputSubdirectoryExists(const std::string& base_dir, const std::string& subdir_name) {
    FileUtils::ensureDirectoryExists(FileUtils::joinPaths(base_dir, subdir_name));
}

int main(int argc, char* argv[]) {
    // === COMMAND LINE PARSING ===
    CalibrationAlgorithm selected_algorithm = CalibrationAlgorithm::PSO_MCMC;
    
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--algorithm" || arg == "-a") {
            if (i + 1 < argc) {
                try {
                    selected_algorithm = parseAlgorithm(argv[i + 1]);
                    i++; // Skip the next argument
                } catch (const invalid_argument& e) {
                    cerr << "Error: " << e.what() << endl;
                    printUsage(argv[0]);
                    return 1;
                }
            } else {
                cerr << "Error: --algorithm option requires a value" << endl;
                printUsage(argv[0]);
                return 1;
            }
        } else {
            cerr << "Error: Unknown option: " << arg << endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    string algorithm_name = (selected_algorithm == CalibrationAlgorithm::PSO_MCMC) ? 
                          "PSO + MCMC" : "Hill Climbing + MCMC";
    
    // === LOGGER SETUP ===
    Logger::getInstance().setLogLevel(LogLevel::INFO);
    Logger::getInstance().info("main", "Starting Age-Structured SEPAIHRD Model Simulation...");
    Logger::getInstance().info("main", "Selected calibration algorithm: " + algorithm_name);
    cout << "Using calibration algorithm: " << algorithm_name << endl;
    
    try {
        // === FILE PATHS SETUP ===
        const int num_age_classes = epidemic::constants::DEFAULT_NUM_AGE_CLASSES;
        const int fixed_kappa_model_index = 0;
        const string project_root = FileUtils::getProjectRoot();
        Logger::getInstance().debug("main", "Project root: " + project_root);
        
        const string data_path = FileUtils::joinPaths(project_root, "data/processed/processed_data.csv");
        const string bounds_file = FileUtils::joinPaths(project_root, "data/configuration/param_bounds.txt");
        const string proposal_file = FileUtils::joinPaths(project_root, "data/configuration/proposal_sigmas.txt");
        const string params_to_calibrate_file = FileUtils::joinPaths(project_root, "data/configuration/params_to_calibrate.txt");
        const string init_guess_file = FileUtils::joinPaths(project_root, "data/configuration/initial_guess.txt");
        const string pso_settings_file = FileUtils::joinPaths(project_root, "data/configuration/pso_settings.txt");
        const string mcmc_settings_file = FileUtils::joinPaths(project_root, "data/configuration/mcmc_settings.txt");
        const string hill_climbing_settings_file = FileUtils::joinPaths(project_root, "data/configuration/hill_climbing_settings.txt");
        const string contact_matrix_path = FileUtils::joinPaths(project_root, "data/contacts.csv");
        const string output_dir = FileUtils::joinPaths(project_root, "data/output");
        
        // === DATA LOADING ===
        Logger::getInstance().info("main", "Loading calibration data from: " + data_path);
        CalibrationData data(data_path, "2020-03-01", "2020-12-31");
        Logger::getInstance().info("main", "Loading contact matrix from: " + contact_matrix_path);
        MatrixXd C = readMatrixFromCSV(contact_matrix_path, num_age_classes, num_age_classes);
        const auto N = data.getPopulationByAgeGroup();
        if (N.size() != num_age_classes) {
            throw DataFormatException("main", "Population data size mismatch");
        }

        // === PARAMETER SETUP ===
        epidemic::SEPAIHRDParameters params = readSEPAIHRDParameters(init_guess_file, num_age_classes);
        params.N = N;
        params.M_baseline = C;
        
        if (params.kappa_end_times.size() != 7 || params.kappa_values.size() != 7) {
            throw DataFormatException("main", "Invalid kappa parameters count - expected 7 each");
        }

        std::vector<std::string> all_kappa_parameter_names;
        for (size_t i = 0; i < params.kappa_values.size(); ++i) {
            all_kappa_parameter_names.push_back("kappa_" + std::to_string(i + 1));
        }

        std::map<std::string, std::pair<double, double>> overall_param_bounds = readParamBounds(bounds_file);
        std::map<std::string, double> proposal_sigmas = readProposalSigmas(proposal_file);
        std::vector<std::string> params_to_calibrate = readParamsToCalibrate(params_to_calibrate_file);

        // === SIMULATION SETUP ===
        vector<double> time_points;
        int num_days = data.getNumDataPoints();
        time_points.reserve(num_days);
        for (int t = 0; t < num_days; ++t) {
            time_points.push_back(static_cast<double>(t));
        }
        Logger::getInstance().info("main", "Simulation time points created (0 to " + std::to_string(num_days - 1) + ")");

        auto solver_strategy = std::make_shared<Dopri5SolverStrategy>();
        double dt_hint = 1.0;
        double abs_err = 1.0e-6;
        double rel_err = 1.0e-6;

        auto initial_state = data.getInitialSEPAIHRDState(
            params.sigma, params.gamma_p, params.gamma_A, 
            params.gamma_I, params.p, params.h
        );

        // Display initial state
        for (int i = 0; i < num_age_classes; ++i) {
            cout << "Initial state for age class " << i << ": "
                 << "S: " << initial_state[i] << ", "
                 << "E: " << initial_state[i + num_age_classes] << ", "
                 << "P: " << initial_state[i + 2 * num_age_classes] << ", "
                 << "A: " << initial_state[i + 3 * num_age_classes] << ", "
                 << "I: " << initial_state[i + 4 * num_age_classes] << ", "
                 << "H: " << initial_state[i + 5 * num_age_classes] << ", "
                 << "ICU: " << initial_state[i + 6 * num_age_classes] << ", "
                 << "R: " << initial_state[i + 7 * num_age_classes] << ", "
                 << "D: " << initial_state[i + 8 * num_age_classes]
                 << endl;
        }

        // === BASELINE SIMULATION ===
        auto npi_strategy = createNpiStrategy(params, all_kappa_parameter_names, overall_param_bounds, fixed_kappa_model_index);
        auto model = std::make_shared<AgeSEPAIHRDModel>(params, npi_strategy);
        
        AgeSEPAIHRDSimulator simulator(model, solver_strategy, time_points.front(), time_points.back(), dt_hint, abs_err, rel_err);
        Logger::getInstance().info("main", "Running baseline simulation with initial guess parameters...");
        SimulationResult baseline_result = simulator.run(initial_state, time_points);
        Logger::getInstance().info("main", "Baseline simulation finished.");
        
        // Display baseline results
        cout << "\n--- Baseline Simulation Results (Sample) ---" << endl;
        cout << "Time | I0 | H0 (First Age Class)" << std::endl;
        int i_idx = 4 * num_age_classes;
        int h_idx = 5 * num_age_classes;
        for (size_t i = 0; i < baseline_result.time_points.size(); i += std::max(1, num_days / 10)) {
            if (baseline_result.solution[i].size() > static_cast<size_t>(std::max(i_idx, h_idx))) {
                 std::cout << std::fixed << std::setprecision(1) << baseline_result.time_points[i] << " | "
                           << std::fixed << std::setprecision(0) << baseline_result.solution[i][i_idx] << " | "
                           << baseline_result.solution[i][h_idx] << std::endl;
            }
        }
        
        // Save baseline results
        FileUtils::ensureDirectoryExists(output_dir);
        string baseline_output_path = FileUtils::joinPaths(output_dir, "sepaihrd_age_baseline_results.csv");
        Logger::getInstance().info("main", "Saving baseline simulation results to: " + baseline_output_path);
        SimulationResultProcessor::saveResultsToCSV(baseline_result, *simulator.getTypedModel(), baseline_output_path);
        
        // === CALIBRATION SETUP ===
        Logger::getInstance().info("main", "Setting up calibration...");
        auto cache = std::make_shared<SimulationCache>();
        
        auto calibration_npi_strategy = createNpiStrategy(params, all_kappa_parameter_names, overall_param_bounds, fixed_kappa_model_index);
        auto calibration_model = std::make_shared<AgeSEPAIHRDModel>(params, calibration_npi_strategy);
        Logger::getInstance().info("main", "Calibration model and NPI strategy created.");

        SEPAIHRDModelCalibration calibration_setup(
            calibration_model, data, time_points, params_to_calibrate,
            proposal_sigmas, overall_param_bounds, solver_strategy, cache
        );

        // Load algorithm settings
        std::map<std::string, double> pso_settings = readParticleSwarmSettings(pso_settings_file);
        std::map<std::string, double> mcmc_settings = readMetropolisHastingsSettings(mcmc_settings_file);
        std::map<std::string, double> hill_climbing_settings = readHillClimbingSettings(hill_climbing_settings_file);

        // === RUN CALIBRATION ===
        Logger::getInstance().info("main", "Starting calibration with " + algorithm_name + "...");

        try {
            ModelCalibrator calibrator = [&]() -> ModelCalibrator {
                switch (selected_algorithm) {
                    case CalibrationAlgorithm::PSO_MCMC:
                        Logger::getInstance().info("main", "Running PSO + MCMC calibration...");
                        return calibration_setup.runPSOMCMC(pso_settings, mcmc_settings);
                        
                    case CalibrationAlgorithm::HILL_MCMC:
                        Logger::getInstance().info("main", "Running Hill Climbing + MCMC calibration...");
                        return calibration_setup.runHillClimbingMCMC(hill_climbing_settings, mcmc_settings);
                        
                    default:
                        throw std::runtime_error("Unknown calibration algorithm");
                }
            }();

            // === PROCESS CALIBRATION RESULTS ===
            const auto& best_params_vector = calibrator.getBestParameterVector();
            double best_obj_val = calibrator.getBestObjectiveValue();
            const auto& actual_calibrated_names = calibrator.getParameterManager().getParameterNames();
            const auto& mcmc_samples = calibrator.getMCMCSamples();
            
            Logger::getInstance().info("main", "Calibration finished.");
            std::cout << "\n--- Calibration Results ---" << std::endl;
            std::cout << "Best Objective Function Value (LogLikelihood): " << best_obj_val << std::endl;
            std::cout << "Number of MCMC samples collected: " << mcmc_samples.size() << std::endl;
            std::cout << "Best Parameters Found:" << std::endl;

            std::map<std::string, double> calibrated_param_map;
            if (static_cast<size_t>(best_params_vector.size()) == actual_calibrated_names.size()) {
                for (size_t i = 0; i < actual_calibrated_names.size(); ++i) {
                    std::cout << "  " << actual_calibrated_names[i] << ": " 
                             << std::setprecision(6) << best_params_vector[i] << std::endl;
                    calibrated_param_map[actual_calibrated_names[i]] = best_params_vector[i];
                }
            } else {
                std::cout << "  (Parameter names size mismatch - cannot display names)" << std::endl;
                std::cout << "  Vector: [" << best_params_vector.transpose() << "]" << std::endl;
            }

            // === FINAL SIMULATION WITH CALIBRATED PARAMETERS ===
            epidemic::SEPAIHRDParameters final_calibrated_params = calibration_model->getModelParameters();

            // Save calibrated parameters
            string calibrated_params_output_path = FileUtils::joinPaths(output_dir, "calibrated_parameters_final.txt");
            Logger::getInstance().info("main", "Saving calibrated parameters to: " + calibrated_params_output_path);
            saveCalibrationResults(
                calibrated_params_output_path, final_calibrated_params,
                actual_calibrated_names, best_obj_val, ""
            );
            
            // Create final simulation model and run
            auto final_npi_strategy = createNpiStrategy(final_calibrated_params, all_kappa_parameter_names, overall_param_bounds, fixed_kappa_model_index);
            auto final_calibrated_model = std::make_shared<AgeSEPAIHRDModel>(final_calibrated_params, final_npi_strategy);

            AgeSEPAIHRDSimulator final_simulator(final_calibrated_model, solver_strategy, time_points.front(), time_points.back(), dt_hint, abs_err, rel_err);
            Logger::getInstance().info("main", "Running simulation with calibrated parameters...");
            SimulationResult final_sim_result = final_simulator.run(initial_state, time_points);
            Logger::getInstance().info("main", "Final simulation finished.");
            
            string final_sim_output_path = FileUtils::joinPaths(output_dir, "sepaihrd_age_final_calibrated_run.csv");
            Logger::getInstance().info("main", "Saving final simulation results to: " + final_sim_output_path);
            SimulationResultProcessor::saveResultsToCSV(final_sim_result, *final_simulator.getTypedModel(), final_sim_output_path);

            // === REPRODUCTION NUMBER CALCULATIONS ===
            Logger::getInstance().info("main", "Calculating reproduction numbers...");
            try {
                epidemic::ReproductionNumberCalculator repro_calculator(final_calibrated_model);
                double R0 = repro_calculator.calculateR0();
                Logger::getInstance().info("main", "Calculated R0: " + std::to_string(R0));
                std::cout << "Calculated R0: " << R0 << std::endl;

                // Calculate Rt at t=0
                if (!final_sim_result.time_points.empty() && !final_sim_result.solution.empty()) {
                    Eigen::VectorXd S_at_t0(num_age_classes);
                    for(int i = 0; i < num_age_classes; ++i) {
                        S_at_t0(i) = initial_state(i);
                    }
                    double Rt_at_t0 = repro_calculator.calculateRt(S_at_t0, final_sim_result.time_points.front());
                    Logger::getInstance().info("main", "Calculated Rt(t=0): " + std::to_string(Rt_at_t0));
                    std::cout << "Calculated Rt(t=0): " << Rt_at_t0 << std::endl;
                }

                // Calculate Rt at midpoint
                if (final_sim_result.time_points.size() > 1) {
                    size_t midpoint_idx = final_sim_result.time_points.size() / 2;
                    double time_mid = final_sim_result.time_points[midpoint_idx];
                    Eigen::VectorXd S_at_midpoint(num_age_classes);
                    if (final_sim_result.solution[midpoint_idx].size() >= num_age_classes) {
                        for (int i = 0; i < num_age_classes; ++i) {
                            S_at_midpoint(i) = final_sim_result.solution[midpoint_idx][i];
                        }
                        double Rt_at_midpoint = repro_calculator.calculateRt(S_at_midpoint, time_mid);
                        Logger::getInstance().info("main", "Calculated Rt(t=" + std::to_string(time_mid) + "): " + std::to_string(Rt_at_midpoint));
                        std::cout << "Calculated Rt(t=" << time_mid << "): " << Rt_at_midpoint << std::endl;
                    }
                }
            } catch (const std::exception& e) {
                Logger::getInstance().error("main", "Failed to calculate reproduction numbers: " + std::string(e.what()));
                std::cerr << "Error calculating reproduction numbers: " << e.what() << std::endl;
            }
            // === POST-CALIBRATION ANALYSIS ===
            Logger::getInstance().info("main", "Starting comprehensive post-calibration analysis...");
            try {
                // Créer le modèle avec les paramètres finaux pour l'analyseur
                auto final_npi_strategy = createNpiStrategy(final_calibrated_params, all_kappa_parameter_names, overall_param_bounds);
                auto final_model_template = std::make_shared<AgeSEPAIHRDModel>(final_calibrated_params, final_npi_strategy);

                auto post_calibration_analyser = std::make_shared<PostCalibrationAnalyser>(
                    final_model_template,
                    solver_strategy,
                    time_points,
                    initial_state,
                    output_dir,
                    data
                );

                int total_sample_size = mcmc_samples.size();
                
                // Lire les paramètres pour l'analyse
                std::map<std::string, double> mcmc_settings = readMetropolisHastingsSettings(FileUtils::joinPaths(project_root, "data/configuration/mcmc_settings.txt"));
                auto get_setting = [&](const std::string& key, int default_val) {
                    auto it = mcmc_settings.find(key);
                    return (it != mcmc_settings.end()) ? static_cast<int>(it->second) : default_val;
                };

                int num_samples_for_ppc = get_setting("ppc_samples", total_sample_size);
                if (mcmc_samples.size() < static_cast<size_t>(num_samples_for_ppc)) {
                    num_samples_for_ppc = mcmc_samples.size();
                }
                int batch_size = get_setting("analysis_batch_size", 100);

                // Lancer le rapport complet optimisé
                post_calibration_analyser->generateFullReport(
                    mcmc_samples,
                    calibrator.getParameterManager(),
                    num_samples_for_ppc,
                    0,
                    1,
                    batch_size
                );

                std::cout << "\n=== POST-CALIBRATION ANALYSIS COMPLETED SUCCESSFULLY ===" << std::endl;
                std::cout << "Results saved to: " << output_dir << std::endl;

            } catch (const std::exception& e) {
                Logger::getInstance().error("main", "Post-calibration analysis failed: " + std::string(e.what()));
                cerr << "Error during post-calibration analysis: " << e.what() << endl;
            }
        } catch (const std::exception& e) {
            Logger::getInstance().error("main", "Calibration failed: " + std::string(e.what()));
            cerr << "Error during calibration: " << e.what() << endl;
            return 1;
        }
        
        Logger::getInstance().info("main", "Simulation and Calibration completed successfully.");
        return 0;
    }
    
    // === EXCEPTION HANDLING ===
    catch (const epidemic::FileIOException& e) {
        Logger::getInstance().fatal("main", "File IO Error: " + std::string(e.what()));
        cerr << "Critical Error: File operation failed. " << e.what() << endl;
        return 1;
    }
    catch (const epidemic::CSVReadException& e) {
        Logger::getInstance().fatal("main", "CSV Read Error: " + std::string(e.what()));
        cerr << "Critical Error: Failed to read CSV data. " << e.what() << endl;
        return 1;
    }
    catch (const epidemic::DataFormatException& e) {
        Logger::getInstance().fatal("main", "Data Format Error: " + std::string(e.what()));
        cerr << "Critical Error: Invalid data format encountered. " << e.what() << endl;
        return 1;
    }
    catch (const epidemic::ModelConstructionException& e) {
        Logger::getInstance().fatal("main", "Model Setup Error: " + std::string(e.what()));
        cerr << "Critical Error: Failed to set up the model. " << e.what() << endl;
        return 1;
    }
    catch (const epidemic::InvalidParameterException& e) {
        Logger::getInstance().fatal("main", "Invalid Parameter Error: " + std::string(e.what()));
        cerr << "Critical Error: Invalid parameter provided. " << e.what() << endl;
        return 1;
    }
    catch (const epidemic::SimulationException& e) {
        Logger::getInstance().fatal("main", "Simulation Error: " + std::string(e.what()));
        cerr << "Critical Error: Simulation failed to run or produced invalid results. " << e.what() << endl;
        return 1;
    }
    catch (const epidemic::InterventionException& e) {
        Logger::getInstance().fatal("main", "Intervention Error: " + std::string(e.what()));
        cerr << "Critical Error: Failed to apply or define intervention. " << e.what() << endl;
        return 1;
    }
    catch (const epidemic::ModelException& e) {
        Logger::getInstance().fatal("main", "General Model Error: " + std::string(e.what()));
        cerr << "Critical Error: An unspecified model error occurred. " << e.what() << endl;
        return 1;
    }
    catch (const std::exception& e) {
        Logger::getInstance().fatal("main", "Standard Exception: " + std::string(e.what()));
        cerr << "Error: An unexpected error occurred: " << e.what() << endl;
        return 1;
    }
    catch (...) {
        Logger::getInstance().fatal("main", "Unknown exception caught.");
        cerr << "Error: An unknown error occurred." << endl;
        return 1;
    }
}