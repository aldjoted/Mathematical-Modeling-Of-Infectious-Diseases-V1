#include <iostream>
#include <fstream>
#include <memory>
#include <filesystem>
#include <iomanip>
#include <vector>
#include <string>
#include <map>

#include "sir_age_structured/parameters/SIRParamerterManager.hpp"
#include "sir_age_structured/objectives/PoissonLikeLihoodObjective.hpp"
#include "sir_age_structured/solvers/Dropri5SolverStrategy.hpp"
#include "sir_age_structured/optimizers/HillClimbingOptimizer.hpp"
#include "sir_age_structured/optimizers/MetropolisHastingsSampler.hpp"
#include "sir_age_structured/interfaces/ISimulationCache.hpp"
#include "sir_age_structured/caching/SimulationCache.hpp"
#include "sir_age_structured/ModelFactory.hpp"
#include "sir_age_structured/ModelCalibrator.hpp"
#include "sir_age_structured/Simulator.hpp"
#include "utils/FileUtils.hpp"
#include "utils/ReadContactMatrix.hpp"
#include "utils/GetCalibrationData.hpp"
#include "exceptions/Exceptions.hpp"
#include "utils/Logger.hpp"

using namespace Eigen;
using namespace std;
using namespace epidemic;

int main() {
    Logger::getInstance().setLogLevel(LogLevel::INFO);
    Logger::getInstance().info("calibration_demo", "Starting Age-Structured SIR Model Calibration Demo...");

    try {
        // --- Configuration ---
        int num_age_classes = 4;
        string project_root = FileUtils::getProjectRoot();
        Logger::getInstance().debug("calibration_demo", "Project root: " + project_root);

        // --- Load Data ---
        string contacts_filename = FileUtils::joinPaths(project_root, "data/contacts.csv");
        Logger::getInstance().info("calibration_demo", "Loading contact matrix from: " + contacts_filename);
        MatrixXd C = readMatrixFromCSV(contacts_filename, num_age_classes, num_age_classes);

        string data_filename = FileUtils::joinPaths(project_root, "data/processed/processed_data.csv");
        Logger::getInstance().info("calibration_demo", "Loading calibration data from: " + data_filename);
        // Assuming default date range or no filtering needed for calibration demo
        CalibrationData calibData(data_filename);
        MatrixXd observed_incidence = calibData.getNewConfirmedCases();

        // --- Initial Model Setup ---
        VectorXd population = calibData.getPopulationByAgeGroup();
        VectorXd initial_gamma(num_age_classes);
        initial_gamma << 0.1, 0.1, 0.1, 0.1; // Example initial values
        double initial_q = 0.1;             // Example initial value
        double initial_scale_C = 1.0;       // Example initial value

        Logger::getInstance().info("calibration_demo", "Creating initial AgeSIRModel instance...");
        auto model = ModelFactory::createAgeSIRModel(population, C, initial_gamma, initial_q, initial_scale_C);

        VectorXd I0 = calibData.getInitialActiveCases();
        VectorXd R0 = VectorXd::Zero(num_age_classes);
        VectorXd S0 = population - I0;
        auto initial_state = ModelFactory::createInitialSIRState(S0, I0, R0);

        // --- Calibration Setup ---
        vector<double> time_points;
        int num_days = calibData.getNumDataPoints(); // Use data length for time points
        time_points.reserve(num_days);
        for (int t = 0; t < num_days; ++t) { // Start from t=0 to match simulation convention
            time_points.push_back(static_cast<double>(t));
        }

        // Define parameters to calibrate and their proposal sigmas
        std::vector<std::string> params_to_calibrate = {"q", "scale_C_total"};
        std::map<std::string, double> proposal_sigmas = {{"q", 0.01}, {"scale_C_total", 0.05}};
        for (int i = 0; i < num_age_classes; ++i) {
            std::string gamma_name = "gamma_" + std::to_string(i);
            params_to_calibrate.push_back(gamma_name);
            proposal_sigmas[gamma_name] = 0.01;
        }

        Logger::getInstance().info("calibration_demo", "Instantiating calibration components...");
        // Ensure SIRParameterManager exists and is correctly implemented
        auto parameterManager = std::make_unique<SIRParameterManager>(model, params_to_calibrate, proposal_sigmas);

        auto solver_strategy = std::make_shared<Dopri5SolverStrategy>();

        double dt_hint = time_points.size() > 1 ? time_points[1] - time_points[0] : 1.0;
        double abs_err = 1.0e-6;
        double rel_err = 1.0e-6;
        // Create a simulator instance specifically for the objective function evaluation
        Simulator objectiveSimulator(model, solver_strategy, time_points.front(), time_points.back(), dt_hint, abs_err, rel_err);

        auto simulationCache = std::make_unique<SimulationCache>();

        // Ensure PoissonLikelihoodObjective exists and is correctly implemented
        auto objectiveFunction = std::make_unique<PoissonLikelihoodObjective>(
            model,
            objectiveSimulator,
            *parameterManager,
            *simulationCache,
            calibData,
            time_points,
            initial_state
        );

        auto phase1_algo = std::make_unique<HillClimbingOptimizer>();
        auto phase2_algo = std::make_unique<MetropolisHastingsSampler>();

        std::map<std::string, std::unique_ptr<IOptimizationAlgorithm>> optimization_algorithms;
        // Use the constants defined in ModelCalibrator
        optimization_algorithms[ModelCalibrator::PHASE1_NAME] = std::move(phase1_algo);
        optimization_algorithms[ModelCalibrator::PHASE2_NAME] = std::move(phase2_algo);

        Logger::getInstance().info("calibration_demo", "Creating ModelCalibrator...");
        ModelCalibrator calibrator(
            std::move(parameterManager),
            std::move(objectiveFunction),
            std::move(optimization_algorithms),
            calibData,
            time_points);

        // --- Calibration Settings ---
        int bestFitIterations = 10000; // Example value
        double initialStep = 1.0;    // Example value
        double coolingRate = 0.995;    // Example value

        int burnIn = 200;         // Example value
        int mcmcIterations = 10000;    // Example value
        int thinning = 1;           // Example value
        double mcmcStepSize = 0.01;   // Example value

        std::map<std::string, double> phase1_settings = {
            {"iterations", static_cast<double>(bestFitIterations)},
            {"initial_step", initialStep}, // Corrected key based on HillClimbingOptimizer
            {"cooling_rate", coolingRate},
            {"refinement_steps", 5.0},     // Example value
            {"report_interval", 100.0}     // Example value
        };
        std::map<std::string, double> phase2_settings = {
            {"burn_in", static_cast<double>(burnIn)},
            {"mcmc_iterations", static_cast<double>(mcmcIterations)},
            {"thinning", static_cast<double>(thinning)},
            {"mcmc_step_size", mcmcStepSize}, // Corrected key based on MetropolisHastingsSampler
            {"report_interval", 200.0},    // Example value
            {"calculate_posterior_mean", 1.0} // Enable posterior mean calculation
        };

        Logger::getInstance().info("calibration_demo", "Starting model calibration process...");
        cout << "Calibration Settings:" << endl;
        cout << "  Phase 1 (" << ModelCalibrator::PHASE1_NAME << ") Iterations: " << bestFitIterations << endl;
        cout << "  Phase 1 Initial Step: " << initialStep << endl;
        cout << "  Phase 1 Cooling Rate: " << coolingRate << endl;
        cout << "  Phase 2 (" << ModelCalibrator::PHASE2_NAME << ") Burn-in: " << burnIn << endl;
        cout << "  Phase 2 Sampling Iterations: " << mcmcIterations << endl;
        cout << "  Phase 2 Thinning: " << thinning << endl;
        cout << "  Phase 2 Step Size (nominal): " << mcmcStepSize << endl;

        calibrator.calibrate(phase1_settings, phase2_settings);

        Logger::getInstance().info("calibration_demo", "Calibration finished.");

        // --- Display Final Results ---
        const auto& best_params_vector = calibrator.getBestParameterVector();
        // Get names from the manager *within* the calibrator after move
        const auto& param_names = calibrator.getParameterManager().getParameterNames();
        double best_objective_value = calibrator.getBestObjectiveValue();

        cout << "\n--- Final Calibration Results ---" << endl;
        cout << std::fixed << std::setprecision(6);
        cout << "Best Parameters:" << endl;
        if (best_params_vector.size() == static_cast<Eigen::Index>(param_names.size())) {
            for (size_t i = 0; i < param_names.size(); ++i) {
                cout << "  " << param_names[i] << ": " << best_params_vector[i] << endl;
            }
        } else {
             Logger::getInstance().error("calibration_demo", "Mismatch between parameter names and best parameter vector size.");
             cout << "  Best Parameters Vector: " << best_params_vector.transpose() << endl;
        }
        cout << "Best Objective Value:        " << best_objective_value << endl;

        // --- Save MCMC Samples ---
        string output_dir = FileUtils::joinPaths(project_root, "data/calibration_output");
        Logger::getInstance().info("calibration_demo", "Creating output directory: " + output_dir);
        FileUtils::ensureDirectoryExists(output_dir);

        const auto& mcmc_samples = calibrator.getMCMCSamples();
        const auto& mcmc_obj_values = calibrator.getMCMCObjectiveValues();
        if (!mcmc_samples.empty()) {
            string mcmc_output_file = FileUtils::joinPaths(output_dir, "mcmc_samples.csv");
            Logger::getInstance().info("calibration_demo", "Saving MCMC samples to: " + mcmc_output_file);
            ofstream mcmc_out(mcmc_output_file);
            if (!mcmc_out) {
                throw FileIOException("calibration_demo", "Failed to open MCMC output file: " + mcmc_output_file);
            }

            mcmc_out << "sample_index,objective_value";
            // Use the same names as displayed above
            for (const auto& name : param_names) {
                mcmc_out << "," << name;
            }
            mcmc_out << "\n";

            mcmc_out << std::fixed << std::setprecision(8);
            for (size_t i = 0; i < mcmc_samples.size(); ++i) {
                mcmc_out << i << "," << mcmc_obj_values[i];
                if (mcmc_samples[i].size() == static_cast<Eigen::Index>(param_names.size())) {
                    for (int j = 0; j < mcmc_samples[i].size(); ++j) {
                        mcmc_out << "," << mcmc_samples[i][j];
                    }
                } else {
                     Logger::getInstance().warning("calibration_demo", "MCMC sample " + std::to_string(i) + " size mismatch.");
                     // Output raw values if size mismatches
                     for (int j = 0; j < mcmc_samples[i].size(); ++j) {
                         mcmc_out << "," << mcmc_samples[i][j];
                     }
                }
                mcmc_out << "\n";
            }
            mcmc_out.close();
            Logger::getInstance().info("calibration_demo", "MCMC samples saved.");
        } else {
             Logger::getInstance().info("calibration_demo", "No MCMC samples generated (Phase 2 might have been skipped or failed).");
        }


        // --- Run Final Simulation with Best Parameters ---
        Logger::getInstance().info("calibration_demo", "Running final simulation with best parameters...");
        // The 'model' shared pointer inside the calibrator's ParameterManager
        // should have been updated with the best parameters.
        // We can re-use the objectiveSimulator as its internal model shared_ptr
        // points to the same updated model instance.
        SimulationResult final_result = objectiveSimulator.run(initial_state, time_points);
        Logger::getInstance().info("calibration_demo", "Final simulation finished.");

        // --- Save Final Simulated Data ---
        // Extract simulated incidence (assuming I is the second compartment group)
        int n_ages = model->getNumAgeClasses();
        int n_times = final_result.time_points.size();
        MatrixXd simulated_incidence_final(n_times, n_ages);
        int i_compartment_offset = n_ages; // S is 0*n_ages, I is 1*n_ages

        for(int i = 0; i < n_times; ++i) {
            const auto& state_vec = final_result.solution[i];
            if (state_vec.size() != static_cast<size_t>(model->getStateSize())) {
                throw InvalidResultException("calibration_demo", "Simulation result state vector size mismatch during final run.");
            }
            // Extract the I compartment values for each age group
            for (int j = 0; j < n_ages; ++j) {
                simulated_incidence_final(i, j) = state_vec[i_compartment_offset + j];
            }
        }

        string sim_output_file = FileUtils::joinPaths(output_dir, "simulated_incidence_best_fit.csv");
        Logger::getInstance().info("calibration_demo", "Saving best-fit simulated incidence to: " + sim_output_file);

        ofstream sim_out(sim_output_file);
        if (!sim_out) {
            throw FileIOException("calibration_demo", "Failed to open simulation output file: " + sim_output_file);
        }
        sim_out << "Time";
        // Use generic age labels or define specific ones if known
        vector<string> age_labels = {"0_30", "30_60", "60_80", "80_plus"}; // Example labels
        if (static_cast<int>(age_labels.size()) != num_age_classes) {
            Logger::getInstance().warning("calibration_demo", "Number of age labels doesn't match num_age_classes. Using generic headers.");
            for (int j = 0; j < num_age_classes; ++j) sim_out << ",simulated_I_" << j;
        } else {
            for (int j = 0; j < num_age_classes; ++j) sim_out << ",simulated_I_" << age_labels[j];
        }
        sim_out << "\n";

        sim_out << std::fixed << std::setprecision(4);
        for (int i = 0; i < simulated_incidence_final.rows(); ++i) {
            sim_out << final_result.time_points[i];
            for (int j = 0; j < simulated_incidence_final.cols(); ++j) {
                sim_out << "," << simulated_incidence_final(i, j);
            }
            sim_out << "\n";
        }
        sim_out.close();
        Logger::getInstance().info("calibration_demo", "Best-fit simulated incidence saved.");

        Logger::getInstance().info("calibration_demo", "Calibration demo completed successfully.");
        return 0;

    } catch (const epidemic::ModelException& e) {
        Logger::getInstance().fatal("calibration_demo", "Epidemic Exception: " + std::string(e.what()));
        cerr << "Critical Error: " << e.what() << endl;
        return 1;
    } catch (const std::exception& e) {
        Logger::getInstance().fatal("calibration_demo", "Standard Exception: " + std::string(e.what()));
        cerr << "Critical Error: An unexpected error occurred: " << e.what() << endl;
        return 1;
    } catch (...) {
        Logger::getInstance().fatal("calibration_demo", "Unknown exception caught.");
        cerr << "Critical Error: An unknown error occurred." << endl;
        return 1;
    }
}