#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <iomanip>
#include <Eigen/Dense>

#include "sir_age_structured/AgeSIRModel.hpp"
#include "sir_age_structured/ModelFactory.hpp"
#include "sir_age_structured/Simulator.hpp"
#include "sir_age_structured/SimulationResult.hpp"
#include "sir_age_structured/solvers/Dropri5SolverStrategy.hpp"
#include "sir_age_structured/InterventionCallBack.hpp"
#include "sir_age_structured/SimulationResultProcessor.hpp"
#include "utils/FileUtils.hpp"
#include "utils/ReadContactMatrix.hpp"
#include "utils/GetCalibrationData.hpp"
#include "exceptions/Exceptions.hpp"
#include "exceptions/CSVReadException.hpp"
#include "utils/Logger.hpp"

using namespace Eigen;
using namespace std;
using namespace epidemic;

int main() {
    Logger::getInstance().setLogLevel(LogLevel::INFO);
    Logger::getInstance().info("main", "Starting Age-Structured SIR Model Simulation...");

    try {
        const int num_age_classes = 4;
        const string project_root = FileUtils::getProjectRoot();
        Logger::getInstance().debug("main", "Project root: " + project_root);

        const string data_path = FileUtils::joinPaths(project_root, "data/processed/processed_data.csv");
        Logger::getInstance().info("main", "Loading calibration data from: " + data_path);
        CalibrationData data(data_path, "2020-03-01", "2020-12-31");

        const string contact_matrix_path = FileUtils::joinPaths(project_root, "data/contacts.csv");
        Logger::getInstance().info("main", "Loading contact matrix from: " + contact_matrix_path);
        MatrixXd C = readMatrixFromCSV(contact_matrix_path, num_age_classes, num_age_classes);

        // --- Initial Conditions & Parameters ---
        const auto N = data.getPopulationByAgeGroup();
        const auto I0 = data.getInitialActiveCases(); 
        const VectorXd R0 = VectorXd::Zero(num_age_classes);
        const VectorXd S0 = N - I0;
        Logger::getInstance().debug("main", "Initial S0: " + std::to_string(S0.sum()));
        Logger::getInstance().debug("main", "Initial I0: " + std::to_string(I0.sum()));

        VectorXd gamma(num_age_classes);
        gamma << 0.20, 0.21, 0.31, 0.13; 
        const double q = 0.15;           
        const double scale_C = 1.0;     
        Logger::getInstance().info("main", "Initial parameters set (q=" + std::to_string(q) + ")");

        // --- Simulation Time Points ---
        vector<double> time_points;
        int num_days = data.getNumDataPoints();
        time_points.reserve(num_days);
        for (int t = 0; t < num_days; ++t) {
            time_points.push_back(static_cast<double>(t));
        }
        Logger::getInstance().info("main", "Simulation time points created (0 to " + std::to_string(num_days - 1) + ")");

        // --- Model & Solver Setup ---
        auto sir_model = ModelFactory::createAgeSIRModel(N, C, gamma, q, scale_C);
        Logger::getInstance().info("main", "AgeSIRModel created.");

        // Ensure Dopri5SolverStrategy exists and is included correctly
        auto solver_strategy = std::make_shared<Dopri5SolverStrategy>();
        double dt_hint = time_points.size() > 1 ? time_points[1] - time_points[0] : 1.0;
        double abs_err = 1.0e-8;
        double rel_err = 1.0e-8;

        Simulator simulator(sir_model, solver_strategy, time_points.front(), time_points.back(), dt_hint, abs_err, rel_err);
        Logger::getInstance().info("main", "Simulator configured (Solver: Dopri5, Tol: " + std::to_string(abs_err) + ").");

        // --- Baseline Simulation ---
        auto initial_state = ModelFactory::createInitialSIRState(S0, I0, R0);
        Logger::getInstance().info("main", "Running baseline simulation...");
        SimulationResult baseline_result = simulator.run(initial_state, time_points);
        Logger::getInstance().info("main", "Baseline simulation finished.");

        // --- Baseline Results Output (Console) ---
        cout << "\n--- Baseline Simulation Results (Sample) ---" << endl;
        cout << "Time | S0 | I0 | R0 (First Age Class)" << std::endl;
        int s_idx = 0;
        int i_idx = num_age_classes;
        int r_idx = 2 * num_age_classes;
        for (size_t i = 0; i < baseline_result.time_points.size(); i += std::max(1, num_days / 10)) {
            if (baseline_result.solution[i].size() > static_cast<size_t>(std::max({s_idx, i_idx, r_idx}))) {
                 std::cout << std::fixed << std::setprecision(1) << baseline_result.time_points[i] << " | "
                           << std::fixed << std::setprecision(0) << baseline_result.solution[i][s_idx] << " | "
                           << baseline_result.solution[i][i_idx] << " | " << baseline_result.solution[i][r_idx] << std::endl;
            }
        }

        // --- Intervention Simulation Setup ---
        Logger::getInstance().info("main", "Setting up intervention simulation...");
        auto intervention_model = ModelFactory::createAgeSIRModel(N, C, gamma, q, scale_C);
        Simulator intervention_sim(intervention_model, solver_strategy, time_points.front(), time_points.back(), dt_hint, abs_err, rel_err);

        double intervention_time = 20.0;
        Eigen::VectorXd intervention_params(1);
        intervention_params << 0.7;
        std::string intervention_name = "contact_reduction";

        // --- Run Simulation with Intervention ---
        std::vector<double> pre_intervention_times;
        std::vector<double> post_intervention_times;
        for(double t : time_points) {
            if (t <= intervention_time) {
                pre_intervention_times.push_back(t);
            }
            if (t >= intervention_time) {
                 post_intervention_times.push_back(t);
            }
        }

        SimulationResult intervention_result;
        Eigen::VectorXd state_at_intervention;

        // Run pre-intervention part
        if (!pre_intervention_times.empty()) {
            Logger::getInstance().info("main", "Running pre-intervention segment (0 to " + std::to_string(intervention_time) + ")...");
            SimulationResult pre_result = intervention_sim.run(initial_state, pre_intervention_times);
            intervention_result = pre_result; 
            state_at_intervention = Eigen::Map<const Eigen::VectorXd>(pre_result.solution.back().data(), pre_result.solution.back().size());
            Logger::getInstance().info("main", "Pre-intervention segment finished.");
        } else {
             state_at_intervention = initial_state;
             Logger::getInstance().warning("main", "Intervention time is at or before the first time point. Applying immediately.");
        }

        // Apply intervention and run post-intervention part
        if (!post_intervention_times.empty()) {
             Logger::getInstance().info("main", "Applying intervention '" + intervention_name + "' at t=" + std::to_string(intervention_time) + " and running post-intervention segment...");
             intervention_model->applyIntervention(intervention_name, intervention_time, intervention_params);
             Logger::getInstance().info("main", "Intervention applied to model state.");

             // Remove the duplicate time point if present
             if (!intervention_result.time_points.empty() && !post_intervention_times.empty() &&
                 std::abs(intervention_result.time_points.back() - post_intervention_times.front()) < 1e-9) {
                 post_intervention_times.erase(post_intervention_times.begin());
             }

             // Only run if there are remaining time points
             if (!post_intervention_times.empty()) {
                 // Run the simulation starting from the state *at* the intervention time
                 SimulationResult post_result = intervention_sim.run(state_at_intervention, post_intervention_times);

                 intervention_result.time_points.insert(intervention_result.time_points.end(),
                                                        post_result.time_points.begin(), post_result.time_points.end());
                 intervention_result.solution.insert(intervention_result.solution.end(),
                                                     post_result.solution.begin(), post_result.solution.end());
                 Logger::getInstance().info("main", "Post-intervention segment finished.");
             } else {
                 Logger::getInstance().info("main", "No further time points after intervention time.");
             }

        } else {
             Logger::getInstance().info("main", "No post-intervention time points to simulate.");
        }


        Logger::getInstance().info("main", "Intervention simulation finished.");

        // --- Intervention Results Output (Console) ---
        cout << "\n--- Intervention Simulation Results (Sample) ---" << std::endl;
        cout << "Time | S0 | I0 | R0 (First Age Class)" << std::endl;
        for (size_t i = 0; i < intervention_result.time_points.size(); i += std::max(1, num_days / 10)) {
             if (intervention_result.solution[i].size() > static_cast<size_t>(std::max({s_idx, i_idx, r_idx}))) {
                 cout << std::fixed << std::setprecision(1) << intervention_result.time_points[i]  << " | "
                      << std::fixed << std::setprecision(0) << intervention_result.solution[i][s_idx] << " | "
                      << intervention_result.solution[i][i_idx] << " | "
                      << intervention_result.solution[i][r_idx] << std::endl;
             }
        }

        // --- Save Results to CSV ---
        const string output_dir = FileUtils::joinPaths(project_root, "data/output");
        FileUtils::ensureDirectoryExists(output_dir);
        string baseline_output_path = FileUtils::joinPaths(output_dir, "sir_age_baseline_results.csv");
        string intervention_output_path = FileUtils::joinPaths(output_dir, "sir_age_intervention_results.csv");

        Logger::getInstance().info("main", "Saving baseline results to: " + baseline_output_path);
        SimulationResultProcessor::saveResultsToCSV(baseline_result, *simulator.getModel(), baseline_output_path);

        Logger::getInstance().info("main", "Saving intervention results to: " + intervention_output_path);
        SimulationResultProcessor::saveResultsToCSV(intervention_result, *intervention_sim.getModel(), intervention_output_path);

        Logger::getInstance().info("main", "Simulation completed successfully.");
        return 0;
    }
    // --- Exception Handling ---
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