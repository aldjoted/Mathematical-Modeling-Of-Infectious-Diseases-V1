#include "sir_age_structured/ModelCalibrator.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <iomanip>

namespace epidemic {

    ModelCalibrator::ModelCalibrator(
        std::unique_ptr<IParameterManager> parameterManager,
        std::unique_ptr<IObjectiveFunction> objectiveFunction,
        std::map<std::string, std::unique_ptr<IOptimizationAlgorithm>> optimization_algorithms,
        const epidemic::CalibrationData& calibration_data,  // Use epidemic:: prefix
        const std::vector<double>& time_points)
        : parameterManager_(std::move(parameterManager)),
          objectiveFunction_(std::move(objectiveFunction)),
          optimization_algorithms_(std::move(optimization_algorithms)),
          observed_data_(calibration_data),
          time_points_(time_points)
    {
        if (!parameterManager_) {
            THROW_INVALID_PARAM("ModelCalibrator", "ParameterManager cannot be null.");
        }
        if (!objectiveFunction_) {
            THROW_INVALID_PARAM("ModelCalibrator", "ObjectiveFunction cannot be null.");
        }
        if (optimization_algorithms_.empty()) {
            THROW_INVALID_PARAM("ModelCalibrator", "At least one optimization algorithm must be provided.");
        }
        if (parameterManager_->getParameterNames() != objectiveFunction_->getParameterNames()) {
             THROW_INVALID_PARAM("ModelCalibrator", "Parameter names mismatch between ParameterManager and ObjectiveFunction.");
        }
    
        best_params_vector_ = parameterManager_->getCurrentParameters();
        best_objective_value_ = objectiveFunction_->calculate(best_params_vector_);
    
        if (std::isnan(best_objective_value_) || std::isinf(best_objective_value_)) {
             std::cerr << "[Calibrator] Warning: Initial parameters yield invalid objective value (" << best_objective_value_ << "). Starting optimization with default best." << std::endl;
             best_objective_value_ = -std::numeric_limits<double>::infinity();
        } else {
             std::cout << "Initial Objective Value: " << best_objective_value_ << std::endl;
        }
    }
    
    void ModelCalibrator::calibrate(const std::map<std::string, double>& phase1_settings,
                                   const std::map<std::string, double>& phase2_settings)
    {
        std::cout << "\n=== Starting Full Calibration Procedure ===" << std::endl;
        std::cout << "Parameters to calibrate: ";
        for(const auto& name : parameterManager_->getParameterNames()) std::cout << name << " ";
        std::cout << std::endl;
    
        Eigen::VectorXd current_best_params = best_params_vector_;
    
        auto it_phase1 = optimization_algorithms_.find(PHASE1_NAME);
        if (it_phase1 != optimization_algorithms_.end()) {
            std::cout << "\n--- Running Phase 1: " << PHASE1_NAME << " ---" << std::endl;
            IOptimizationAlgorithm* phase1_algo = it_phase1->second.get();
            phase1_algo->configure(phase1_settings);
            phase1_result_ = phase1_algo->optimize(current_best_params, *objectiveFunction_, *parameterManager_);
    
            if (phase1_result_.bestObjectiveValue > best_objective_value_) {
                best_objective_value_ = phase1_result_.bestObjectiveValue;
                best_params_vector_ = phase1_result_.bestParameters;
                std::cout << "Phase 1 New Best Objective: " << best_objective_value_ << std::endl;
            }
            current_best_params = best_params_vector_;
            std::cout << "--- Phase 1 Completed ---" << std::endl;
        } else {
            std::cout << "Skipping Phase 1: Algorithm '" << PHASE1_NAME << "' not provided." << std::endl;
        }
    
        auto it_phase2 = optimization_algorithms_.find(PHASE2_NAME);
        if (it_phase2 != optimization_algorithms_.end()) {
            std::cout << "\n--- Running Phase 2: " << PHASE2_NAME << " ---" << std::endl;
            IOptimizationAlgorithm* phase2_algo = it_phase2->second.get();
            phase2_algo->configure(phase2_settings);
            phase2_result_ = phase2_algo->optimize(current_best_params, *objectiveFunction_, *parameterManager_);
    
            if (phase2_result_.bestObjectiveValue > best_objective_value_) {
                best_objective_value_ = phase2_result_.bestObjectiveValue;
                best_params_vector_ = phase2_result_.bestParameters;
                 std::cout << "Phase 2 New Best Objective (from samples/mean): " << best_objective_value_ << std::endl;
            }
    
            for (const auto& mcmcSample : phase2_result_.samples) {
                double objectiveValue = objectiveFunction_->calculate(mcmcSample);
                mcmcObjectiveValues_.push_back(objectiveValue);
            }

            std::cout << "--- Phase 2 Completed ---" << std::endl;
        } else {
            std::cout << "Skipping Phase 2: Algorithm '" << PHASE2_NAME << "' not provided." << std::endl;
        }
    
        parameterManager_->updateModelParameters(best_params_vector_);
    
        std::cout << "\n=== Calibration Procedure Finished ===" << std::endl;
        std::cout << std::fixed << std::setprecision(6); 
        std::cout << "Overall Best Parameters Found (Vector): [" << best_params_vector_.transpose() << "]" << std::endl;
        std::cout << "Overall Best Parameters Found (Named):" << std::endl;
        const auto& names = parameterManager_->getParameterNames();
        for(size_t i=0; i < names.size(); ++i) {
            std::cout << "  " << names[i] << " = " << best_params_vector_[i] << std::endl;
        }
        std::cout << "Best Objective Value: " << best_objective_value_ << std::endl;
        if (!phase2_result_.samples.empty()) {
            std::cout << "MCMC Samples Collected: " << phase2_result_.samples.size() << std::endl;
        }
    }
    
    const std::vector<Eigen::VectorXd>& ModelCalibrator::getMCMCSamples() const {
        return phase2_result_.samples;
    }
    
}