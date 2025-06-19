#include "model/SEPAIHRDModelCalibration.hpp"
#include "model/parameters/SEPAIHRDParameterManager.hpp"
#include "model/objectives/SEPAIHRDObjectiveFunction.hpp"
#include "exceptions/Exceptions.hpp"
#include <iostream>
#include <utility>

namespace epidemic {

SEPAIHRDModelCalibration::SEPAIHRDModelCalibration(
    std::shared_ptr<AgeSEPAIHRDModel> model_ptr,
    const epidemic::CalibrationData& calibration_data,  // Note the :: prefix
    const std::vector<double>& time_points,
    const std::vector<std::string>& params_to_calibrate,
    const std::map<std::string, double>& proposal_sigmas,
    const std::map<std::string, std::pair<double, double>>& param_bounds,
    std::shared_ptr<IOdeSolverStrategy> solver_strategy,
    std::shared_ptr<ISimulationCache> cache)
    : model_(model_ptr),
      observed_data_(calibration_data),
      time_points_(time_points),
      params_to_calibrate_(params_to_calibrate),
      proposal_sigmas_(proposal_sigmas),
      param_bounds_(param_bounds),
      solver_strategy_(solver_strategy),
      cache_(cache)
{
    if (!model_) THROW_INVALID_PARAM("SEPAIHRDModelCalibration", "Model pointer cannot be null.");
    if (time_points_.empty()) THROW_INVALID_PARAM("SEPAIHRDModelCalibration", "Time points vector cannot be empty.");
    if (params_to_calibrate_.empty()) THROW_INVALID_PARAM("SEPAIHRDModelCalibration", "Parameters to calibrate list cannot be empty.");
    if (!solver_strategy_) THROW_INVALID_PARAM("SEPAIHRDModelCalibration", "Solver strategy cannot be null.");
    if (!cache_) THROW_INVALID_PARAM("SEPAIHRDModelCalibration", "Cache cannot be null.");

    try {
        const auto& model_params = model_->getModelParameters();
        initial_state_cached_ = observed_data_.getInitialSEPAIHRDState(
            model_params.sigma,
            model_params.gamma_p,
            model_params.gamma_A,
            model_params.gamma_I,
            model_params.p,    
            model_params.h     
        );
        if (initial_state_cached_.size() == 0) {
             THROW_INVALID_PARAM("SEPAIHRDModelCalibration", "Initial state derived from calibration data is empty.");
        }

        try {
            parameter_manager_ = std::make_unique<SEPAIHRDParameterManager>(
                model_, params_to_calibrate_, proposal_sigmas_, param_bounds_);
        } catch (const std::exception& e) {
            throw ModelConstructionException("SEPAIHRDModelCalibration", 
                                            std::string("Failed to create parameter manager: ") + e.what());
        }

    } catch (const std::exception& e) {
         throw ModelConstructionException("SEPAIHRDModelCalibration", std::string("Failed to get initial state from calibration data: ") + e.what());
    }
}

ModelCalibrator SEPAIHRDModelCalibration::setupCalibrator(
    std::map<std::string, std::unique_ptr<IOptimizationAlgorithm>> algorithms)
{

    std::unique_ptr<SEPAIHRDParameterManager> parameterManager;
    try {
        parameterManager = std::make_unique<SEPAIHRDParameterManager>(
            model_, params_to_calibrate_, proposal_sigmas_, param_bounds_);
    } catch (const std::exception& e) {
        throw ModelConstructionException("SEPAIHRDModelCalibration::setupCalibrator",
                                         std::string("Failed to create SEPAIHRDParameterManager: ") + e.what());
    }

    std::unique_ptr<SEPAIHRDObjectiveFunction> objectiveFunction;
     try {
        objectiveFunction = std::make_unique<SEPAIHRDObjectiveFunction>(
            model_,
            *parameterManager,
            *cache_,
            observed_data_,
            time_points_,
            initial_state_cached_,
            solver_strategy_
            );
    } catch (const std::exception& e) {
        throw ModelConstructionException("SEPAIHRDModelCalibration::setupCalibrator",
                                         std::string("Failed to create SEPAIHRDObjectiveFunction: ") + e.what());
    }

    try {
        ModelCalibrator calibrator(
            std::move(parameterManager),
            std::move(objectiveFunction),
            std::move(algorithms),
            observed_data_,
            time_points_);
        return calibrator;
    } catch (const std::exception& e) {
         throw ModelConstructionException("SEPAIHRDModelCalibration::setupCalibrator",
                                          std::string("Failed to create ModelCalibrator: ") + e.what());
    }
}

SEPAIHRDParameterManager& SEPAIHRDModelCalibration::getParameterManager() {
    if (!parameter_manager_) {
        throw ModelConstructionException("SEPAIHRDModelCalibration::getParameterManager",
                                         "Parameter manager not initialized");
    }
    return *parameter_manager_;
}

Eigen::VectorXd SEPAIHRDModelCalibration::getCurrentParameterValues() {
    if (!parameter_manager_) {
        throw ModelConstructionException("SEPAIHRDModelCalibration::getCurrentParameterValues",
                                         "Parameter manager not initialized");
    }
    return parameter_manager_->getCurrentParameters();
}

ModelCalibrator SEPAIHRDModelCalibration::runHillClimbingMCMC(
    const std::map<std::string, double>& phase1_settings,
    const std::map<std::string, double>& phase2_settings)
{
    std::cout << "\n=== Setting up SEPAIHRD Calibration (Hill Climbing + MCMC) ===" << std::endl;

    auto phase1_algo = std::make_unique<HillClimbingOptimizer>();
    auto phase2_algo = std::make_unique<MetropolisHastingsSampler>();

    std::map<std::string, std::unique_ptr<IOptimizationAlgorithm>> optimization_algorithms;

    optimization_algorithms[ModelCalibrator::PHASE1_NAME] = std::move(phase1_algo);
    optimization_algorithms[ModelCalibrator::PHASE2_NAME] = std::move(phase2_algo);

    ModelCalibrator calibrator = setupCalibrator(std::move(optimization_algorithms));
    std::cout << "--- Starting Calibration ---" << std::endl;
    calibrator.calibrate(phase1_settings, phase2_settings);
    std::cout << "--- Calibration Finished ---" << std::endl;

    try {
         const auto& best_params_vec = calibrator.getBestParameterVector();
         std::cout << "Best parameters vector found: [" << best_params_vec.transpose() << "]" << std::endl;
         std::cout << "Best objective function value: " << calibrator.getBestObjectiveValue() << std::endl;
    } catch (const std::exception& e) {
         std::cerr << "Error accessing calibration results: " << e.what() << std::endl;
    }
    return calibrator;
}

ModelCalibrator SEPAIHRDModelCalibration::runPSOMCMC(
    const std::map<std::string, double>& phase1_settings,
    const std::map<std::string, double>& phase2_settings)
{
     std::cout << "\n=== Setting up SEPAIHRD Calibration (PSO + MCMC) ===" << std::endl;

    auto phase1_algo = std::make_unique<ParticleSwarmOptimization>();
    auto phase2_algo = std::make_unique<MetropolisHastingsSampler>();

    std::map<std::string, std::unique_ptr<IOptimizationAlgorithm>> optimization_algorithms;

    optimization_algorithms[ModelCalibrator::PHASE1_NAME] = std::move(phase1_algo);
    optimization_algorithms[ModelCalibrator::PHASE2_NAME] = std::move(phase2_algo);

    ModelCalibrator calibrator = setupCalibrator(std::move(optimization_algorithms));

    std::cout << "--- Starting Calibration ---" << std::endl;
    calibrator.calibrate(phase1_settings, phase2_settings);
    std::cout << "--- Calibration Finished ---" << std::endl;

    try {
         const auto& best_params_vec = calibrator.getBestParameterVector();
         std::cout << std::setprecision(6);
         std::cout << "Best parameters vector found: [" << best_params_vec.transpose() << "]" << std::endl;
         std::cout << "Best objective function value: " << calibrator.getBestObjectiveValue() << std::endl;
    } catch (const std::exception& e) {
         std::cerr << "Error accessing calibration results: " << e.what() << std::endl;
    }
    return calibrator;
}

} // namespace epidemic