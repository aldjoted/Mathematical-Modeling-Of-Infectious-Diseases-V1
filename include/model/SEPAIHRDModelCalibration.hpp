#ifndef SEPAIHRD_MODEL_CALIBRATOR_HPP
#define SEPAIHRD_MODEL_CALIBRATOR_HPP

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <utility>
#include "model/parameters/SEPAIHRDParameterManager.hpp"
#include "sir_age_structured/ModelCalibrator.hpp"
#include "sir_age_structured/interfaces/IOdeSolverStrategy.hpp"
#include "sir_age_structured/interfaces/ISimulationCache.hpp"
#include "sir_age_structured/caching/SimulationCache.hpp"      
#include "sir_age_structured/solvers/Dropri5SolverStrategy.hpp"
#include "sir_age_structured/optimizers/HillClimbingOptimizer.hpp"
#include "sir_age_structured/optimizers/MetropolisHastingsSampler.hpp"
#include "model/AgeSEPAIHRDModel.hpp"
#include "model/optimizers/ParticleSwarmOptimizer.hpp"
#include "utils/GetCalibrationData.hpp"

namespace epidemic {

/**
 * @class SEPAIHRDModelCalibration
 * @brief Sets up and runs calibration for the AgeSEPAIHRDModel using the modular ModelCalibrator framework.
 */
class SEPAIHRDModelCalibration {
public:
    /**
     * @brief Constructor.
     *
     * @param model_ptr Shared pointer to the configured AgeSEPAIHRDModel instance.
     * @param calibration_data Observed data (population, incidence, etc.).
     * @param time_points Simulation time points matching observed data rows.
     * @param params_to_calibrate List of parameter names to calibrate (e.g., "beta", "theta").
     * @param proposal_sigmas Map of parameter names to proposal sigmas for MCMC/Optimization.
     * @param param_bounds Map of parameter names to min/max bounds (pair<double, double>).
     * @param solver_strategy Optional ODE solver strategy (defaults to Dopri5).
     * @param cache Optional simulation cache (defaults to basic SimulationCache).
     */
    SEPAIHRDModelCalibration(
        std::shared_ptr<AgeSEPAIHRDModel> model_ptr,
        const CalibrationData& calibration_data,
        const std::vector<double>& time_points,
        const std::vector<std::string>& params_to_calibrate,
        const std::map<std::string, double>& proposal_sigmas,
        const std::map<std::string, std::pair<double, double>>& param_bounds,
        std::shared_ptr<IOdeSolverStrategy> solver_strategy = std::make_shared<Dopri5SolverStrategy>(),
        std::shared_ptr<ISimulationCache> cache = std::make_shared<SimulationCache>()
        );

    /**
     * @brief Get reference to the parameter manager for accessing initial parameters.
     * @return Reference to the parameter manager.
     */
    SEPAIHRDParameterManager& getParameterManager();
    
    /**
     * @brief Get the current parameter values as a vector.
     * @return Vector of current parameter values in calibration order.
     */
    Eigen::VectorXd getCurrentParameterValues();

    /**
     * @brief Runs a two-phase calibration: Hill Climbing followed by MCMC.
     * @param phase1_settings Configuration settings for Hill Climbing.
     * @param phase2_settings Configuration settings for MCMC.
     * @return ModelCalibrator The calibrator instance containing results after execution.
     */
    ModelCalibrator runHillClimbingMCMC(
        const std::map<std::string, double>& phase1_settings,
        const std::map<std::string, double>& phase2_settings);

    /**
     * @brief Runs a two-phase calibration: PSO followed by MCMC.
     * @param phase1_settings Configuration settings for PSO.
     * @param phase2_settings Configuration settings for MCMC.
     * @return ModelCalibrator The calibrator instance containing results after execution.
     */
    ModelCalibrator runPSOMCMC(
        const std::map<std::string, double>& phase1_settings,
        const std::map<std::string, double>& phase2_settings);

private:
    /** @brief Shared pointer to the AgeSEPAIHRDModel instance. */
    std::shared_ptr<AgeSEPAIHRDModel> model_;
    /** @brief Observed data used for calibration (population, incidence, etc.). Stored as a const reference. */
    const CalibrationData& observed_data_; // Store as const reference
    /** @brief Vector of time points for the simulation, matching observed data. */
    std::vector<double> time_points_;
    /** @brief List of parameter names that are subject to calibration. */
    std::vector<std::string> params_to_calibrate_;
    /** @brief Map of parameter names to their proposal sigmas for MCMC or optimization algorithms. */
    std::map<std::string, double> proposal_sigmas_;
    /** @brief Map of parameter names to their minimum and maximum bounds (pair<double, double>). */
    std::map<std::string, std::pair<double, double>> param_bounds_;
    /** @brief Shared pointer to the ODE solver strategy. */
    std::shared_ptr<IOdeSolverStrategy> solver_strategy_;
    /** @brief Shared pointer to the simulation cache. */
    std::shared_ptr<ISimulationCache> cache_;
    /** @brief Cached initial state vector derived from the observed data. */
    Eigen::VectorXd initial_state_cached_;
    /** @brief Unique pointer to the parameter manager. */
    std::unique_ptr<SEPAIHRDParameterManager> parameter_manager_;

    /**
     * @brief Helper to set up the generic ModelCalibrator with SEPAIHRD-specific components.
     * @param algorithms Map containing the configured optimization algorithms for different phases.
     * @return A configured ModelCalibrator instance.
     */
    ModelCalibrator setupCalibrator(
        std::map<std::string, std::unique_ptr<IOptimizationAlgorithm>> algorithms);
};

} // namespace epidemic

#endif // SEPAIHRD_MODEL_CALIBRATOR_HPP