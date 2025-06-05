#ifndef MODEL_CALIBRATOR_H
#define MODEL_CALIBRATOR_H

#include "sir_age_structured/interfaces/IObjectiveFunction.hpp"
#include "sir_age_structured/interfaces/IOptimizationAlgorithm.hpp"
#include "sir_age_structured/interfaces/IParameterManager.hpp"
#include "sir_age_structured/AgeSIRModel.hpp"
#include "utils/GetCalibrationData.hpp"
#include "exceptions/Exceptions.hpp"
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <limits>
#include <unordered_map>

namespace epidemic {

/**
 * @class ModelCalibrator
 * @brief Coordinates the calibration process for an epidemic model.
 *
 * This class orchestrates the calibration by using injected components:
 * - An `IObjectiveFunction` to evaluate parameter sets.
 * - One or more `IOptimizationAlgorithm` instances for finding optimal parameters (e.g., best-fit search, MCMC).
 * - An `IParameterManager` to handle interaction with the model's parameters.
 *
 * It manages the overall workflow, stores the best results found, and provides access to MCMC samples if applicable.
 */
class ModelCalibrator {
public:
    /**
     * @brief Identifier for the first optimization phase, typically used for best-fit search.
     */
    static constexpr const char* PHASE1_NAME = "Phase1_BestFit";
    
    /**
     * @brief Identifier for the second optimization phase, typically used for MCMC sampling.
     */
    static constexpr const char* PHASE2_NAME = "Phase2_MCMC";

    /**
     * @brief Construct a new Model Calibrator.
     * 
     * Initializes the calibrator with necessary components and validates initial configuration.
     * Sets up initial best parameters and objective values.
     *
     * @param parameterManager Manager for getting/setting model parameters.
     * @param objectiveFunction Function that calculates the score for parameter sets.
     * @param optimization_algorithms Map of phase names to optimization algorithm instances.
     * @param calibration_data Reference to observed data used for calibration.
     * @param time_points Reference to the simulation time points.
     * @throws InvalidParameterException If parameterManager is null, objectiveFunction is null, 
     *         optimization_algorithms is empty, or parameter names mismatch between manager and function.
     */
    ModelCalibrator(
        std::unique_ptr<IParameterManager> parameterManager,
        std::unique_ptr<IObjectiveFunction> objectiveFunction,
        std::map<std::string, std::unique_ptr<IOptimizationAlgorithm>> optimization_algorithms,
        const CalibrationData& calibration_data,
        const std::vector<double>& time_points);

    /**
     * @brief Run the full calibration procedure using the configured algorithms.
     *
     * Executes the optimization algorithms in sequence (Phase 1 for best-fit, Phase 2 for MCMC if provided).
     * For each phase, it configures the algorithm with the provided settings, runs the optimization,
     * and updates the best parameters if better results are found. For Phase 2, it also collects
     * MCMC samples and calculates their objective values.
     *
     * @param phase1_settings Configuration settings for the first optimization phase.
     * @param phase2_settings Configuration settings for the second optimization phase.
     */
    void calibrate(const std::map<std::string, double>& phase1_settings,
                   const std::map<std::string, double>& phase2_settings);

    /**
     * @brief Retrieve the best parameter vector identified during calibration.
     *
     * @return const Eigen::VectorXd& The best parameter vector found.
     */
    const Eigen::VectorXd& getBestParameterVector() const { return best_params_vector_; }

    /**
     * @brief Get the highest objective function value achieved during calibration.
     *
     * @return double The maximum objective value corresponding to `getBestParameterVector()`.
     */
    double getBestObjectiveValue() const { return best_objective_value_; }

    /**
     * @brief Get the MCMC parameter samples collected during Phase 2 (if used).
     *
     * @return const std::vector<Eigen::VectorXd>& Vector of MCMC parameter samples. 
     *         Returns empty vector if no MCMC phase was executed.
     */
    const std::vector<Eigen::VectorXd>& getMCMCSamples() const;

    /**
     * @brief Get the objective values corresponding to the collected MCMC samples.
     *
     * @return const std::vector<double>& Vector of objective values, parallel to `getMCMCSamples()`. 
     *         Returns empty vector if no MCMC phase was executed.
     */
    const std::vector<double>& getMCMCObjectiveValues() const;

    /**
     * @brief Get the parameter manager instance.
     * 
     * @return IParameterManager& Reference to the parameter manager.
     */
    IParameterManager& getParameterManager() const { return *parameterManager_; }

    /**
     * @brief Get the objective function instance.
     * 
     * @return IObjectiveFunction& Reference to the objective function.
     */
    IObjectiveFunction& getObjectiveFunction() const { return *objectiveFunction_; }

private:
    /** @brief Manages access to and modification of model parameters. */
    std::unique_ptr<IParameterManager> parameterManager_;
    /** @brief Calculates the objective function value for a given set of parameters. */
    std::unique_ptr<IObjectiveFunction> objectiveFunction_;
    /** @brief A map of optimization algorithms used in different calibration phases. */
    std::map<std::string, std::unique_ptr<IOptimizationAlgorithm>> optimization_algorithms_;
    /** @brief The observed calibration data used to evaluate model fit. */
    const CalibrationData& observed_data_; 
    /** @brief The time points for the simulation. */
    const std::vector<double>& time_points_;

    // Store results
    /** @brief The best parameter vector found during calibration. */
    Eigen::VectorXd best_params_vector_;
    /** @brief The highest objective function value achieved. */
    double best_objective_value_ = -std::numeric_limits<double>::infinity();
    /** @brief Objective function values corresponding to MCMC samples. */
    std::vector<double> mcmcObjectiveValues_;
    /** @brief Stores the results from the first optimization phase (e.g., best-fit). */
    OptimizationResult phase1_result_;
    /** @brief Stores the results from the second optimization phase (e.g., MCMC). */
    OptimizationResult phase2_result_;

    // Cache for simulation results to avoid redundant calculations
    /** @brief Caches objective function values for parameter sets to avoid re-computation. */
    std::unordered_map<std::string, double> objective_cache;
};

} // namespace epidemic

#endif // MODEL_CALIBRATOR_H