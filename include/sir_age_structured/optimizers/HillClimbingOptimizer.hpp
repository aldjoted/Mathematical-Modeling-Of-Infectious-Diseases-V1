#ifndef HILL_CLIMBING_OPTIMIZER_HPP
#define HILL_CLIMBING_OPTIMIZER_HPP

#include "sir_age_structured/interfaces/IOptimizationAlgorithm.hpp"
#include "sir_age_structured/interfaces/IObjectiveFunction.hpp"
#include "sir_age_structured/interfaces/IParameterManager.hpp"
#include "exceptions/Exceptions.hpp"
#include <Eigen/Dense>
#include <map>
#include <random>
#include <string>
#include <vector>
#include <memory>

namespace epidemic {

/**
 * @brief Stochastic hill climbing optimizer with bidirectional search,
 *        step elongation, and binary refinement.
 *
 * This optimizer implements optimization strategies including:
 * - Bidirectional proposal testing (forward and reverse)
 * - Step elongation for successful directions
 * - Binary search refinement
 * - Adaptive step sizing during burn-in
 * - Periodic global restarts
 */
class HillClimbingOptimizer : public IOptimizationAlgorithm {
public:
    /**
     * @brief Default constructor.
     * 
     * Initializes the optimizer with default configuration parameters
     * and sets up the random number generator.
     */
    HillClimbingOptimizer();

    /**
     * @brief Configure optimizer hyperparameters.
     * 
     * @param settings Map of configuration parameter names to their values.
     *                 Supported parameters include: iterations, initial_step,
     *                 cooling_rate, refinement_steps, burnin_factor, etc.
     */
    void configure(const std::map<std::string, double>& settings) override;

    /**
     * @brief Run optimization from a given starting point.
     * 
     * @param initialParameters Starting point for the optimization.
     * @param objectiveFunction Function to maximize during optimization.
     * @param parameterManager Manager for parameter constraints and properties.
     * @return OptimizationResult containing the best parameters and objective value found.
     */
    OptimizationResult optimize(
        const Eigen::VectorXd& initialParameters,
        IObjectiveFunction& objectiveFunction,
        IParameterManager& parameterManager) override;

private:
    // Configuration parameters
    int    iterations_ = 5000;             ///< Maximum number of optimization iterations
    double initial_step_coef_ = 1.0;      ///< Initial step size coefficient
    double cooling_rate_ = 0.995;           ///< Rate at which step size decreases (0 < rate < 1)
    int    refinement_steps_ = 5;       ///< Number of binary refinement steps
    double burnin_factor_ = 0.2;          ///< Fraction of iterations for burn-in phase (0 <= factor <= 1)
    double burnin_step_increase_ = 1.5;   ///< Step size multiplier during burn-in
    double post_burnin_step_coef_ = 1.0;  ///< Step size coefficient after burn-in
    double one_param_step_coef_ = 1.0;    ///< Step size coefficient for single parameter updates
    double min_step_coef_ = 0.01;         ///< Minimum allowed step size coefficient
    int    report_interval_ = 100;        ///< Interval for progress reporting
    int    restart_interval_ = 0;       ///< Interval for global restarts (0 = no restarts)
    bool   restart_resets_step_ = true;    ///< Whether restarts reset the step size
    bool   enable_bidirectional_ = true;   ///< Enable bidirectional search (forward and reverse)
    bool   enable_elongation_ = true;      ///< Enable step elongation for successful directions
    
    std::mt19937 gen_;  ///< Random number generator for stochastic operations

    /**
     * @brief Structure to hold evaluation results
     */
    struct EvaluationResult {
        Eigen::VectorXd parameters;
        double objective;
        bool valid;
    };

    /**
     * @brief Perform optimized randomization with bidirectional search and refinement
     * 
     * This implements the core optimization logic from opti_randomize in the C code
     */
    EvaluationResult performOptimizedStep(
        const Eigen::VectorXd& currentParams,
        double currentObjective,
        int iteration,
        IObjectiveFunction& objectiveFunction,
        IParameterManager& parameterManager);

    /**
     * @brief Elongate a successful step direction
     * 
     * Implements the step elongation logic from the C code
     */
    EvaluationResult elongateStep(
        const Eigen::VectorXd& baseParams,
        const Eigen::VectorXd& stepDirection,
        double baseObjective,
        double stepCoef,
        IObjectiveFunction& objectiveFunction,
        IParameterManager& parameterManager);

    /**
     * @brief Perform binary search refinement
     * 
     * Implements the binary refinement from the C code
     */
    EvaluationResult binaryRefinement(
        const Eigen::VectorXd& baseParams,
        const Eigen::VectorXd& stepDirection,
        double baseObjective,
        double initialStepCoef,
        int numSteps,
        IObjectiveFunction& objectiveFunction,
        IParameterManager& parameterManager);

    /**
     * @brief Evaluate parameters with constraint checking
     */
    EvaluationResult evaluateParameters(
        const Eigen::VectorXd& params,
        IObjectiveFunction& objectiveFunction,
        IParameterManager& parameterManager);

    /**
     * @brief Generate random step for all parameters
     */
    Eigen::VectorXd generateStepAll(double stepCoef, IParameterManager& parameterManager);

    /**
     * @brief Generate random step for one parameter
     */
    Eigen::VectorXd generateStepOne(double stepCoef, IParameterManager& parameterManager);
};

} // namespace epidemic

#endif // HILL_CLIMBING_OPTIMIZER_HPP