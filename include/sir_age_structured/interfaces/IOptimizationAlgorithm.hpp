#ifndef I_OPTIMIZATION_ALGORITHM_HPP
#define I_OPTIMIZATION_ALGORITHM_HPP

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <map>

namespace epidemic {

// Forward declarations
class IObjectiveFunction;
class IParameterManager;

/**
 * @brief Structure to hold the results of an optimization run.
 */
struct OptimizationResult {
    Eigen::VectorXd bestParameters;
    double bestObjectiveValue = -std::numeric_limits<double>::infinity();
    std::vector<Eigen::VectorXd> samples; // For MCMC-like algorithms
    std::vector<double> sampleObjectiveValues; // For MCMC-like algorithms
};

/**
 * @brief Interface for optimization algorithms used in model calibration.
 */
class IOptimizationAlgorithm {
public:
    virtual ~IOptimizationAlgorithm() = default;

    /**
     * @brief Run the optimization algorithm.
     *
     * @param initialParameters The starting point for the optimization.
     * @param objectiveFunction The objective function to maximize/minimize.
     * @param parameterManager Manager to handle parameter updates and constraints.
     * @return OptimizationResult Structure containing the best parameters found and other results.
     */
    virtual OptimizationResult optimize(
        const Eigen::VectorXd& initialParameters,
        IObjectiveFunction& objectiveFunction,
        IParameterManager& parameterManager) = 0;

    /**
     * @brief Configure algorithm-specific settings.
     * @param settings Map of setting names to values (e.g., "iterations", "step_size").
     */
    virtual void configure(const std::map<std::string, double>& settings) = 0;
};

} // namespace epidemic

#endif // I_OPTIMIZATION_ALGORITHM_HPP
