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
 * @brief Parallel Adaptive Hill Climbing Optimizer (Phase 1 Calibration).
 *
 * This optimizer implements a "Cloud Search" strategy designed to utilize 
 * multi-core CPUs to rapidly climb the likelihood surface.
 * * Strategy:
 * 1. Cloud Generation: Generates N candidate points in parallel using an 
 * Adaptive Covariance Matrix (50% global correlated moves, 50% local axis-aligned moves).
 * 2. Winner Selection: Evaluates all candidates and picks the single best point.
 * 3. Directional Line Search: Performs an aggressive expansion/contraction line search 
 * along the vector of improvement to accelerate convergence.
 * 4. Covariance Adaptation: Updates the search distribution shape based on successful steps.
 * * Note: This optimizer does not store sample history as it is intended for 
 * finding the Maximum A Posteriori (MAP) point, not for posterior estimation.
 */
class HillClimbingOptimizer : public IOptimizationAlgorithm {
public:
    HillClimbingOptimizer();
    virtual ~HillClimbingOptimizer() = default;

    /**
     * @brief Configure optimizer parameters.
     * Keys:
     * - "iterations": Total optimization steps (default: 2000).
     * - "report_interval": Logging frequency (default: 100).
     * - "cloud_size_multiplier": Cloud size = num_threads * multiplier (default: 8).
     */
    void configure(const std::map<std::string, double>& settings) override;

    /**
     * @brief Execute the Parallel Cloud Search optimization.
     */
    OptimizationResult optimize(
        const Eigen::VectorXd& initialParameters,
        IObjectiveFunction& objectiveFunction,
        IParameterManager& parameterManager) override;

private:
    int iterations_ = 2000;
    int report_interval_ = 100;
    int cloud_size_multiplier_ = 8;
    
    std::mt19937 gen_;
};

} // namespace epidemic

#endif // HILL_CLIMBING_OPTIMIZER_HPP