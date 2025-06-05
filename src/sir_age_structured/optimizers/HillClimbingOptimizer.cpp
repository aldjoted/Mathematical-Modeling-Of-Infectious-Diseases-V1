#include "sir_age_structured/optimizers/HillClimbingOptimizer.hpp"
#include "utils/Logger.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <limits>

namespace epidemic {

HillClimbingOptimizer::HillClimbingOptimizer()
    : gen_{std::random_device{}()} {}

void HillClimbingOptimizer::configure(const std::map<std::string, double>& settings) {
    auto get = [&](const std::string& key, double def) {
        auto it = settings.find(key);
        return it != settings.end() ? it->second : def;
    };

    iterations_            = static_cast<int>(get("iterations", iterations_));
    initial_step_coef_     = get("initial_step", initial_step_coef_);
    cooling_rate_          = get("cooling_rate", cooling_rate_);
    refinement_steps_      = static_cast<int>(get("refinement_steps", refinement_steps_));
    burnin_factor_         = get("burnin_factor", burnin_factor_);
    burnin_step_increase_  = get("burnin_step_increase", burnin_step_increase_);
    post_burnin_step_coef_ = get("post_burnin_step_coef", post_burnin_step_coef_);
    one_param_step_coef_   = get("one_param_step_coef", one_param_step_coef_);
    min_step_coef_         = get("min_step_coef", min_step_coef_);
    report_interval_       = static_cast<int>(get("report_interval", report_interval_));
    restart_interval_      = static_cast<int>(get("restart_interval", restart_interval_));
    restart_resets_step_   = get("restart_resets_step", 1.0) != 0.0;
    enable_bidirectional_  = get("enable_bidirectional", 1.0) != 0.0;
    enable_elongation_     = get("enable_elongation", 1.0) != 0.0;

    // Validate parameters
    iterations_            = std::max(iterations_, 1);
    initial_step_coef_     = std::max(initial_step_coef_, 1e-6);
    cooling_rate_          = std::clamp(cooling_rate_, 0.001, 0.999);
    refinement_steps_      = std::max(refinement_steps_, 0);
    burnin_factor_         = std::clamp(burnin_factor_, 0.0, 1.0);
    min_step_coef_         = std::max(min_step_coef_, 1e-12);
    report_interval_       = std::max(report_interval_, 1);

    Logger::getInstance().info("HillClimbingOptimizer", 
        "Configured with bidirectional=" + std::to_string(enable_bidirectional_) +
        ", elongation=" + std::to_string(enable_elongation_) +
        ", refinement_steps=" + std::to_string(refinement_steps_));
}

OptimizationResult HillClimbingOptimizer::optimize(
    const Eigen::VectorXd& initialParameters,
    IObjectiveFunction& objectiveFunction,
    IParameterManager& parameterManager) {
    
    const std::string F_NAME = "HillClimbingOptimizer::optimize";
    Logger::getInstance().info(F_NAME, "Starting optimization with " + 
                              std::to_string(iterations_) + " iterations");

    OptimizationResult result;
    result.bestParameters = initialParameters;
    result.bestObjectiveValue = objectiveFunction.calculate(initialParameters);

    if (!std::isfinite(result.bestObjectiveValue)) {
        Logger::getInstance().warning(F_NAME, "Invalid initial objective value");
        result.bestObjectiveValue = -std::numeric_limits<double>::infinity();
    }

    Eigen::VectorXd current = initialParameters;
    double currentObjective = result.bestObjectiveValue;
    double stepCoef = initial_step_coef_;
    int burnin_iters = static_cast<int>(iterations_ * burnin_factor_);
    int accepted = 0;
    int improved = 0;

    for (int iter = 1; iter <= iterations_; ++iter) {
        // Handle restarts
        if (restart_interval_ > 0 && iter % restart_interval_ == 0) {
            Logger::getInstance().info(F_NAME, "Restarting from best parameters at iteration " + 
                                     std::to_string(iter));
            current = result.bestParameters;
            currentObjective = result.bestObjectiveValue;
            if (restart_resets_step_) {
                stepCoef = initial_step_coef_;
            }
        }

        // Perform optimized step
        auto stepResult = performOptimizedStep(
            current, currentObjective, iter, objectiveFunction, parameterManager);

        if (stepResult.valid && stepResult.objective > currentObjective) {
            current = stepResult.parameters;
            currentObjective = stepResult.objective;
            accepted++;

            if (currentObjective > result.bestObjectiveValue) {
                result.bestObjectiveValue = currentObjective;
                result.bestParameters = current;
                improved++;
                Logger::getInstance().info(F_NAME, 
                    "New best at iteration " + std::to_string(iter) + 
                    ": " + std::to_string(currentObjective));
            }
        }

        // Cool down step size
        if (iter > burnin_iters) {
            stepCoef = std::max(stepCoef * cooling_rate_, min_step_coef_);
        }

        // Progress reporting
        if (iter % report_interval_ == 0 || iter == iterations_) {
            double acceptRate = 100.0 * accepted / iter;
            Logger::getInstance().info(F_NAME, 
                "Iteration " + std::to_string(iter) + "/" + std::to_string(iterations_) +
                ", Current: " + std::to_string(currentObjective) +
                ", Best: " + std::to_string(result.bestObjectiveValue) +
                ", Accept rate: " + std::to_string(acceptRate) + "%" +
                ", Step coef: " + std::to_string(stepCoef));
        }
    }

    Logger::getInstance().info(F_NAME, 
        "Optimization complete. Accepted: " + std::to_string(accepted) + "/" + 
        std::to_string(iterations_) + ", Improved: " + std::to_string(improved));

    return result;
}

HillClimbingOptimizer::EvaluationResult HillClimbingOptimizer::performOptimizedStep(
    const Eigen::VectorXd& currentParams,
    double currentObjective,
    int iteration,
    IObjectiveFunction& objectiveFunction,
    IParameterManager& parameterManager) {
    
    int burnin_iters = static_cast<int>(iterations_ * burnin_factor_);
    bool in_burnin = iteration <= burnin_iters;
    
    // Determine step type and coefficient
    double stepCoef;
    Eigen::VectorXd stepDirection;
    
    if (iteration == 1) {
        // First iteration: use burn-in start coefficient
        stepCoef = burnin_step_increase_;
        stepDirection = generateStepAll(1.0, parameterManager);
    } else if (in_burnin || iteration % 2 == 0) {
        // All parameters
        stepCoef = in_burnin ? burnin_step_increase_ : post_burnin_step_coef_;
        stepDirection = generateStepAll(1.0, parameterManager);
    } else {
        // One parameter
        stepCoef = one_param_step_coef_;
        stepDirection = generateStepOne(1.0, parameterManager);
    }

    // Try forward direction
    Eigen::VectorXd candidateParams = currentParams + stepCoef * stepDirection;
    auto forwardResult = evaluateParameters(candidateParams, objectiveFunction, parameterManager);
    
    EvaluationResult bestResult = forwardResult;
    bool improved = forwardResult.valid && forwardResult.objective > currentObjective;

    // Try reverse direction if enabled and forward didn't improve
    if (enable_bidirectional_ && !improved) {
        candidateParams = currentParams - stepCoef * stepDirection;
        auto reverseResult = evaluateParameters(candidateParams, objectiveFunction, parameterManager);
        
        if (reverseResult.valid && reverseResult.objective > currentObjective) {
            bestResult = reverseResult;
            stepDirection = -stepDirection;
            improved = true;
        }
    }

    // If we found improvement, try elongation and refinement
    if (improved) {
        // Elongate successful step
        if (enable_elongation_) {
            auto elongResult = elongateStep(
                bestResult.parameters, stepDirection, bestResult.objective,
                stepCoef, objectiveFunction, parameterManager);
            
            if (elongResult.valid && elongResult.objective > bestResult.objective) {
                bestResult = elongResult;
            }
        }

        // Binary refinement
        if (refinement_steps_ > 0) {
            auto refineResult = binaryRefinement(
                bestResult.parameters, stepDirection, bestResult.objective,
                stepCoef, refinement_steps_, objectiveFunction, parameterManager);
            
            if (refineResult.valid && refineResult.objective > bestResult.objective) {
                bestResult = refineResult;
            }
        }
    }

    return bestResult;
}

HillClimbingOptimizer::EvaluationResult HillClimbingOptimizer::elongateStep(
    const Eigen::VectorXd& baseParams,
    const Eigen::VectorXd& stepDirection,
    double baseObjective,
    double stepCoef,
    IObjectiveFunction& objectiveFunction,
    IParameterManager& parameterManager) {
    
    EvaluationResult result{baseParams, baseObjective, true};
    
    // Keep extending in the successful direction
    while (true) {
        Eigen::VectorXd extendedParams = result.parameters + stepCoef * stepDirection;
        auto extendedResult = evaluateParameters(extendedParams, objectiveFunction, parameterManager);
        
        if (extendedResult.valid && extendedResult.objective > result.objective) {
            result = extendedResult;
        } else {
            break; 
        }
    }
    
    return result;
}

HillClimbingOptimizer::EvaluationResult HillClimbingOptimizer::binaryRefinement(
    const Eigen::VectorXd& baseParams,
    const Eigen::VectorXd& stepDirection,
    double baseObjective,
    double initialStepCoef,
    int numSteps,
    IObjectiveFunction& objectiveFunction,
    IParameterManager& parameterManager) {
    
    EvaluationResult result{baseParams, baseObjective, true};
    double alpha = initialStepCoef;
    
    for (int k = 0; k < numSteps; ++k) {
        alpha *= 0.5;  // Halve the step size
        
        // Try both directions
        for (int sign : {-1, 1}) {
            Eigen::VectorXd refinedParams = result.parameters + sign * alpha * stepDirection;
            auto refinedResult = evaluateParameters(refinedParams, objectiveFunction, parameterManager);
            
            if (refinedResult.valid && refinedResult.objective > result.objective) {
                result = refinedResult;
            }
        }
    }
    
    return result;
}

HillClimbingOptimizer::EvaluationResult HillClimbingOptimizer::evaluateParameters(
    const Eigen::VectorXd& params,
    IObjectiveFunction& objectiveFunction,
    IParameterManager& parameterManager) {
    
    // Apply constraints
    Eigen::VectorXd constrainedParams = parameterManager.applyConstraints(params);
    
    // Evaluate objective
    double objective = objectiveFunction.calculate(constrainedParams);
    
    bool valid = std::isfinite(objective);
    if (!valid) {
        objective = -std::numeric_limits<double>::infinity();
    }
    
    return {constrainedParams, objective, valid};
}

Eigen::VectorXd HillClimbingOptimizer::generateStepAll(
    double stepCoef, IParameterManager& parameterManager) {
    
    size_t paramCount = parameterManager.getParameterCount();
    Eigen::VectorXd steps(paramCount);
    
    for (size_t i = 0; i < paramCount; ++i) {
        double sigma = parameterManager.getSigmaForParamIndex(i);
        std::normal_distribution<> dist(0.0, sigma * stepCoef);
        steps[i] = dist(gen_);
    }
    
    return steps;
}

Eigen::VectorXd HillClimbingOptimizer::generateStepOne(
    double stepCoef, IParameterManager& parameterManager) {
    
    size_t paramCount = parameterManager.getParameterCount();
    Eigen::VectorXd steps = Eigen::VectorXd::Zero(paramCount);
    
    if (paramCount > 0) {
        std::uniform_int_distribution<> idxDist(0, paramCount - 1);
        int idx = idxDist(gen_);
        
        double sigma = parameterManager.getSigmaForParamIndex(idx);
        std::normal_distribution<> dist(0.0, sigma * stepCoef);
        steps[idx] = dist(gen_);
    }
    
    return steps;
}

} // namespace epidemic