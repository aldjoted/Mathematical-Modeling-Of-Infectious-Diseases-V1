#ifndef SEPAIHRD_GRADIENT_OBJECTIVE_FUNCTION_HPP
#define SEPAIHRD_GRADIENT_OBJECTIVE_FUNCTION_HPP

#include "model/objectives/SEPAIHRDObjectiveFunction.hpp"
#include "model/interfaces/IGradientObjectiveFunction.hpp"
#include <Eigen/Dense>
#include <future>     // For std::async
#include <vector>     // For std::vector<std::future>
#include <span>       // For std::span

namespace epidemic {

/**
 * @class SEPAIHRDGradientObjectiveFunction
 * @brief Extends SEPAIHRDObjectiveFunction to provide gradient calculations via finite differences
 *
 * This wrapper adds gradient computation capabilities to the existing objective function
 * using numerical differentiation. For production use, analytical gradients would be preferable.
 */
class SEPAIHRDGradientObjectiveFunction : public virtual SEPAIHRDObjectiveFunction, 
                                          public IGradientObjectiveFunction {
public:
    // Inherit constructors
    using SEPAIHRDObjectiveFunction::SEPAIHRDObjectiveFunction;
    
    // Epsilon for finite differences - use a larger value for better numerical stability
    // For relative perturbation: actual_eps = epsilon_ * max(1, |param_value|)
    double epsilon_ = 1e-4;  // Increased from 1e-8 for better gradient estimates

    /**
     * @brief Evaluates the objective function and its gradient in parallel.
     *
     * @param params The Eigen vector of parameters.
     * @param grad Output Eigen vector that will be filled with the gradient.
     * @return The value of the objective function (log-likelihood) at 'params'.
     */
    double evaluate_with_gradient(
        const Eigen::VectorXd& params, 
        Eigen::VectorXd& grad) const override;
};

} // namespace epidemic

#endif // SEPAIHRD_GRADIENT_OBJECTIVE_FUNCTION_HPP