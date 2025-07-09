#ifndef SEPAIHRD_GRADIENT_OBJECTIVE_FUNCTION_HPP
#define SEPAIHRD_GRADIENT_OBJECTIVE_FUNCTION_HPP

#include "model/objectives/SEPAIHRDObjectiveFunction.hpp"
#include "model/interfaces/IGradientObjectiveFunction.hpp"
#include <vector>

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

    /**
     * @brief Evaluates the objective function and its gradient using finite differences
     *
     * @param params The vector of parameters at which to evaluate the function
     * @param grad Output vector that will be filled with the gradient
     * @return The value of the objective function (log-likelihood)
     */
    double evaluate_with_gradient(const std::vector<double>& params, std::vector<double>& grad) const override {
        const double eps = 1e-8;  // Finite difference step size
        int n_params = params.size();
        
        // Convert to Eigen vector
        Eigen::VectorXd param_vec = Eigen::Map<const Eigen::VectorXd>(params.data(), n_params);
        
        // Evaluate function at current point
        double f_center = this->calculate(param_vec);
        
        // Resize gradient vector
        grad.resize(n_params);
        
        // Compute gradient using central differences
        for (int i = 0; i < n_params; ++i) {
            Eigen::VectorXd param_plus = param_vec;
            Eigen::VectorXd param_minus = param_vec;
            
            // Get bounds for this parameter
            double lower = parameterManager_.getLowerBoundForParamIndex(i);
            double upper = parameterManager_.getUpperBoundForParamIndex(i);
            
            // Adaptive step size near boundaries
            double h = eps;
            if (param_vec[i] + h > upper) {
                h = (upper - param_vec[i]) * 0.5;
            }
            if (param_vec[i] - h < lower) {
                h = (param_vec[i] - lower) * 0.5;
            }
            
            // Ensure h is not too small
            h = std::max(h, eps * 0.01);
            
            param_plus[i] += h;
            param_minus[i] -= h;
            
            // Apply constraints
            param_plus = parameterManager_.applyConstraints(param_plus);
            param_minus = parameterManager_.applyConstraints(param_minus);
            
            // Evaluate function at perturbed points
            double f_plus = this->calculate(param_plus);
            double f_minus = this->calculate(param_minus);
            
            // Handle cases where function evaluation fails
            if (!std::isfinite(f_plus) || !std::isfinite(f_minus)) {
                // Try one-sided difference
                if (std::isfinite(f_plus)) {
                    grad[i] = (f_plus - f_center) / h;
                } else if (std::isfinite(f_minus)) {
                    grad[i] = (f_center - f_minus) / h;
                } else {
                    // Both failed, set gradient to zero
                    grad[i] = 0.0;
                }
            } else {
                // Central difference
                grad[i] = (f_plus - f_minus) / (2.0 * h);
            }
            
            // Ensure gradient is finite
            if (!std::isfinite(grad[i])) {
                grad[i] = 0.0;
            }
        }
        
        return f_center;
    }
    
    // Override the base calculate method to match the interface
    double calculate(const Eigen::VectorXd& parameters) const override {
        return SEPAIHRDObjectiveFunction::calculate(parameters);
    }
    
    const std::vector<std::string>& getParameterNames() const override {
        return SEPAIHRDObjectiveFunction::getParameterNames();
    }
};

} // namespace epidemic

#endif // SEPAIHRD_GRADIENT_OBJECTIVE_FUNCTION_HPP