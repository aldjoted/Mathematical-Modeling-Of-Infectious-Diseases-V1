#ifndef IGRADIENTOBJECTIVEFUNCTION_HPP
#define IGRADIENTOBJECTIVEFUNCTION_HPP

#include "sir_age_structured/interfaces/IObjectiveFunction.hpp"

namespace epidemic {

/**
 * @class IGradientObjectiveFunction
 * @brief An optional, extended interface for objective functions that can provide gradients.
 *
 * This inherits from the base IObjectiveFunction to maintain compatibility with older
 * algorithms, but adds a method for gradient evaluation required by advanced
 * samplers like NUTS. A class can implement this interface to signal that it
 * can provide gradients.
 */
class IGradientObjectiveFunction : public virtual IObjectiveFunction {
public:
    virtual ~IGradientObjectiveFunction() = default;

    /**
     * @brief Evaluates the objective function (log-posterior) and its gradient.
     *
     * @param params The vector of parameters at which to evaluate the function.
     * @param grad A vector that will be filled with the gradient of the log-posterior
     * with respect to the parameters. The vector will be resized if necessary.
     * @return The value of the objective function (log-posterior).
     */
    virtual double evaluate_with_gradient(const std::vector<double>& params, std::vector<double>& grad) const = 0;
};

} // namespace epidemic

#endif // IGRADIENTOBJECTIVEFUNCTION_HPP