#ifndef I_OBJECTIVE_FUNCTION_HPP
#define I_OBJECTIVE_FUNCTION_HPP

#include <Eigen/Dense>
#include <vector>
#include <string>

namespace epidemic {

/**
 * @brief Interface for objective function calculation used in model calibration.
 */
class IObjectiveFunction {
public:
    virtual ~IObjectiveFunction() = default;

    /**
     * @brief Calculate the objective function score (e.g., log-likelihood) for a given parameter set.
     *
     * @param parameters The parameter vector to evaluate.
     * @return double The calculated objective score. Higher values are typically better.
     *         Returns -infinity or NaN on calculation failure.
     */
    virtual double calculate(const Eigen::VectorXd& parameters) = 0;

    /**
     * @brief Get the names of the parameters expected by this objective function.
     * @return const std::vector<std::string>&
     */
    virtual const std::vector<std::string>& getParameterNames() const = 0;
};

} // namespace epidemic

#endif // I_OBJECTIVE_FUNCTION_HPP
