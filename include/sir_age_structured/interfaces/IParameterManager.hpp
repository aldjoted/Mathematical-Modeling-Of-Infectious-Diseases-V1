#ifndef I_PARAMETER_MANAGER_HPP
#define I_PARAMETER_MANAGER_HPP

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <map>

namespace epidemic {

// Forward declaration
class IEpidemicModel;

/**
 * @brief Interface for managing model parameters during calibration.
 */
class IParameterManager {
public:
    virtual ~IParameterManager() = default;

    /**
     * @brief Get the current values of the managed parameters from the model.
     * @return Eigen::VectorXd Parameter values in a defined order.
     */
    virtual Eigen::VectorXd getCurrentParameters() const = 0;

    /**
     * @brief Update the model's parameters with the given values.
     * Handles constraints and updates the underlying model.
     * @param parameters The parameter vector.
     */
    virtual void updateModelParameters(const Eigen::VectorXd& parameters) = 0;

    /**
     * @brief Get the names of the parameters being managed.
     * @return const std::vector<std::string>&
     */
    virtual const std::vector<std::string>& getParameterNames() const = 0;

    /**
     * @brief Get the number of parameters being managed.
     * @return size_t
     */
    virtual size_t getParameterCount() const = 0;

    /**
     * @brief Get the proposal standard deviation (sigma) for a specific parameter index.
     * Used by optimization algorithms to generate steps.
     * @param index The index of the parameter.
     * @return double The standard deviation (sigma).
     */
    virtual double getSigmaForParamIndex(int index) const = 0;

    /**
     * @brief Apply constraints to a parameter vector (e.g., ensure positivity).
     * @param parameters Vector to apply constraints to (in-place or returns new).
     * @return Eigen::VectorXd The constrained parameter vector.
     */
    virtual Eigen::VectorXd applyConstraints(const Eigen::VectorXd& parameters) const = 0;

    /**
     * @brief Get the lower bound for a specific parameter index.
     * @param idx The index of the parameter.
     * @return double The lower bound.
     */
    virtual double getLowerBoundForParamIndex(int idx) const = 0;

    /**
     * @brief Get the upper bound for a specific parameter index.
     * @param idx The index of the parameter.
     * @return double The upper bound.
     */
    virtual double getUpperBoundForParamIndex(int idx) const = 0;

};

} // namespace epidemic

#endif // I_PARAMETER_MANAGER_HPP
