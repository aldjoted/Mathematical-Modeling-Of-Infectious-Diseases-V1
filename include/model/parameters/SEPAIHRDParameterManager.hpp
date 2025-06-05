#ifndef SEPAIHRD_PARAMETER_MANAGER_HPP
#define SEPAIHRD_PARAMETER_MANAGER_HPP

#include "sir_age_structured/interfaces/IParameterManager.hpp"
#include "model/AgeSEPAIHRDModel.hpp"
#include "exceptions/Exceptions.hpp"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <algorithm>

namespace epidemic {

/**
 * @brief Manages parameters (beta, theta,..., potentially NPI) for the AgeSEPAIHRDModel during calibration.
 */
class SEPAIHRDParameterManager : public IParameterManager {
public:
    /**
     * @brief Constructor.
     * @param model Shared pointer to the AgeSEPAIHRDModel instance.
     * @param params_to_calibrate List of parameter names to manage (e.g., "beta", "theta", "p_0", "kappa_2").
     *                              For NPI parameters, use names consistent with the NPI strategy's calibratable parameters
     *                              (e.g., "kappa_2", "kappa_3" if kappa_1 is fixed).
     * @param proposal_sigmas Map of parameter names to their proposal standard deviations.
     * @param param_bounds Map of parameter names to their min/max bounds (pair<double, double>).
     */
    SEPAIHRDParameterManager(
        std::shared_ptr<AgeSEPAIHRDModel> model,
        const std::vector<std::string>& params_to_calibrate,
        const std::map<std::string, double>& proposal_sigmas,
        const std::map<std::string, std::pair<double, double>>& param_bounds);

    /**
     * @brief Retrieves the current values of the managed parameters.
     * @return An Eigen::VectorXd containing the current parameter values, in the order defined by getParameterNames().
     * @see IParameterManager::getCurrentParameters()
     */
    Eigen::VectorXd getCurrentParameters() const override;
    /**
     * @brief Updates the model's parameters with the given values.
     * Applies constraints before updating.
     * @param parameters An Eigen::VectorXd containing the new parameter values, in the order defined by getParameterNames().
     * @see IParameterManager::updateModelParameters(const Eigen::VectorXd&)
     */
    void updateModelParameters(const Eigen::VectorXd& parameters) override;
    /**
     * @brief Gets the names of the parameters being managed.
     * @return A constant reference to a vector of strings containing parameter names.
     * @see IParameterManager::getParameterNames()
     */
    const std::vector<std::string>& getParameterNames() const override { return param_names_; }
    /**
     * @brief Gets the total number of parameters being managed.
     * @return The count of managed parameters.
     * @see IParameterManager::getParameterCount()
     */
    size_t getParameterCount() const override { return param_names_.size(); }
    /**
     * @brief Gets the proposal sigma for a parameter at a specific index.
     * @param index The index of the parameter (corresponds to the order in getParameterNames()).
     * @return The proposal standard deviation for the parameter.
     * @throws std::out_of_range if the index is invalid.
     * @throws epidemic::InvalidParameterException if sigma is not found for the parameter name.
     * @see IParameterManager::getSigmaForParamIndex(int)
     */
    double getSigmaForParamIndex(int index) const override;
    /**
     * @brief Applies defined constraints (e.g., bounds) to a vector of parameter values.
     * @param parameters The input parameter vector.
     * @return An Eigen::VectorXd with constraints applied.
     * @throws epidemic::InvalidParameterException if parameter vector size mismatches.
     * @see IParameterManager::applyConstraints(const Eigen::VectorXd&)
     */
    Eigen::VectorXd applyConstraints(const Eigen::VectorXd& parameters) const override;
    /**
     * @brief Get the lower bound for a specific parameter index.
     * @param idx The index of the parameter.
     * @return double The lower bound.
     */
    virtual double getLowerBoundForParamIndex(int idx) const;

    /**
     * @brief Get the upper bound for a specific parameter index.
     * @param idx The index of the parameter.
     * @return double The upper bound.
     */
    virtual double getUpperBoundForParamIndex(int idx) const;


private:
    /** @brief Shared pointer to the AgeSEPAIHRDModel instance being managed. */
    std::shared_ptr<AgeSEPAIHRDModel> model_;
    /** @brief Ordered list of parameter names that this manager handles. */
    std::vector<std::string> param_names_;
    /** @brief Map of parameter names to their proposal standard deviations for MCMC or optimization. */
    std::map<std::string, double> proposal_sigmas_;
    /** @brief Map of parameter names to their allowed lower and upper bounds (pair<double, double>). */
    std::map<std::string, std::pair<double, double>> param_bounds_;

    /**
     * @brief Helper function to get the index of a parameter given its name.
     * @param name The name of the parameter.
     * @return The index of the parameter in param_names_, or -1 if not found.
     */
    int getIndexForParam(const std::string& name) const;
};

} // namespace epidemic

#endif // SEPAIHRD_PARAMETER_MANAGER_HPP