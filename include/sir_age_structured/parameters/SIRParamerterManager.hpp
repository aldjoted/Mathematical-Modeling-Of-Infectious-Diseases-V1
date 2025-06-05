#ifndef SIR_PARAMETER_MANAGER_HPP
#define SIR_PARAMETER_MANAGER_HPP

#include "../interfaces/IParameterManager.hpp"
#include "sir_age_structured/AgeSIRModel.hpp"
#include "exceptions/Exceptions.hpp"
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <limits>

namespace epidemic {
    /**
     * @brief Manages parameters for AgeSIRModel calibration.
     *
     * Provides a unified interface for calibration algorithms to access and update the model parameters 
     * (q, scale_C_total, gamma_i). Handles parameter constraints, proposal distributions, and
     * mapping between parameter names and indices.
     */
    class SIRParameterManager : public IParameterManager {
    public:
        /**
         * @brief Constructs a parameter manager for AgeSIRModel calibration.
         * 
         * @param model Shared pointer to the model to be calibrated
         * @param params_to_calibrate List of parameter names to be calibrated 
         *        (supported: "q", "scale_C_total", "gamma_{i}" where i is age class index)
         * @param proposal_sigmas Map of parameter names to their proposal standard deviations
         *        (uses defaults if not provided)
         * 
         * @throws InvalidParameterException If model pointer is null or parameter list is empty
         * @throws ModelException If parameter names are invalid or age indices are out of range
         */
        SIRParameterManager(
            std::shared_ptr<AgeSIRModel> model,
            const std::vector<std::string>& params_to_calibrate,
            const std::map<std::string, double>& proposal_sigmas);

        /**
         * @brief Gets current parameter values from the model.
         * @return Vector of current parameter values in order of param_names_
         */
        Eigen::VectorXd getCurrentParameters() const override;

        /**
         * @brief Updates model with new parameter values.
         * @param parameters Vector of parameter values to set
         * @throws InvalidParameterException If parameter vector size mismatches
         */
        void updateModelParameters(const Eigen::VectorXd& parameters) override;

        /**
         * @brief Gets the list of parameter names being managed.
         * @return Reference to the parameter names vector
         */
        const std::vector<std::string>& getParameterNames() const override;

        /**
         * @brief Gets the number of parameters being managed.
         * @return Size of the parameter vector
         */
        size_t getParameterCount() const override;

        /**
         * @brief Gets proposal standard deviation for a parameter by index.
         * @param index Index of the parameter
         * @return Standard deviation for proposal distribution
         * @throws std::out_of_range If index is invalid
         * @throws InvalidParameterException If sigma not found for parameter
         */
        double getSigmaForParamIndex(int index) const override;

        /**
         * @brief Applies constraints to parameter values.
         * @param parameters Raw parameter vector
         * @return Constrained parameter vector
         * @throws InvalidParameterException If parameter vector size mismatches
         */
        Eigen::VectorXd applyConstraints(const Eigen::VectorXd& parameters) const override;
        /**
         * @brief Get the lower bound for a specific parameter index.
         * Assumes non-negativity for SIR parameters.
         * @param idx The index of the parameter.
         * @return double The lower bound (0.0).
         */
        double getLowerBoundForParamIndex(int idx) const override {
            if (idx < 0 || static_cast<size_t>(idx) >= param_names_.size()) {
                 throw std::out_of_range("Parameter index out of range in getLowerBoundForParamIndex");
            }
            return 0.0;
        }
        /**
         * @brief Get the upper bound for a specific parameter index.
         * Returns a specific upper bound of 2.0.
         * @param idx The index of the parameter.
         * @return double The upper bound (2.0).
         */
        double getUpperBoundForParamIndex(int idx) const override {
             if (idx < 0 || static_cast<size_t>(idx) >= param_names_.size()) {
                 throw std::out_of_range("Parameter index out of range in getUpperBoundForParamIndex");
            }
            return 2.0;
        }
        
    private:
        /** @brief Shared pointer to the AgeSIRModel instance being managed. */
        std::shared_ptr<AgeSIRModel> model_;
        /** @brief Vector of parameter names that are being calibrated. */
        std::vector<std::string> param_names_;
        /** @brief Map of parameter names to their proposal standard deviations for MCMC. */
        std::map<std::string, double> proposal_sigmas_;
        /** @brief Map of parameter names to their corresponding index in the parameter vector. */
        std::unordered_map<std::string, size_t> param_name_to_index_;

        /**
         * @brief Gets default sigma for a parameter by name.
         * @param param_name Parameter name
         * @return Default standard deviation
         * @throws InvalidParameterException If no default sigma exists for parameter
         */
        double getDefaultSigmaForParam(const std::string& param_name) const;
    };
}

#endif // SIR_PARAMETER_MANAGER_HPP