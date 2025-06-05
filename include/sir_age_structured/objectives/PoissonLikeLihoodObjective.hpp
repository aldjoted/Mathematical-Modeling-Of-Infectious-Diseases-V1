#ifndef POISSON_LIKELIHOOD_OBJECTIVE_HPP
#define POISSON_LIKELIHOOD_OBJECTIVE_HPP

#include "sir_age_structured/interfaces/IObjectiveFunction.hpp"
#include "sir_age_structured/interfaces/IParameterManager.hpp"
#include "sir_age_structured/interfaces/ISimulationCache.hpp"
#include "sir_age_structured/interfaces/IEpidemicModel.hpp"
#include "sir_age_structured/Simulator.hpp"
#include "exceptions/Exceptions.hpp"
#include "utils/GetCalibrationData.hpp"
#include <Eigen/Core>
#include <memory>
#include <vector>
#include <string>
#include <limits>
#include <cmath>
#include <iostream>
#include <optional>

namespace epidemic {

    /**
     * @brief Calculates the Poisson log-likelihood objective function for epidemic model calibration.
     *
     * This class evaluates how well the simulated incidence data from an epidemic model
     * matches the observed incidence data, assuming a Poisson distribution for the observed counts.
     * It uses a simulator, parameter manager, and caching mechanism to efficiently compute
     * the likelihood for different parameter sets.
     */
    class PoissonLikelihoodObjective : public IObjectiveFunction {
        public:
            /**
             * @brief Constructs a PoissonLikelihoodObjective instance.
             *
             * @param[in] model A shared pointer to the epidemic model implementing IEpidemicModel.
             * @param[in] simulator A reference to the Simulator instance used to run simulations.
             * @param[in] parameterManager A reference to the parameter manager implementing IParameterManager.
             * @param[in] cache A reference to the simulation cache implementing ISimulationCache.
             * @param[in] calibrationData A constant reference to the CalibrationData object containing observed data.
             * @param[in] timePoints A constant reference to the vector of time points for the simulation.
             * @param[in] initialState A constant reference to the initial state vector for the simulation.
             * @param[in] compartmentForLikelihood The name of the compartment used for likelihood calculation (default: "I"). Currently unused in the provided implementation but kept for potential future use.
             * @throws InvalidParameterException If the model pointer is null, the timePoints vector is empty, or the size of timePoints does not match the rows of the observed incidence data.
             */
            PoissonLikelihoodObjective(
                std::shared_ptr<IEpidemicModel> model,
                Simulator& simulator,
                IParameterManager& parameterManager,
                ISimulationCache& cache,
                const CalibrationData& calibrationData,
                const std::vector<double>& timePoints,
                const Eigen::VectorXd& initialState,
                std::string compartmentForLikelihood = "I");

            /**
             * @brief Calculates the Poisson log-likelihood for a given set of parameters.
             *
             * Runs a simulation with the provided parameters (using caching if available),
             * extracts the simulated incidence, and computes the log-likelihood against
             * the observed incidence data.
             *
             * @param[in] parameters An Eigen::VectorXd containing the parameter values to evaluate.
             * @return The calculated Poisson log-likelihood value. Returns -infinity if the simulation fails,
             *         produces invalid results, encounters parameter errors, model errors, or if the
             *         calculated likelihood is NaN or infinite.
             * @throws InvalidParameterException If parameter validation within parameterManager_ fails.
             * @throws SimulationException If the simulation run fails or if the dimensions of simulated and observed data mismatch.
             * @throws InvalidResultException If the simulation result is marked as invalid.
             * @throws ModelException If the model provides inconsistent information (e.g., non-positive age classes/state size, state size not divisible by age classes).
             * @throws std::exception For other potential errors during simulation or calculation.
             */
            double calculate(const Eigen::VectorXd& parameters) override;

            /**
             * @brief Gets the names of the parameters managed by the associated IParameterManager.
             *
             * @return A constant reference to a vector of strings containing the parameter names.
             */
            const std::vector<std::string>& getParameterNames() const override;

        private:
            ///< Shared pointer to the epidemic model.
            std::shared_ptr<IEpidemicModel> model_;
            ///< Reference to the simulator instance.
            Simulator& simulator_;
            ///< Reference to the parameter manager instance.
            IParameterManager& parameterManager_;
            ///< Reference to the simulation cache instance.
            ISimulationCache& cache_;
            ///< Constant reference to the observed calibration data.
            const CalibrationData& observedData_;
            ///< Constant reference to the simulation time points.
            const std::vector<double>& timePoints_;
            ///< Constant reference to the initial state vector.
            const Eigen::VectorXd& initialState_;
            ///< Name of the compartment used for likelihood calculation.
            std::string compartmentForLikelihood_;
            ///< Matrix holding the observed incidence data extracted from calibrationData.
            Eigen::MatrixXd observedIncidence_;
            ///< Vector storing the names of the parameters being optimized.
            std::vector<std::string> parameterNames_;

            /**
             * @brief Calculates the Poisson log-likelihood sum based on simulated and observed incidence.
             *
             * Formula: sum(y_obs * log(y_sim) - y_sim) over all time points and age groups.
             * Ensures y_sim is at least 1e-9 to avoid log(0).
             *
             * @param[in] simulated_incidence An Eigen::MatrixXd containing the incidence values from the simulation.
             * @param[in] observed_incidence An Eigen::MatrixXd containing the observed incidence values.
             * @return The calculated log-likelihood sum as a double. Returns -infinity if inputs contain non-finite values or if the final sum is NaN or infinite.
             * @throws InvalidParameterException If the dimensions of simulated_incidence and observed_incidence do not match, or if either matrix is empty.
             */
            double calculate_log_likelihood(const Eigen::MatrixXd& simulated_incidence,
                const Eigen::MatrixXd& observed_incidence);

            };

}// namespace epidemic
#endif // POISSON_LIKELIHOOD_OBJECTIVE_HPP