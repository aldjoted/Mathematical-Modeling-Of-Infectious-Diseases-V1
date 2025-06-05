#ifndef SEPAIHRD_OBJECTIVE_FUNCTION_HPP
#define SEPAIHRD_OBJECTIVE_FUNCTION_HPP

#include "sir_age_structured/interfaces/IObjectiveFunction.hpp"
#include "sir_age_structured/interfaces/IParameterManager.hpp"
#include "sir_age_structured/interfaces/ISimulationCache.hpp"
#include "model/AgeSEPAIHRDModel.hpp"
#include "model/AgeSEPAIHRDsimulator.hpp"
#include "utils/GetCalibrationData.hpp"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <memory>
#include <limits> 
#include <future>

namespace epidemic {

    /**
     * @brief Calculates the Poisson log-likelihood objective for the AgeSEPAIHRDModel.
     *
     * The likelihood function is defined as:
     *
     * log L(Θ | data) = ∑[ y(t,type,i) × log( m(t,type,i) + ε ) - ( m(t,type,i) + ε ) ]
     *
     * where:
     * - y(t,type,i) are the observed incidence counts 
     * - m(t,type,i) are the model-predicted daily incidence flows 
     * - ε is a small constant to avoid log(0)
     *
     */
    class SEPAIHRDObjectiveFunction : public IObjectiveFunction {
    public:
        /**
         * @brief Constructor.
         * @param model Shared pointer to the AgeSEPAIHRDModel.
         * @param parameterManager Reference to the parameter manager.
         * @param cache Reference to the simulation cache.
         * @param calibration_data Reference to the observed calibration data.
         * @param time_points Vector of time points for simulation output.
         * @param initial_state The initial state vector for simulations.
         * @param solver_strategy Shared pointer to the ODE solver strategy.
         * @param abs_error Absolute error tolerance for the simulator.
         * @param rel_error Relative error tolerance for the simulator.
         */
        SEPAIHRDObjectiveFunction(
            std::shared_ptr<AgeSEPAIHRDModel> model,
            IParameterManager& parameterManager,
            ISimulationCache& cache,
            const CalibrationData& calibration_data,
            const std::vector<double>& time_points,
            const Eigen::VectorXd& initial_state,
            std::shared_ptr<IOdeSolverStrategy> solver_strategy,
            double abs_error = 1.0e-6,
            double rel_error = 1.0e-6);
    
        /**
         * @brief Calculates the overall log-likelihood for the provided parameters.
         * @param parameters A vector of model parameters to be calibrated.
         * @return The total Poisson log-likelihood.
         */
        double calculate(const Eigen::VectorXd& parameters) override;
    
        /**
         * @brief Returns the names of the parameters used in the calibration.
         * @return A reference to a vector of parameter names.
         */
        const std::vector<std::string>& getParameterNames() const override;

    protected:
        /**
         * @brief Calculates the Poisson log-likelihood for a single type of incidence data.
         * @param simulated The model-predicted incidence matrix.
         * @param observed The observed incidence matrix.
         * @param dataTypeForLog A string identifying the data type for logging purposes (e.g., "Hospitalizations").
         * @return The Poisson log-likelihood for the given data.
         *
         * This method uses a small epsilon to avoid issues with log(0) and skips negative or NaN values in observed data.
         */
        double calculateSingleLogLikelihood(const Eigen::MatrixXd& simulated,
                                            const Eigen::MatrixXd& observed,
                                            const std::string& dataTypeForLog) const;

    private:
        /** @brief Shared pointer to the epidemiological model. */
        std::shared_ptr<AgeSEPAIHRDModel> model_;
        /** @brief Reference to the manager for model parameters. */
        IParameterManager& parameterManager_;
        /** @brief Reference to the cache for simulation results and likelihoods. */
        ISimulationCache& cache_;
        /** @brief Reference to the observed calibration data. */
        const CalibrationData& observed_data_;
        /** @brief Vector of time points for simulation. */
        std::vector<double> time_points_;
        /** @brief Initial state vector for simulations. */
        Eigen::VectorXd initial_state_;
        /** @brief Shared pointer to the ODE solver strategy. */
        std::shared_ptr<IOdeSolverStrategy> solver_strategy_;
        /** @brief Absolute error tolerance for the ODE solver. */
        double abs_err_;
        /** @brief Relative error tolerance for the ODE solver. */
        double rel_err_;
    
        // Internal simulator instance
        /** @brief Unique pointer to the internal ODE simulator instance. */
        std::unique_ptr<AgeSEPAIHRDSimulator> simulator_;

        // For Suggestion 1: Cache-Aware Computation (intermediate simulation results)
        /** @brief Structure to cache intermediate simulation results (I, H, D compartments) and associated parameters. */
        struct CachedSimulationData {
            /** @brief Cached simulation data for Infected (I), Hospitalized (H), and Deceased (D) compartments. */
            Eigen::MatrixXd I_data, H_data, D_data;
            /** @brief Parameters for which the simulation data (I_data, H_data, D_data) was cached. */
            Eigen::VectorXd parameters_cache_key;
            /** @brief Flag indicating if the cache is populated with valid data. True if populated, false otherwise. */
            bool populated = false;
            /** @brief Tolerance for comparing parameter vectors to determine cache validity. */
            static constexpr double PARAM_CACHE_TOLERANCE = 1e-10;

            /**
             * @brief Checks if the cached data is valid for the given set of current parameters.
             * @param current_params The current parameters to check against the cached parameters.
             * @return True if the cache is populated and parameters match within tolerance, false otherwise.
             */
            bool isValid(const Eigen::VectorXd& current_params) const {
                if (!populated || parameters_cache_key.size() != current_params.size()) {
                    return false;
                }
                if (parameters_cache_key.size() == 0) {
                    return current_params.size() == 0;
                }
                return (parameters_cache_key - current_params).norm() < PARAM_CACHE_TOLERANCE;
            }

            /** @brief Invalidates the cached simulation data, marking it as not populated. */
            void invalidate() {
                populated = false;
            }
        };
        /** @brief Cache for intermediate simulation results (I, H, D compartments). */
        mutable CachedSimulationData cached_sim_data_;

        // Pre-allocated matrices for incidence calculations
        /** @brief Pre-allocated mutable matrix for storing simulated hospitalizations. */
        mutable Eigen::MatrixXd simulated_hospitalizations_;
        /** @brief Pre-allocated mutable matrix for storing simulated ICU admissions. */
        mutable Eigen::MatrixXd simulated_icu_admissions_;
        /** @brief Pre-allocated mutable matrix for storing simulated deaths. */
        mutable Eigen::MatrixXd simulated_deaths_;

        /** @brief Preallocates internal matrices based on model and time points. */
        void preallocateInternalMatrices();
    
        /**
         * @brief Ensures that the internal simulator instance exists. Creates it if necessary.
         *
         * @throws SimulationException if the simulator creation fails.
         */
        void ensureSimulatorExists();

    };
    
} // namespace epidemic

#endif // SEPAIHRD_OBJECTIVE_FUNCTION_HPP