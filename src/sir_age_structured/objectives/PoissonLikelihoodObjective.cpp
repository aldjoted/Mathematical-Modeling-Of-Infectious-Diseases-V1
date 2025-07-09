#include "sir_age_structured/objectives/PoissonLikeLihoodObjective.hpp"
#include "sir_age_structured/SimulationResult.hpp"
#include "sir_age_structured/SimulationResultProcessor.hpp"
#include "sir_age_structured/interfaces/IEpidemicModel.hpp"
#include "exceptions/Exceptions.hpp" 
#include <iostream>
#include <limits>
#include <cmath>
#include <optional>

namespace epidemic {
    PoissonLikelihoodObjective::PoissonLikelihoodObjective(
        std::shared_ptr<IEpidemicModel> model,
        Simulator& simulator,
        IParameterManager& parameterManager,
        ISimulationCache& cache,
        const CalibrationData& calibrationData,
        const std::vector<double>& timePoints,
        const Eigen::VectorXd& initialState,
        std::string compartmentForLikelihood)
        : model_(model),
          simulator_(simulator),
          parameterManager_(parameterManager),
          cache_(cache),
          observedData_(calibrationData),
          timePoints_(timePoints),
          initialState_(initialState),
          compartmentForLikelihood_(std::move(compartmentForLikelihood)),
          observedIncidence_(calibrationData.getNewConfirmedCases())
    {
        if (!model_) {
            THROW_INVALID_PARAM("PoissonLikelihoodObjective", "Model pointer is null.");
        }
        if (timePoints_.empty()) {
            THROW_INVALID_PARAM("PoissonLikelihoodObjective", "Time points vector is empty.");
        }
        // Ensure observedIncidence_ is initialized before this check
        if (static_cast<Eigen::Index>(timePoints_.size()) != observedIncidence_.rows()) {
             THROW_INVALID_PARAM("PoissonLikelihoodObjective",
                "Time points size (" + std::to_string(timePoints_.size()) +
                ") does not match observed data rows (" + std::to_string(observedIncidence_.rows()) + ").");
        }
        parameterNames_ = parameterManager.getParameterNames();
    }

    double PoissonLikelihoodObjective::calculate(const Eigen::VectorXd& parameters) const {
        std::optional<double> cached_result = cache_.get(parameters);
        if (cached_result.has_value()) {
            return cached_result.value();
        }

        double log_likelihood = -std::numeric_limits<double>::infinity();

        try {
            parameterManager_.updateModelParameters(parameters);
            SimulationResult result = simulator_.run(initialState_, timePoints_);
            if (!result.isValid()) {
                std::cerr << "[ObjectiveFunc] Warning: Simulation with given parameters resulted in invalid state." << std::endl;
                return log_likelihood; // Return -inf, do not cache
            }

            int total_state_size = model_->getStateSize();
            int n_ages = model_->getNumAgeClasses();
            if (n_ages <= 0) {
                throw ModelException("PoissonLikelihoodObjective::calculate", "Model returned non-positive number of age classes.");
            }
            if (total_state_size <= 0) {
                throw ModelException("PoissonLikelihoodObjective::calculate", "Model returned non-positive state size.");
            }
            if (total_state_size % n_ages != 0) {
               throw ModelException("PoissonLikelihoodObjective::calculate", "Model state size is not divisible by the number of age classes.");
            }
            int num_compartments_per_age = total_state_size / n_ages;

            Eigen::MatrixXd simulated_incidence = SimulationResultProcessor::getIncidenceData(
                result, *model_, num_compartments_per_age);

            if (simulated_incidence.rows() != observedIncidence_.rows() || simulated_incidence.cols() != observedIncidence_.cols()) {
                throw SimulationException("PoissonLikelihoodObjective::calculate",
                                          "Extracted simulated data dimensions (" + std::to_string(simulated_incidence.rows()) + "," + std::to_string(simulated_incidence.cols()) +
                                          ") mismatch observed data dimensions (" + std::to_string(observedIncidence_.rows()) + "," + std::to_string(observedIncidence_.cols()) + ").");
            }

            log_likelihood = calculate_log_likelihood(simulated_incidence, observedIncidence_);

            if (!std::isnan(log_likelihood) && !std::isinf(log_likelihood)) {
                cache_.set(parameters, log_likelihood);
            } else {
                std::cerr << "[ObjectiveFunc] Warning: Calculated log-likelihood is NaN or Inf. Not caching." << std::endl;
                log_likelihood = -std::numeric_limits<double>::infinity(); // Ensure -inf is returned
            }

        } catch (const InvalidParameterException& e) {
            std::cerr << "[ObjectiveFunc] Parameter Error: " << e.what() << std::endl;
            log_likelihood = -std::numeric_limits<double>::infinity();
        } catch (const SimulationException& e) {
            std::cerr << "[ObjectiveFunc] Simulation Error: " << e.what() << std::endl;
            log_likelihood = -std::numeric_limits<double>::infinity();
        } catch (const InvalidResultException& e) {
            std::cerr << "[ObjectiveFunc] Result Error: " << e.what() << std::endl;
            log_likelihood = -std::numeric_limits<double>::infinity();
        } catch (const ModelException& e) {
            std::cerr << "[ObjectiveFunc] Model Error: " << e.what() << std::endl;
            log_likelihood = -std::numeric_limits<double>::infinity();
        } catch (const std::exception& e) {
            std::cerr << "[ObjectiveFunc] Generic Error during simulation or likelihood calculation: " << e.what() << std::endl;
            log_likelihood = -std::numeric_limits<double>::infinity();
        }

        return log_likelihood;
    }

    double PoissonLikelihoodObjective::calculate_log_likelihood(const Eigen::MatrixXd& simulated_incidence,
        const Eigen::MatrixXd& observed_incidence) const {
            if(simulated_incidence.rows() != observed_incidence.rows() || simulated_incidence.cols() != observed_incidence.cols()) {
                THROW_INVALID_PARAM("PoissonLikelihoodObjective::calculate_log_likelihood", "Simulated and observed incidence matrices must have the same dimensions.");
            }
            if (simulated_incidence.size() == 0 || observed_incidence.size() == 0) {
                 THROW_INVALID_PARAM("PoissonLikelihoodObjective::calculate_log_likelihood", "Simulated or observed incidence matrices are empty.");
            }

            // Ensure observed values are non-negative
            Eigen::MatrixXd y_obs = observed_incidence.cwiseMax(0.0);
            // Ensure simulated values are positive to avoid log(0) or log(<0)
            Eigen::MatrixXd y_sim = simulated_incidence.cwiseMax(1e-9);

            // Check for non-finite values *after* potential adjustments
            if (!y_sim.allFinite() || !y_obs.allFinite()) {
                std::cerr << "[Likelihood] Error: Non-finite values detected in inputs after adjustments." << std::endl;
                return -std::numeric_limits<double>::infinity();
           }

           // Poisson log-likelihood: sum(y_obs * log(y_sim) - y_sim)
           Eigen::MatrixXd log_likelihood_matrix = (y_obs.array() * y_sim.array().log()) - y_sim.array();

           double log_likelihood_sum = log_likelihood_matrix.sum();

            if (std::isnan(log_likelihood_sum) || std::isinf(log_likelihood_sum)) {
                std::cerr << "[Likelihood] Error: NaN or Inf encountered in final sum." << std::endl;
                return -std::numeric_limits<double>::infinity();
            }

            return log_likelihood_sum;
    }

    const std::vector<std::string>& PoissonLikelihoodObjective::getParameterNames() const {
        return parameterNames_;
    }
}