#ifndef METROPOLIS_HASTINGS_SAMPLER_HPP
#define METROPOLIS_HASTINGS_SAMPLER_HPP

#include "sir_age_structured/interfaces/IOptimizationAlgorithm.hpp"
#include "sir_age_structured/interfaces/IObjectiveFunction.hpp"
#include "sir_age_structured/interfaces/IParameterManager.hpp"
#include "exceptions/Exceptions.hpp"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <map>
#include <random>
#include <limits>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <utility>

namespace epidemic {
    
    /**
     * @brief Metropolis-Hastings MCMC sampler with optional proposal refinement.
     * 
     * Implements Random-Walk Metropolis-Hastings for parameter optimization.
     * Supports burn-in, thinning, and an optional refinement step for proposals
     * which checks both forward and potentially reversed proposal directions before local search.
     */
    class MetropolisHastingsSampler : public IOptimizationAlgorithm  {
    public:
        /**
         * @brief Initializes the random number generator.
         */
        MetropolisHastingsSampler();

        /**
         * @brief Performs MCMC optimization.
         * 
         * @param initialParameters Starting parameters for the chain.
         * @param objectiveFunction Function to be optimized (maximized).
         * @param parameterManager Manages parameter constraints and properties.
         * @return OptimizationResult Best parameters, objective value, and samples.
         * @throws InvalidParameterException If parameter vector sizes mismatch.
         */
        OptimizationResult optimize(
            const Eigen::VectorXd& initialParameters,
            IObjectiveFunction& objectiveFunction,
            IParameterManager& parameterManager) override;

        /**
         * @brief Configures sampler parameters.
         * 
         * Settings include:
         * - `burn_in`: Iterations to discard (default: 5000).
         * - `mcmc_iterations`: Iterations post-burn-in (default: 10000).
         * - `thinning`: Sample collection interval (default: 10).
         * - `mcmc_step_size`: Proposal step size coefficient (default: 0.1).
         * - `calculate_posterior_mean`: Compute posterior mean (default: true).
         * - `report_interval`: Progress reporting frequency (default: 200).
         * - `mcmc_enable_refinement`: Enable proposal refinement (0/1, default: 0).
         * - `mcmc_refinement_steps`: Binary refinement steps if enabled (default: 0).
         * 
         * @param settings Map of setting names to values.
         */
        void configure(const std::map<std::string, double>& settings) override;

    private:
        /** @brief Number of iterations to discard at the beginning of the MCMC chain (burn-in period). */
        int burn_in_ = 5000;
        /** @brief Number of MCMC iterations to perform after the burn-in period. */
        int mcmc_iterations_ = 10000;
        /** @brief Interval for collecting samples from the MCMC chain (e.g., collect every 10th sample). */
        int thinning_ = 10;
        /** @brief Coefficient for the proposal step size in the MCMC algorithm. */
        double step_size_ = 0.1;
        /** @brief Flag indicating whether to calculate the posterior mean of the parameters. */
        bool calculate_posterior_mean_ = true;
        /** @brief Frequency for reporting progress during MCMC sampling. */
        int report_interval_ = 200;
        /** @brief Flag indicating whether to enable proposal refinement. */
        bool enable_refinement_ = false;
        /** @brief Number of binary refinement steps to perform if refinement is enabled. */
        int refinement_steps_ = 0;

        /** @brief Mersenne Twister 19937 random number generator. */
        std::mt19937 gen_;
        /** @brief Uniform real distribution for generating random numbers between 0.0 and 1.0. */
        std::uniform_real_distribution<> unif_;

        /**
         * @brief Perturbs all parameters with Gaussian noise.
         * @param params_vec Vector to modify.
         * @param step_coef Step size coefficient.
         * @param paramManager Parameter manager.
         */
        void randomStepAll(Eigen::VectorXd& params_vec, double step_coef, IParameterManager& paramManager);

        /**
         * @brief Perturbs one randomly chosen parameter with Gaussian noise.
         * @param params_vec Vector to modify.
         * @param step_coef Step size coefficient.
         * @param paramManager Parameter manager.
         */
        void randomStepOne(Eigen::VectorXd& params_vec, double step_coef, IParameterManager& paramManager);

        /**
         * @brief Locally refines a proposal to potentially improve its objective value.
         * 
         * Attempts elongation along the step direction, followed by binary search-like steps.
         * 
         * @param P1_initial Initial proposed parameter vector.
         * @param P1_initial_logL Log-likelihood of P1_initial.
         * @param step_direction Direction vector (P1_initial - P0_current_params).
         * @param num_refinement_steps Number of binary refinement iterations.
         * @param objectiveFunction The objective function.
         * @param parameterManager The parameter manager.
         * @return Pair of (best refined parameters, their log-likelihood).
         */
        std::pair<Eigen::VectorXd, double> perform_refinement(
            const Eigen::VectorXd& P1_initial,
            double P1_initial_logL,
            const Eigen::VectorXd& step_direction,
            int num_refinement_steps,
            IObjectiveFunction& objectiveFunction,
            IParameterManager& parameterManager);

        /**
         * @brief Saves MCMC samples and objective values to a CSV file.
         * @param samples Collected parameter samples.
         * @param objectiveValues Objective values for each sample.
         * @param parameterNames Names of the parameters.
         * @param filepath Path to the output CSV file.
         */
        void saveSamplesToCSV(
            const std::vector<Eigen::VectorXd>& samples,
            const std::vector<double>& objectiveValues,
            const std::vector<std::string>& parameterNames,
            const std::string& filepath
        );
    };

} // namespace epidemic

#endif // METROPOLIS_HASTINGS_SAMPLER_HPP