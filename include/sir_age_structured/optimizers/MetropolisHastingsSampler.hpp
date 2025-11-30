#ifndef EPIDEMIC_METROPOLIS_HASTINGS_SAMPLER_HPP
#define EPIDEMIC_METROPOLIS_HASTINGS_SAMPLER_HPP

#include "sir_age_structured/interfaces/IOptimizationAlgorithm.hpp"
#include "sir_age_structured/interfaces/IObjectiveFunction.hpp"
#include "sir_age_structured/interfaces/IParameterManager.hpp"
#include "sir_age_structured/SimulationResult.hpp" 

#include <Eigen/Dense>
#include <vector>
#include <map>
#include <string>
#include <random>
#include <memory>

namespace epidemic {

    /**
     * @brief A Metropolis-Hastings Sampler enhanced with Simulated Annealing.
     * * This optimizer abandons greedy hill-climbing in favor of a stochastic process
     * that can cross low-likelihood valleys.
     * * Key Features:
     * 1. Perturbation Strategies:
     * - Random Step All: Correlated perturbation of all parameters (Adaptive Metropolis).
     * - Random Step One: Axis-aligned perturbation of a single parameter.
     * * 2. Simulated Annealing:
     * - Accepts worse solutions with probability P = exp((L_new - L_old) / T).
     * - Temperature T decays over time (Cooling Schedule).
     */
    class MetropolisHastingsSampler : public IOptimizationAlgorithm {
    public:
        MetropolisHastingsSampler();
        virtual ~MetropolisHastingsSampler() = default;

        /**
         * @brief Configures the optimizer.
         * Keys:
         * - "mcmc_iterations": Total steps (default: 5000)
         * - "sa_initial_temp": Starting temperature for annealing (default: 100.0)
         * - "sa_cooling_rate": Geometric cooling factor per step (default: 0.995)
         * - "step_all_prob": Probability of choosing 'Step All' vs 'Step One' (default: 0.3)
         * - "cloud_size_multiplier": Cloud size = num_threads * multiplier (default: 2)
         */
        void configure(const std::map<std::string, double>& settings) override;

        /**
         * @brief Set initial covariance from Phase 1 for warm-starting the MCMC.
         * This allows the sampler to immediately use the correlation structure
         * learned during optimization, avoiding redundant burn-in.
         * @param cov The learned covariance matrix from Phase 1 (HillClimbingOptimizer).
         */
        void setInitialCovariance(const Eigen::MatrixXd& cov);

        OptimizationResult optimize(
            const Eigen::VectorXd& initialParameters,
            IObjectiveFunction& objectiveFunction,
            IParameterManager& parameterManager) override;

    private:
        // --- Helper Methods ---
        
        // Perturbs all parameters (correlated using covariance)
        Eigen::VectorXd randomStepAll(
            const Eigen::VectorXd& currentParams,
            const Eigen::MatrixXd& choleskyCov,
            double scale,
            IParameterManager& pm);

        // Perturbs exactly one parameter (axis-aligned)
        Eigen::VectorXd randomStepOne(
            const Eigen::VectorXd& currentParams,
            IParameterManager& pm);

        // Safe evaluation wrapper (handles NaNs)
        double safeEvaluate(
            IObjectiveFunction& func, 
            const Eigen::VectorXd& p);

        /**
         * @brief Helper to save the full optimization trace to a CSV file.
         */
        void saveSamplesToCSV(
            const std::vector<Eigen::VectorXd>& samples,
            const std::vector<double>& objectiveValues,
            const std::vector<std::string>& parameterNames,
            const std::string& filepath);

        /**
         * @brief Helper to save an intermediate checkpoint to disk for recovery or monitoring.
         */
        void saveCheckpoint(const OptimizationResult& res, 
                            IParameterManager& pm, 
                            bool final = false);

        // --- Helper for parallel cloud generation ---
        Eigen::VectorXd generateCandidate(
            const Eigen::VectorXd& currentParams,
            const Eigen::MatrixXd& choleskyCov,
            double scale,
            IParameterManager& pm,
            std::mt19937& localGen);

        // --- Configuration ---
        int iterations_ = 5000;
        int report_interval_ = 100;
        int thinning_ = 1;
        int cloud_size_multiplier_ = 2; // Cloud size = num_threads * multiplier
        
        // Simulated Annealing Settings
        double initial_temp_ = 50.0;
        double cooling_rate_ = 0.999;
        double step_all_prob_ = 0.4; // 40% Global moves, 60% Local moves

        // RNG
        std::mt19937 gen_;
        
        // Initial covariance from Phase 1 (if provided)
        Eigen::MatrixXd initialCovariance_;
        bool hasInitialCovariance_ = false;
    };

} // namespace epidemic

#endif // EPIDEMIC_METROPOLIS_HASTINGS_SAMPLER_HPP