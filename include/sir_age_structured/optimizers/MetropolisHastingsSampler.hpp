#ifndef EPIDEMIC_METROPOLIS_HASTINGS_SAMPLER_HPP
#define EPIDEMIC_METROPOLIS_HASTINGS_SAMPLER_HPP

#include "sir_age_structured/interfaces/IOptimizationAlgorithm.hpp"
#include "sir_age_structured/interfaces/IObjectiveFunction.hpp"
#include "sir_age_structured/interfaces/IParameterManager.hpp"
#include "sir_age_structured/SimulationResult.hpp" 

#include <Eigen/Dense>
#include <vector>
#include <deque>
#include <map>
#include <string>
#include <random>
#include <memory>

namespace epidemic {

    /**
     * @brief A Bayesian Valid Adaptive Metropolis (AM) Sampler.
     * 
     * THEORETICAL BASIS:
     * Implements the Adaptive Metropolis algorithm by Haario et al. (2001).
     * "An adaptive Metropolis algorithm" - Bernoulli 7(2), 2001.
     * 
     * This sampler learns the covariance structure of the posterior distribution
     * on-the-fly to ensure efficient mixing in high dimensions (d > 60).
     * 
     * FEATURES:
     * - Adaptively updates proposal covariance Sigma based on chain history.
     * - Uses optimal scaling factor (2.38^2 / d) from Roberts & Rosenthal (2001).
     * - Dual adaptation: covariance learning + global scale adjustment.
     * - Targets optimal acceptance rate (~23.4%) via Robbins-Monro updates.
     * - Strictly operates in Log-Space to prevent numerical underflow.
     * - Enforces detailed balance (Ergodicity) via proper MH acceptance ratio.
     * - Supports warm-start from Phase 1 covariance for faster convergence.
     * 
     * RUNTIME OPTIMIZATIONS:
     * - Rank-1 covariance updates (Welford-style) after burn-in.
     * - Cholesky decomposition updated periodically, not every iteration.
     * - Early rejection for out-of-bounds proposals (skip expensive likelihood).
     * - Optional parallel evaluation of proposals (cloud-based).
     * 
     * THEORETICAL GUARANTEES:
     * - Ergodicity: Preserved via symmetric random walk proposal.
     * - Detailed Balance: Log-space MH ratio with proper acceptance.
     * - Diminishing Adaptation: Adaptation rate gamma(t) -> 0 as t -> infinity.
     */
    class MetropolisHastingsSampler : public IOptimizationAlgorithm {
    public:
        MetropolisHastingsSampler();
        virtual ~MetropolisHastingsSampler() = default;

        /**
         * @brief Configures the Adaptive Metropolis parameters.
         * 
         * @param settings Map of configuration keys:
         *   - "mcmc_iterations": Total MCMC steps (default: 10000)
         *   - "burn_in": Steps before adaptation begins (default: 1000)
         *   - "adaptation_period": Steps between covariance updates (default: 100)
         *   - "report_interval": Steps between progress reports (default: 100)
         *   - "thinning": Store every N-th sample (default: 1, not recommended for inference)
         *   - "regularization_epsilon": Regularization for covariance (default: 1e-6)
         *   - "target_acceptance_rate": Target acceptance rate (default: 0.234)
         *   - "adapt_scale": Enable global scale adaptation (default: 1)
         *   - "store_samples": Whether to store samples (default: 1)
         *   - "write_checkpoints": Whether to write checkpoint files (default: 1)
         *   - "write_trace": Whether to write final trace file (default: 1)
         */
        void configure(const std::map<std::string, double>& settings) override;

        /**
         * @brief Set initial covariance from Phase 1 for warm-starting the MCMC.
         * This allows the sampler to immediately use the correlation structure
         * learned during optimization, significantly reducing burn-in time.
         * @param cov The learned covariance matrix from Phase 1 (PSO/HillClimbing).
         */
        void setInitialCovariance(const Eigen::MatrixXd& cov);

        /**
         * @brief Executes the Adaptive Metropolis algorithm.
         * 
         * @param initialParameters Starting point (typically from Phase 1 optimization).
         * @param objectiveFunction Log-posterior function (returns log-likelihood).
         * @param parameterManager Manages parameter bounds and constraints.
         * @return OptimizationResult containing the full posterior trace.
         */
        OptimizationResult optimize(
            const Eigen::VectorXd& initialParameters,
            IObjectiveFunction& objectiveFunction,
            IParameterManager& parameterManager) override;

    private:
        // --- Core Algorithm Logic ---
        
        /**
         * @brief Generates a proposal from N(current_params, scale² * Sigma).
         * Uses Cholesky decomposition L where Sigma = L*L^T.
         * @param currentParams Current position in parameter space.
         * @return Proposed parameter vector.
         */
        Eigen::VectorXd generateProposal(const Eigen::VectorXd& currentParams);

        /**
         * @brief Adapts the global scaling factor based on acceptance rate.
         * Uses Robbins-Monro stochastic approximation to target optimal acceptance.
         * log(scale) += gamma * (accept - target_rate)
         * @param accepted Whether the last proposal was accepted.
         * @param step Current iteration number.
         */
        void adaptGlobalScale(bool accepted, int step);

        /**
         * @brief Updates the proposal covariance matrix using the chain history.
         * 
         * Implementation of the Haario et al. (2001) recursive update with
         * rank-1 updates for efficiency. Uses diminishing adaptation rate
         * gamma(t) = 10.0 / (t + 100) to ensure ergodicity.
         * 
         * @param step Current iteration number.
         * @param n_params Number of parameters.
         */
        void updateCovarianceRank1(int step, int n_params);

        /**
         * @brief Performs full covariance recomputation from chain history.
         * Called periodically to correct accumulated numerical errors.
         * @param n_params Number of parameters.
         */
        void recomputeFullCovariance(int n_params);

        /**
         * @brief Check if parameters are within valid bounds.
         * @param params Parameter vector to check.
         * @param pm Parameter manager with bounds.
         * @return true if all parameters within bounds, false otherwise.
         */
        bool areParametersValid(const Eigen::VectorXd& params, 
                                IParameterManager& pm) const;

        /**
         * @brief Safe evaluation wrapper (handles NaNs and exceptions).
         * @param func Objective function.
         * @param p Parameter vector.
         * @return Log-likelihood or -1e18 on failure.
         */
        double safeEvaluate(IObjectiveFunction& func, const Eigen::VectorXd& p);

        // --- I/O ---
        void saveCheckpoint(const OptimizationResult& res, 
                            IParameterManager& pm, 
                            bool final = false);

        void saveSamplesToCSV(
            const std::vector<Eigen::VectorXd>& samples,
            const std::vector<double>& objectiveValues,
            const std::vector<std::string>& parameterNames,
            const std::string& filepath);

        // --- Configuration ---
        int iterations_ = 10000;
        int burn_in_ = 1000;
        int adaptation_period_ = 100;     // Update covariance every N steps
        int report_interval_ = 100;
        int thinning_ = 1;
        double regularization_epsilon_ = 1e-6; // Prevents singular matrices
        double target_acceptance_rate_ = 0.234; // Optimal for Gaussian targets (Roberts 2001)
        bool adapt_scale_ = true;              // Enable global scale adaptation

        // --- State ---
        std::mt19937 gen_;
        std::vector<Eigen::VectorXd> chain_history_; // Stores history for covariance calculation
        
        // Adaptive MCMC Matrices
        Eigen::MatrixXd proposal_cholesky_;    // L matrix for generating proposals
        Eigen::MatrixXd current_covariance_;   // Estimated covariance of the posterior
        Eigen::VectorXd running_mean_;         // Running mean for rank-1 updates
        
        // Global scale adaptation (Robbins-Monro)
        double log_scale_ = 0.0;               // log(global_scale), adapted online
        double global_scale_ = 1.0;            // exp(log_scale_), multiplies proposals
        
        // Emergency adaptation state
        std::deque<int> recent_accepts_;       // Sliding window of recent accept/reject (1/0)
        int emergency_shrink_count_ = 0;       // Count of emergency scale reductions

        // Optional I/O controls
        bool store_samples_ = true;
        bool write_checkpoints_ = true;
        bool write_trace_ = true;
        
        // Initial covariance from Phase 1 (if provided)
        Eigen::MatrixXd initialCovariance_;
        bool hasInitialCovariance_ = false;
    };

} // namespace epidemic

#endif // EPIDEMIC_METROPOLIS_HASTINGS_SAMPLER_HPP