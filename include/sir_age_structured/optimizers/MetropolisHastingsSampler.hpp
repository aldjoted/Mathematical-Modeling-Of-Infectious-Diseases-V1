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
#include <execution>
#include <thread>
#include <atomic>

namespace epidemic {
    
    /**
     * @brief Enhanced Metropolis-Hastings MCMC sampler with adaptive proposals and performance optimizations
     * 
     * This class implements a high-performance Metropolis-Hastings algorithm with several 
     * advanced features including vectorized operations, parallel evaluation, adaptive step sizes,
     * and optional proposal refinement. The implementation is based on the adaptive proposal
     * strategies described in Cai et al. (2008) with additional modern optimizations.
     * 
     * @details
     * Key features:
     * - **Adaptive Proposals**: Dynamically adjusts proposal step size to maintain optimal acceptance rate
     * - **Vectorization**: Uses SIMD operations for efficient batch processing of proposals
     * - **Parallelization**: Optional parallel evaluation of objective function for multiple proposals
     * - **Proposal Refinement**: Optional elongation and binary search refinement of proposals
     * - **Mixed Update Strategy**: Alternates between updating all parameters and single parameters
     * 
     * The sampler supports both burn-in and thinning to ensure proper convergence and reduce
     * autocorrelation in the collected samples. It can calculate posterior means and automatically
     * saves samples to CSV files for further analysis.
     * 
     * @note Thread-safety: The class uses atomic counters for statistics tracking when parallel
     * execution is enabled. The random number generator is not thread-safe and should not be
     * accessed concurrently.
     * 
     * @references
     * Cai, B., Meyer, R., & Perron, F. (2008). Metropolis–Hastings algorithms with adaptive proposals. 
     * Statistics and Computing, 18, 421-433. https://doi.org/10.1007/s11222-008-9051-5
     * 
     * @see IOptimizationAlgorithm
     */
    class MetropolisHastingsSampler : public IOptimizationAlgorithm {
    public:
        /**
         * @brief Constructs a Metropolis-Hastings sampler with default settings
         * 
         * Initializes the random number generator with a random seed from the system's
         * random device. Default settings are configured for general-purpose MCMC sampling.
         */
        MetropolisHastingsSampler();

        /**
         * @brief Performs MCMC optimization to find the maximum of the objective function
         * 
         * Runs the Metropolis-Hastings algorithm starting from the given initial parameters.
         * The algorithm consists of a burn-in phase followed by the main sampling phase.
         * Samples are collected according to the thinning interval after burn-in.
         * 
         * @param initialParameters Starting point for the Markov chain (must match dimension expected by objective function)
         * @param objectiveFunction Function to be maximized (typically log-likelihood or log-posterior)
         * @param parameterManager Handles parameter constraints and transformations
         * 
         * @return OptimizationResult containing:
         *         - bestParameters: Parameters with highest objective value encountered
         *         - bestObjectiveValue: Maximum objective value found
         *         - samples: Vector of parameter samples collected after burn-in
         *         - sampleObjectiveValues: Objective values corresponding to samples
         * 
         * @throws InvalidParameterException if initial parameters have incorrect dimension
         * @throws std::runtime_error if objective function evaluation fails consistently
         * 
         * @note The algorithm automatically saves samples to a CSV file in data/mcmc_samples/
         */
        OptimizationResult optimize(
            const Eigen::VectorXd& initialParameters,
            IObjectiveFunction& objectiveFunction,
            IParameterManager& parameterManager) override;

        /**
         * @brief Configures the sampler with custom settings
         * 
         * Allows customization of all sampler parameters through a key-value map.
         * Unspecified settings retain their default values.
         * 
         * @param settings Map of setting names to values. Supported settings:
         * 
         * **Core MCMC Settings:**
         * - `burn_in`: Number of initial iterations to discard (default: 5000)
         * - `mcmc_iterations`: Number of iterations after burn-in (default: 10000)
         * - `thinning`: Interval for collecting samples, e.g., every 10th (default: 10)
         * - `mcmc_step_size`: Initial proposal step size coefficient (default: 0.1)
         * - `calculate_posterior_mean`: Whether to compute posterior mean (0/1, default: 1)
         * - `report_interval`: Iterations between progress reports (default: 200)
         * 
         * **Refinement Settings:**
         * - `mcmc_enable_refinement`: Enable proposal refinement (0/1, default: 0)
         * - `mcmc_refinement_steps`: Number of binary refinement iterations (default: 0)
         * 
         * **Performance Settings:**
         * - `mcmc_enable_parallel`: Enable parallel objective evaluation (0/1, default: 0)
         * - `mcmc_num_threads`: Number of threads for parallel execution (default: hardware concurrency)
         * - `mcmc_enable_vectorization`: Use vectorized operations (0/1, default: 1)
         * - `mcmc_vector_batch_size`: Size of vectorized batches (default: 64)
         * 
         * **Adaptive Settings:**
         * - `mcmc_adaptive_step`: Enable adaptive step size (0/1, default: 1)
         * - `mcmc_target_acceptance`: Target acceptance rate for adaptation (default: 0.234)
         * - `mcmc_adaptation_rate`: Learning rate for step size adaptation (default: 0.05)
         * 
         * @example
         * ```cpp
         * std::map<std::string, double> settings = {
         *     {"burn_in", 10000},
         *     {"mcmc_iterations", 50000},
         *     {"mcmc_enable_parallel", 1},
         *     {"mcmc_num_threads", 8}
         * };
         * sampler.configure(settings);
         * ```
         */
        void configure(const std::map<std::string, double>& settings) override;

    private:
        // ============ Configuration Parameters ============
        
        /** @brief Number of initial iterations to discard for chain convergence */
        int burn_in_ = 5000;
        
        /** @brief Number of MCMC iterations to perform after burn-in */
        int mcmc_iterations_ = 10000;
        
        /** @brief Thinning interval: collect every nth sample to reduce autocorrelation */
        int thinning_ = 10;
        
        /** @brief Current proposal step size coefficient (may be adapted during sampling) */
        double step_size_ = 0.1;
        
        /** @brief Whether to calculate and test the posterior mean as potential optimum */
        bool calculate_posterior_mean_ = true;
        
        /** @brief Number of iterations between progress reports */
        int report_interval_ = 200;
        
        /** @brief Whether to use proposal refinement (elongation + binary search) */
        bool enable_refinement_ = false;
        
        /** @brief Number of binary search refinement steps if refinement is enabled */
        int refinement_steps_ = 0;
        
        // ============ Parallelization Settings ============
        
        /** @brief Whether to use parallel evaluation of objective function */
        bool enable_parallel_ = false;
        
        /** @brief Number of threads to use for parallel execution */
        int num_threads_ = std::thread::hardware_concurrency();
        
        // ============ Vectorization Settings ============
        
        /** @brief Whether to use vectorized proposal generation and evaluation */
        bool enable_vectorized_proposals_ = true;
        
        /** @brief Number of proposals to generate in vectorized batches */
        int vector_batch_size_ = 64;
        
        // ============ Adaptive Step Size Settings ============
        
        /** @brief Whether to adapt step size based on acceptance rate */
        bool enable_adaptive_step_ = true;
        
        /** @brief Target acceptance rate for optimal mixing (0.234 for high-dimensional problems) */
        double target_acceptance_rate_ = 0.234;
        
        /** @brief Learning rate for Robbins-Monro step size adaptation */
        double adaptation_rate_ = 0.05;

        // ============ State Variables ============
        
        /** @brief Mersenne Twister random number generator */
        std::mt19937 gen_;
        
        /** @brief Uniform distribution for acceptance decisions */
        std::uniform_real_distribution<> unif_;
        
        /** @brief Thread-safe counter for accepted proposals (after burn-in) */
        std::atomic<int> accepted_count_{0};
        
        /** @brief Thread-safe counter for total objective function evaluations */
        std::atomic<int> total_evaluations_{0};

        // ============ Private Methods ============

        /**
         * @brief Generates multiple proposal vectors using vectorized operations
         * 
         * Creates a batch of proposals by adding Gaussian noise to the current parameters.
         * Uses parameter-specific standard deviations scaled by step_coef.
         * All proposals are constrained to valid parameter space.
         * 
         * @param current Current parameter vector
         * @param num_proposals Number of proposals to generate
         * @param step_coef Coefficient to scale the proposal standard deviations
         * @param paramManager Parameter manager for constraints and sigmas
         * @return Matrix where each column is a proposal vector
         */
        Eigen::MatrixXd generateVectorizedProposals(
            const Eigen::VectorXd& current,
            int num_proposals,
            double step_coef,
            IParameterManager& paramManager);

        /**
         * @brief Evaluates multiple proposals in parallel if enabled
         * 
         * Uses OpenMP to evaluate the objective function for multiple proposals
         * concurrently. Falls back to sequential evaluation if parallel execution
         * is disabled or the number of proposals is small.
         * 
         * @param proposals Matrix of proposal vectors (each column is a proposal)
         * @param objectiveFunction Function to evaluate
         * @return Vector of objective values corresponding to each proposal
         * 
         * @note Increments total_evaluations_ atomically for thread safety
         */
        std::vector<double> evaluateProposalsParallel(
            const Eigen::MatrixXd& proposals,
            IObjectiveFunction& objectiveFunction);

        /**
         * @brief Performs vectorized proposal refinement using elongation and binary search
         * 
         * Implements the refinement strategy from Cai et al. (2008):
         * 1. **Elongation phase**: Tests proposals at increasing distances along step direction
         * 2. **Binary search phase**: Refines the best point found during elongation
         * 
         * Uses vectorized evaluation for efficiency.
         * 
         * @param initial Initial proposal point
         * @param initial_logL Log-likelihood of initial point
         * @param step_direction Direction vector for refinement (initial - current)
         * @param num_refinement_steps Number of binary search iterations
         * @param objectiveFunction Function to optimize
         * @param parameterManager For applying parameter constraints
         * @return Pair of (refined parameters, refined log-likelihood)
         */
        std::pair<Eigen::VectorXd, double> performVectorizedRefinement(
            const Eigen::VectorXd& initial,
            double initial_logL,
            const Eigen::VectorXd& step_direction,
            int num_refinement_steps,
            IObjectiveFunction& objectiveFunction,
            IParameterManager& parameterManager);

        /**
         * @brief Updates the proposal step size based on recent acceptance rate
         * 
         * Uses Robbins-Monro stochastic approximation to adapt the step size
         * towards achieving the target acceptance rate. The adaptation follows:
         * step_size *= exp(adaptation_rate * log(current_rate / target_rate))
         * 
         * @param acceptance_rate Recent acceptance rate (should be between 0 and 1)
         * 
         * @note Step size is bounded between 1e-6 and 10.0 to prevent numerical issues
         */
        void updateStepSize(double acceptance_rate);

        /**
         * @brief Saves MCMC samples and objective values to a CSV file
         * 
         * Creates a CSV file with columns for sample ID, objective value, and all parameters.
         * The file is saved in the data/mcmc_samples/ directory with a timestamp.
         * 
         * @param samples Vector of parameter samples
         * @param objectiveValues Corresponding objective function values
         * @param parameterNames Names of parameters for column headers
         * @param filepath Full path to the output CSV file
         * 
         * @note Creates the output directory if it doesn't exist
         */
        void saveSamplesToCSV(
            const std::vector<Eigen::VectorXd>& samples,
            const std::vector<double>& objectiveValues,
            const std::vector<std::string>& parameterNames,
            const std::string& filepath);
    };

} // namespace epidemic

#endif // METROPOLIS_HASTINGS_SAMPLER_HPP