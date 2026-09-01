#include "sir_age_structured/optimizers/HillClimbingOptimizer.hpp"
#include "utils/Logger.hpp"
#include <Eigen/Cholesky>
#include <Eigen/StdVector>  // Required for aligned_allocator with Eigen types
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace epidemic {

    static Logger& logger = Logger::getInstance();
    static const std::string LOG_SOURCE = "HILL_CLIMBING";

    // --- Helper: Safe Evaluation ---
    // Wraps the objective function to handle NaN/Inf gracefully.
    // NOTE: Thread safety is ensured by the implementations:
    //   - IObjectiveFunction::calculate() clones internal model state
    //   - IParameterManager::applyConstraints() is const and read-only
    static double safe_evaluate(IObjectiveFunction& func, const Eigen::VectorXd& p) {
        try {
            double val = func.calculate(p);
            if (std::isnan(val) || std::isinf(val)) return -1e18; 
            return val;
        } catch (...) { 
            return -1e18; 
        }
    }

    // --- Helper: Robust Adaptive Line Search ---
    // Phase 1: Backtracking (Safety) - Find any valid improvement with progressively smaller steps
    // Phase 2: Expansion (Speed) - Accelerate along the successful ridge using moving anchor
    static bool performRobustLineSearch(
        Eigen::VectorXd& current_params, 
        double& current_logL,
        const Eigen::VectorXd& direction, 
        IObjectiveFunction& func, 
        IParameterManager& pm) 
    {
        // Tuning constants
        const double shrinkage = 0.5;
        const double growth = 2.0;        // Patch E: More aggressive expansion
        const int max_backtrack = 10;
        const int max_expansion = 12;     // Patch E: More expansion steps
        
        // --- Phase 1: Find ANY Valid Improvement (Backtracking) ---
        double step = 1.0;
        Eigen::VectorXd improved_params = current_params;
        double improved_logL = current_logL;
        bool found_improvement = false;
        
        // Try progressively smaller steps to find a foothold
        for (int i = 0; i < max_backtrack; ++i) {
            Eigen::VectorXd candidate = pm.applyConstraints(current_params + direction * step);
            
            // Early exit: If step is too small to make a difference, stop.
            if ((candidate - current_params).squaredNorm() < 1e-16) break;

            double candidate_logL = safe_evaluate(func, candidate);
            
            if (candidate_logL > improved_logL) {
                improved_params = candidate;
                improved_logL = candidate_logL;
                found_improvement = true;
                break;  // Found a foothold, move to expansion
            }
            step *= shrinkage;
        }
        
        if (!found_improvement) return false;
        
        // --- Phase 2: Incremental Expansion (Crawler Strategy) ---
        // We calculate the effective vector that worked (handling constraints)
        // and try to accelerate along that vector relative to the NEW point.
        
        Eigen::VectorXd best_params = improved_params;
        double best_logL = improved_logL;
        
        // This captures the "slide" along boundaries if constraints were hit
        Eigen::VectorXd current_step = (improved_params - current_params);
        
        for (int i = 0; i < max_expansion; ++i) {
            // Grow the step size
            current_step *= growth;
            
            // Move relative to the *latest* best point (Moving Anchor)
            Eigen::VectorXd candidate = pm.applyConstraints(best_params + current_step);
            double candidate_logL = safe_evaluate(func, candidate);
            
            if (candidate_logL > best_logL) {
                best_params = candidate;
                best_logL = candidate_logL;
                // Note: We update best_params, so the next iteration 
                // steps off from this new, further point.
            } else {
                break;  // Overshot the peak/ridge
            }
        }
        
        // Update state
        current_params = best_params;
        current_logL = best_logL;
        return true;
    }

    // --- Class Implementation ---

    HillClimbingOptimizer::HillClimbingOptimizer() {
        gen_ = std::mt19937(std::random_device{}());
    }

    void HillClimbingOptimizer::configure(const std::map<std::string, double>& settings) {
        auto get = [&](const std::string& key, double def) {
            auto it = settings.find(key);
            return (it != settings.end()) ? it->second : def;
        };

        iterations_ = static_cast<int>(get("iterations", 2000.0));
        report_interval_ = static_cast<int>(get("report_interval", 100.0));
        cloud_size_multiplier_ = std::max(1, static_cast<int>(get("cloud_size_multiplier", 8.0)));

        logger.info(LOG_SOURCE, "Configured Parallel Hill Climber: Iterations=" + std::to_string(iterations_) +
                    ", CloudMultiplier=" + std::to_string(cloud_size_multiplier_));
    }

    OptimizationResult HillClimbingOptimizer::optimize(
        const Eigen::VectorXd& initialParameters,
        IObjectiveFunction& objectiveFunction,
        IParameterManager& parameterManager) {

        OptimizationResult result;
        result.bestParameters = initialParameters;
        result.bestObjectiveValue = safe_evaluate(objectiveFunction, initialParameters);
        
        Eigen::VectorXd current_params = initialParameters;
        double current_logL = result.bestObjectiveValue;
        int n_params = static_cast<int>(initialParameters.size());

        // --- 1. Adaptive Covariance Initialization ---
        // Initialize diagonal covariance from parameter sigmas (heuristics)
        Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(n_params, n_params);
        for(int i=0; i<n_params; ++i) {
             double s = parameterManager.getSigmaForParamIndex(i);
             cov(i,i) = (s > 0 ? s*s : 1e-4);
        }

        // Pre-compute Cholesky decomposition for correlated sampling
        Eigen::MatrixXd L = cov.llt().matrixL();

        // --- 2. Dynamic Parallelism Setup ---
        int available_threads = 1;
        #ifdef _OPENMP
        available_threads = omp_get_max_threads();
        if (available_threads < 1) available_threads = 1;
        #endif
        
        // Candidates per step: Scale with threads (e.g., 4 threads * 8 multiplier = 32 candidates)
        const int num_candidates = std::max(4, available_threads * cloud_size_multiplier_);

        logger.info(LOG_SOURCE, "Starting Parallel Search. Threads: " + std::to_string(available_threads) + 
                                ", Cloud Size: " + std::to_string(num_candidates) +
                                ", Initial LogL: " + std::to_string(result.bestObjectiveValue));

        // --- 3. Main Optimization Loop ---
        // NOTE: We do not save samples vector as requested for Phase 1.
        
        // History for covariance adaptation (store last 2 successful points to define vector)
        Eigen::VectorXd prev_params = current_params;

        // Random number generation setup - create properly seeded thread-local RNGs
        // Use a master RNG to generate independent seeds for each thread
        std::mt19937 master_gen(std::random_device{}());
        std::vector<std::mt19937> thread_rngs(available_threads);
        std::vector<std::normal_distribution<double>> thread_norms(available_threads);
        for (int t = 0; t < available_threads; ++t) {
            // Use seed_seq for proper initialization (better statistical independence)
            std::seed_seq seed{master_gen(), master_gen(), master_gen(), master_gen()};
            thread_rngs[t].seed(seed);
            thread_norms[t] = std::normal_distribution<double>(0.0, 1.0);
        }

        // Reuse buffers across iterations to avoid repeated allocations.
        std::vector<Eigen::VectorXd> candidates(num_candidates, Eigen::VectorXd(n_params));
        std::vector<Eigen::VectorXd> constrained_candidates(num_candidates, Eigen::VectorXd(n_params));
        std::vector<double> scores(num_candidates, -1e18);

        for (int iter = 0; iter < iterations_; ++iter) {
            
            // A. Generate Candidate Cloud (Parallel)
            // Mix of Global Correlated moves (using L) and Local Axis-Aligned moves
            // (candidates/scores reused)

            #pragma omp parallel for num_threads(available_threads) schedule(static)
            for(int i=0; i<num_candidates; ++i) {
                #ifdef _OPENMP
                int tid = omp_get_thread_num();
                #else
                int tid = 0;
                #endif
                std::mt19937& local_rng = thread_rngs[tid];
                std::normal_distribution<double>& norm = thread_norms[tid];
                
                if (i < num_candidates / 2) {
                    // Strategy 1: Correlated Global Move
                    Eigen::VectorXd z(n_params);
                    for(int k=0; k<n_params; ++k) z(k) = norm(local_rng);
                    candidates[i] = L * z; 
                } else {
                    // Strategy 2: Axis-Aligned Local Move
                    std::uniform_int_distribution<int> param_dist(0, n_params - 1);
                    int idx = param_dist(local_rng);
                    double sigma = std::sqrt(cov(idx, idx)); 
                    candidates[i] = Eigen::VectorXd::Zero(n_params);
                    candidates[i](idx) = sigma * norm(local_rng); 
                }
            }

            // B. Parallel Evaluation
            // Distribute the costly objective function calls across cores
            // Patch A: Store constrained points to get the TRUE direction after projection
            // (constrained_candidates reused)
            
            #pragma omp parallel for num_threads(available_threads) schedule(dynamic)
            for(int i=0; i<num_candidates; ++i) {
                // Candidate vector represents the *step*, so add to current
                Eigen::VectorXd p_test = parameterManager.applyConstraints(current_params + candidates[i]);
                constrained_candidates[i] = p_test;  // Patch A: Store constrained point
                scores[i] = safe_evaluate(objectiveFunction, p_test);
            }

            // C. Winner Selection
            int best_idx = -1;
            double best_val = -1e18;
            for(int i=0; i<num_candidates; ++i) {
                if (scores[i] > best_val) {
                    best_val = scores[i];
                    best_idx = i;
                }
            }

            // D. Early Accept + Robust Line Search
            bool moved = false;
            if (best_idx != -1 && best_val > -1e18) {
                // Patch A: Compute the TRUE constrained direction (handles boundary projection)
                Eigen::VectorXd best_constrained_point = constrained_candidates[best_idx];
                Eigen::VectorXd constrained_direction = best_constrained_point - current_params;
                
                // Patch B: Early accept - if cloud found improvement, take it NOW
                // This ensures we never lose a good point even if line search fails
                if (best_val > current_logL) {
                    current_params = best_constrained_point;
                    current_logL = best_val;
                    moved = true;
                }
                
                // Try to exploit the constrained direction further via line search
                // Patch A: Pass constrained_direction instead of raw candidates[best_idx]
                bool line_search_improved = performRobustLineSearch(
                    current_params, current_logL, constrained_direction, objectiveFunction, parameterManager
                );
                moved = moved || line_search_improved;
            }

            // E. Update & Adaptation
            if (moved) {
                if (current_logL > result.bestObjectiveValue) {
                    result.bestObjectiveValue = current_logL;
                    result.bestParameters = current_params;
                }

                // Covariance Update (Haario Adaptation)
                // Use the vector of the actual move taken (current - prev)
                Eigen::VectorXd actual_step = current_params - prev_params;
                double step_norm = actual_step.squaredNorm();
                
                if (step_norm > 1e-14) {
                    // Patch D: Adaptive alpha (CMA-ES style) - scales with dimension
                    double alpha = 2.0 / (n_params + 2.0);
                    cov *= (1.0 - alpha);
                    cov += alpha * (actual_step * actual_step.transpose());
                    
                    // Patch C: Force symmetry (floating point noise can break it)
                    cov = 0.5 * (cov + cov.transpose());
                    
                    // Patch C: Systematic jitter for numerical stability
                    double jitter = 1e-8 * cov.trace() / n_params;
                    cov += jitter * Eigen::MatrixXd::Identity(n_params, n_params);
                    
                    // Enforce minimum diagonal floor to prevent covariance collapse
                    // This ensures a minimum exploration capability even when stuck
                    for (int i = 0; i < n_params; ++i) {
                        double min_var = parameterManager.getSigmaForParamIndex(i);
                        min_var = (min_var > 0 ? min_var * min_var * 0.01 : 1e-8);  // 1% of original variance
                        if (cov(i, i) < min_var) cov(i, i) = min_var;
                    }
                }
                prev_params = current_params;
            }

            // F. Refresh Cholesky more frequently (every 10 iterations)
            // Since N (parameters) is small, O(N³) Cholesky is negligible vs simulation cost
            // This ensures L reflects recent covariance changes for efficient sampling
            if (iter > 0 && iter % 10 == 0) {
                // Ensure covariance is positive definite with adaptive regularization
                Eigen::LLT<Eigen::MatrixXd> llt(cov);
                if (llt.info() == Eigen::Success) {
                    L = llt.matrixL();
                } else {
                    // Regularization: add jitter proportional to trace
                    double lambda = 1e-6 * cov.trace() / n_params;
                    int max_attempts = 5;
                    bool regularized = false;
                    for (int attempt = 0; attempt < max_attempts; ++attempt) {
                        cov += lambda * Eigen::MatrixXd::Identity(n_params, n_params);
                        Eigen::LLT<Eigen::MatrixXd> llt_retry(cov);
                        if (llt_retry.info() == Eigen::Success) {
                            L = llt_retry.matrixL();
                            regularized = true;
                            break;
                        }
                        lambda *= 10.0;  // Increase jitter if still failing
                    }
                    // If all attempts fail, reset covariance to diagonal
                    if (!regularized) {
                        L = cov.diagonal().cwiseSqrt().asDiagonal();
                        // Reset covariance off-diagonals to match L
                        cov = cov.diagonal().asDiagonal();
                        logger.warning(LOG_SOURCE, "Covariance reset to diagonal due to instability");
                    }
                }
            }

            // G. Logging
            if ((iter + 1) % report_interval_ == 0) {
                 logger.info(LOG_SOURCE, "Iter " + std::to_string(iter+1) + 
                             " | Best LogL: " + std::to_string(result.bestObjectiveValue) +
                             " | Current LogL: " + std::to_string(current_logL));
            }
        }

        // Store the learned covariance for Phase 2 transfer
        result.finalCovariance = cov;
        logger.info(LOG_SOURCE, "Phase 1 complete. Learned covariance matrix stored for Phase 2 transfer.");

        // Return only the best result (no samples history)
        return result;
    }

} // namespace epidemic