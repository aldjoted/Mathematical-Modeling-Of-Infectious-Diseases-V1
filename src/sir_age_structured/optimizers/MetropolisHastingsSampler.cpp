#include "sir_age_structured/optimizers/MetropolisHastingsSampler.hpp"
#include "utils/Logger.hpp"
#include "utils/FileUtils.hpp"

#include <Eigen/Cholesky>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace epidemic {

    static Logger& logger = Logger::getInstance();
    static const std::string LOG_SOURCE = "MCMC_SA";

    MetropolisHastingsSampler::MetropolisHastingsSampler() {
        std::random_device rd;
        gen_ = std::mt19937(rd());
    }

    void MetropolisHastingsSampler::configure(const std::map<std::string, double>& settings) {
        auto get = [&](const std::string& key, double def) {
            auto it = settings.find(key);
            return (it != settings.end()) ? it->second : def;
        };

        iterations_ = static_cast<int>(get("mcmc_iterations", 5000.0));
        report_interval_ = static_cast<int>(get("report_interval", 100.0));
        thinning_ = std::max(1, static_cast<int>(get("thinning", 1.0)));
        
        // Simulated Annealing Parameters
        initial_temp_ = get("sa_initial_temp", 100.0);
        cooling_rate_ = get("sa_cooling_rate", 0.999);
        step_all_prob_ = get("step_all_prob", 0.4);
        cloud_size_multiplier_ = std::max(1, static_cast<int>(get("cloud_size_multiplier", 2.0)));

        logger.info(LOG_SOURCE, "Configured SA-MCMC: Iter=" + std::to_string(iterations_) +
            ", T0=" + std::to_string(initial_temp_) +
            ", CoolRate=" + std::to_string(cooling_rate_) + 
            ", GlobalStepProb=" + std::to_string(step_all_prob_) +
            ", CloudMultiplier=" + std::to_string(cloud_size_multiplier_));
    }

    void MetropolisHastingsSampler::setInitialCovariance(const Eigen::MatrixXd& cov) {
        if (cov.rows() > 0 && cov.cols() > 0 && cov.rows() == cov.cols()) {
            initialCovariance_ = cov;
            hasInitialCovariance_ = true;
            logger.info(LOG_SOURCE, "Initial covariance set from Phase 1 (" + 
                        std::to_string(cov.rows()) + "x" + std::to_string(cov.cols()) + "). "
                        "MCMC will warm-start with learned correlation structure.");
        } else {
            hasInitialCovariance_ = false;
            logger.warning(LOG_SOURCE, "Invalid covariance matrix provided. Using default diagonal initialization.");
        }
    }

    double MetropolisHastingsSampler::safeEvaluate(IObjectiveFunction& func, const Eigen::VectorXd& p) {
        try {
            double val = func.calculate(p);
            // Treat NaN/Inf as extremely low likelihood (effectively -infinity)
            if (std::isnan(val) || std::isinf(val)) return -1e18;
            return val;
        } catch (...) {
            return -1e18;
        }
    }

    // --- 1. Random Step All (Global Correlated Move) ---
    // Corresponds to legacy 'random_step_all' but uses adaptive covariance 
    // to handle parameter correlations correctly.
    Eigen::VectorXd MetropolisHastingsSampler::randomStepAll(
        const Eigen::VectorXd& currentParams,
        const Eigen::MatrixXd& choleskyCov,
        double scale,
        IParameterManager& pm) {
        
        std::normal_distribution<> norm(0.0, 1.0);
        int n = static_cast<int>(currentParams.size());

        // Generate standard normal vector
        Eigen::VectorXd z(n);
        for(int i=0; i<n; ++i) z(i) = norm(gen_);

        // Apply correlated perturbation: x_new = x + scale * L * z
        Eigen::VectorXd perturbation = scale * (choleskyCov * z);
        
        return pm.applyConstraints(currentParams + perturbation);
    }

    // --- 2. Random Step One (Local Axis-Aligned Move) ---
    // Corresponds to legacy 'random_step_one'. Perturbs only one dimension.
    Eigen::VectorXd MetropolisHastingsSampler::randomStepOne(
        const Eigen::VectorXd& currentParams,
        IParameterManager& pm) {
        
        std::uniform_int_distribution<> dist_idx(0, static_cast<int>(currentParams.size()) - 1);
        std::normal_distribution<> norm(0.0, 1.0);

        int idx = dist_idx(gen_);
        double sigma = pm.getSigmaForParamIndex(idx);

        Eigen::VectorXd candidate = currentParams;
        // Apply perturbation to single index
        candidate(idx) += norm(gen_) * sigma;

        return pm.applyConstraints(candidate);
    }

    // --- 3. Thread-safe Candidate Generation for Parallel Cloud ---
    // Generates a candidate using a thread-local RNG for parallel execution
    Eigen::VectorXd MetropolisHastingsSampler::generateCandidate(
        const Eigen::VectorXd& currentParams,
        const Eigen::MatrixXd& choleskyCov,
        double scale,
        IParameterManager& pm,
        std::mt19937& localGen) {
        
        std::uniform_real_distribution<> uniform(0.0, 1.0);
        std::normal_distribution<> norm(0.0, 1.0);
        
        int n = static_cast<int>(currentParams.size());
        bool isGlobal = (uniform(localGen) < step_all_prob_);
        
        if (isGlobal) {
            // Global Move (Correlated)
            Eigen::VectorXd z(n);
            for(int i = 0; i < n; ++i) z(i) = norm(localGen);
            Eigen::VectorXd perturbation = scale * (choleskyCov * z);
            return pm.applyConstraints(currentParams + perturbation);
        } else {
            // Local Move (One Parameter)
            std::uniform_int_distribution<> dist_idx(0, n - 1);
            int idx = dist_idx(localGen);
            double sigma = pm.getSigmaForParamIndex(idx);
            
            Eigen::VectorXd candidate = currentParams;
            candidate(idx) += norm(localGen) * sigma;
            return pm.applyConstraints(candidate);
        }
    }

    OptimizationResult MetropolisHastingsSampler::optimize(
        const Eigen::VectorXd& initialParameters,
        IObjectiveFunction& objectiveFunction,
        IParameterManager& parameterManager) {
        
        OptimizationResult result;
        result.bestParameters = initialParameters;
        result.bestObjectiveValue = safeEvaluate(objectiveFunction, initialParameters);
        
        // Current state
        Eigen::VectorXd currentParams = initialParameters;
        double currentLikelihood = result.bestObjectiveValue;

        // --- Adaptive Metropolis Setup ---
        int n_params = static_cast<int>(initialParameters.size());
        
        // --- 1. Covariance Initialization ---
        // Use Phase 1 covariance if available (warm start), otherwise initialize from sigmas
        Eigen::MatrixXd cov;
        if (hasInitialCovariance_ && 
            initialCovariance_.rows() == n_params && 
            initialCovariance_.cols() == n_params) {
            logger.info(LOG_SOURCE, "Importing learned covariance from Phase 1 for warm start.");
            cov = initialCovariance_;
        } else {
            // Initial covariance (diagonal based on sigmas)
            cov = Eigen::MatrixXd::Identity(n_params, n_params);
            for(int i=0; i<n_params; ++i) {
                double s = parameterManager.getSigmaForParamIndex(i);
                cov(i,i) = (s > 0 ? s*s : 1e-6);
            }
        }

        // Cholesky decomposition of covariance
        Eigen::MatrixXd L = cov.llt().matrixL();
        
        // Track mean for AM update
        Eigen::VectorXd meanParams = initialParameters;

        // Simulated Annealing State
        double T = initial_temp_;
        std::uniform_real_distribution<> uniform(0.0, 1.0);
        
        // Stats
        int accepted = 0;
        int cloud_improvements = 0;

        // --- Parallel Candidate Cloud Setup ---
        #ifdef _OPENMP
        int num_threads = omp_get_max_threads();
        #else
        int num_threads = 1;
        #endif
        int cloud_size = num_threads * cloud_size_multiplier_;
        
        // --- 2. Robust RNG Setup (using seed_seq for proper statistical independence) ---
        std::vector<std::mt19937> thread_rngs(cloud_size);
        std::random_device rd;
        // Generate a sequence of seeds to ensure independence across threads
        std::vector<unsigned int> seed_data(cloud_size * 4);
        for(auto& s : seed_data) s = rd();
        std::seed_seq seq(seed_data.begin(), seed_data.end());
        
        // Generate actual seeds from seed_seq
        std::vector<unsigned int> actual_seeds(cloud_size);
        seq.generate(actual_seeds.begin(), actual_seeds.end());
        
        for (int i = 0; i < cloud_size; ++i) {
            thread_rngs[i].seed(actual_seeds[i]);
        }
        
        // Pre-compute optimal scale for global moves
        double global_scale = 2.38 / std::sqrt(n_params);
        
        logger.info(LOG_SOURCE, "Starting Optimization Loop. Initial Likelihood: " + std::to_string(currentLikelihood) +
                    " | CloudSize=" + std::to_string(cloud_size) + " (Threads=" + std::to_string(num_threads) + ")");

        for (int iter = 0; iter < iterations_; ++iter) {
            
            // 1. Generate Parallel Candidate Cloud
            std::vector<Eigen::VectorXd> candidateCloud(cloud_size);
            std::vector<double> candidateLikelihoods(cloud_size);
            
            #pragma omp parallel for schedule(dynamic)
            for (int c = 0; c < cloud_size; ++c) {
                // Generate candidate using thread-local RNG
                candidateCloud[c] = generateCandidate(
                    currentParams, L, global_scale, parameterManager, thread_rngs[c]);
                
                // Evaluate candidate
                candidateLikelihoods[c] = safeEvaluate(objectiveFunction, candidateCloud[c]);
            }

            // 2. Find Best Candidate from Cloud
            int bestIdx = 0;
            double bestCloudLikelihood = candidateLikelihoods[0];
            for (int c = 1; c < cloud_size; ++c) {
                if (candidateLikelihoods[c] > bestCloudLikelihood) {
                    bestCloudLikelihood = candidateLikelihoods[c];
                    bestIdx = c;
                }
            }
            
            Eigen::VectorXd candidateParams = candidateCloud[bestIdx];
            double candidateLikelihood = bestCloudLikelihood;
            
            // Track if cloud found improvement over all candidates
            if (candidateLikelihood > currentLikelihood) {
                cloud_improvements++;
            }

            // 3. Acceptance Criteria (Simulated Annealing on Best Candidate)
            // Delta L (Maximization problem)
            double deltaL = candidateLikelihood - currentLikelihood;
            
            bool accept = false;
            if (deltaL > 0) {
                // Always accept improvement
                accept = true;
            } else {
                // Accept degradation with probability exp(delta / T)
                // Note: deltaL is negative here
                double acceptanceProb = std::exp(deltaL / T);
                if (uniform(gen_) < acceptanceProb) {
                    accept = true;
                }
            }

            // 4. Update State
            if (accept) {
                currentParams = candidateParams;
                currentLikelihood = candidateLikelihood;
                accepted++;

                // Update Global Best?
                if (currentLikelihood > result.bestObjectiveValue) {
                    result.bestObjectiveValue = currentLikelihood;
                    result.bestParameters = currentParams;
                    // Logging immediate breakthrough can be helpful
                    if (iter % report_interval_ == 0) {
                         logger.info(LOG_SOURCE, "[NEW BEST] Iter " + std::to_string(iter) + 
                                     " | L=" + std::to_string(currentLikelihood));
                    }
                }
            }

            // 5. Sample Storage (Thinning)
            if (iter % thinning_ == 0) {
                result.samples.push_back(currentParams);
                result.sampleObjectiveValues.push_back(currentLikelihood);
            }

            // 6. Adaptive Covariance Update (Haario et al.)
            // Recursive update of covariance matrix to learn correlations
            // Use a floor on gamma to prevent covariance adaptation from freezing too early
            if (iter > 100) { 
                // Learning rate with floor: prevents freezing if chain hasn't converged
                double gamma = std::max(0.01, 10.0 / (iter + 100.0));
                Eigen::VectorXd diff = currentParams - meanParams;
                meanParams += gamma * diff;
                
                // Rank-1 update
                cov = (1.0 - gamma) * cov + gamma * (diff * diff.transpose());
                
                // Recompute Cholesky every 50 iterations to keep proposal in sync with learned covariance
                // (Previously 500 was too infrequent - only 10 updates in 5000 iterations)
                if (iter % 50 == 0) {
                    // Add small epsilon to diagonal for numerical stability
                    Eigen::MatrixXd cov_stable = cov + 1e-6 * Eigen::MatrixXd::Identity(n_params, n_params);
                    L = cov_stable.llt().matrixL();
                }
            }

            // 7. Cooling Schedule
            T *= cooling_rate_;

            // 8. Progress Report & Checkpointing
            if ((iter + 1) % report_interval_ == 0) {
                std::stringstream ss;
                ss << "Iter " << std::setw(5) << (iter + 1) 
                   << " | T=" << std::fixed << std::setprecision(4) << T
                   << " | CurL=" << std::setprecision(2) << currentLikelihood
                   << " | BestL=" << std::setprecision(2) << result.bestObjectiveValue
                   << " | AccRate=" << std::setprecision(1) << (100.0 * accepted / (iter + 1)) << "%"
                   << " | CloudImprv=" << std::setprecision(1) << (100.0 * cloud_improvements / (iter + 1)) << "%";
                logger.info(LOG_SOURCE, ss.str());

                // Save checkpoint
                saveCheckpoint(result, parameterManager, false);
            }
        }

        // Final Save
        saveCheckpoint(result, parameterManager, true);

        if (!result.samples.empty()) {
            std::string dir = FileUtils::joinPaths(FileUtils::getProjectRoot(), "data/mcmc_samples");
            FileUtils::ensureDirectoryExists(dir);
            std::string path = FileUtils::joinPaths(dir, "sa_optimization_trace.csv");
            
            saveSamplesToCSV(result.samples, result.sampleObjectiveValues, parameterManager.getParameterNames(), path);
        }

        return result;
    }

    void MetropolisHastingsSampler::saveSamplesToCSV(
        const std::vector<Eigen::VectorXd>& samples,
        const std::vector<double>& objectiveValues,
        const std::vector<std::string>& parameterNames,
        const std::string& filepath) {
        
        std::ofstream file(filepath);
        if (!file.is_open()) {
            logger.error(LOG_SOURCE, "Failed to open CSV for writing: " + filepath);
            return;
        }

        file << "iter,objective";
        for (const auto& name : parameterNames) file << "," << name;
        file << "\n";

        for (size_t i = 0; i < samples.size(); ++i) {
            file << i << "," << objectiveValues[i];
            for (int j = 0; j < samples[i].size(); ++j) {
                file << "," << std::scientific << std::setprecision(10) << samples[i][j];
            }
            file << "\n";
        }
        logger.info(LOG_SOURCE, "Saved full trace to: " + filepath);
    }

    void MetropolisHastingsSampler::saveCheckpoint(const OptimizationResult& res, 
                                                   IParameterManager& pm, 
                                                   bool final) {
        if (res.samples.empty()) return;

        std::string dir = FileUtils::joinPaths(FileUtils::getProjectRoot(), "data/checkpoints");
        FileUtils::ensureDirectoryExists(dir);
        std::string fname = final ? "opt_final.csv" : "opt_checkpoint.csv";
        std::string path = FileUtils::joinPaths(dir, fname);
        
        std::ofstream file(path);
        if(file.is_open()) {
            file << "step,logL";
            for(const auto& n : pm.getParameterNames()) file << "," << n;
            file << "\n";
            
            // Save tail of trace to keep checkpoint files manageable, or all if final
            // Typically saving last 1000 samples is enough for inspection unless final
            size_t start = final ? 0 : (res.samples.size() > 1000 ? res.samples.size() - 1000 : 0);
            
            for(size_t i = start; i < res.samples.size(); ++i) {
                file << i << "," << res.sampleObjectiveValues[i];
                for(int j=0; j<res.samples[i].size(); ++j) file << "," << res.samples[i][j];
                file << "\n";
            }
        }
    }

} // namespace epidemic