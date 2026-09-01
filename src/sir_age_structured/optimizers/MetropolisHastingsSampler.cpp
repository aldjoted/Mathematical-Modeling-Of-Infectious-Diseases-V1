#include "sir_age_structured/optimizers/MetropolisHastingsSampler.hpp"
#include "model/parameters/SEPAIHRDParameterManager.hpp"
#include "utils/Logger.hpp"
#include "utils/FileUtils.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <numeric>
#include <algorithm>

namespace epidemic {

    static Logger& logger = Logger::getInstance();
    static const std::string LOG_SOURCE = "MCMC_AM"; // Adaptive Metropolis

    MetropolisHastingsSampler::MetropolisHastingsSampler() {
        std::random_device rd;
        gen_ = std::mt19937(rd());
    }

    void MetropolisHastingsSampler::configure(const std::map<std::string, double>& settings) {
        auto get = [&](const std::string& key, double def) {
            auto it = settings.find(key);
            return (it != settings.end()) ? it->second : def;
        };

        iterations_ = static_cast<int>(get("mcmc_iterations", 10000.0));
        burn_in_ = static_cast<int>(get("burn_in", 1000.0));
        adaptation_period_ = static_cast<int>(get("adaptation_period", 100.0));
        report_interval_ = static_cast<int>(get("report_interval", 100.0));
        thinning_ = std::max(1, static_cast<int>(get("thinning", 1.0)));
        regularization_epsilon_ = get("regularization_epsilon", 1e-6);
        target_acceptance_rate_ = get("target_acceptance_rate", 0.234);
        adapt_scale_ = (get("adapt_scale", 1.0) != 0.0);

        // Optional I/O controls
        store_samples_ = (get("store_samples", 1.0) != 0.0);
        write_checkpoints_ = (get("write_checkpoints", 1.0) != 0.0);
        write_trace_ = (get("write_trace", 1.0) != 0.0);

        logger.info(LOG_SOURCE, "Configured Adaptive Metropolis: Iterations=" + std::to_string(iterations_) + 
                                ", Burn-In=" + std::to_string(burn_in_) +
                                ", AdaptPeriod=" + std::to_string(adaptation_period_) +
                                ", TargetAccRate=" + std::to_string(target_acceptance_rate_) +
                                ", AdaptScale=" + std::to_string(adapt_scale_));
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
            // Treat NaN/Inf as effectively -infinity (zero probability)
            if (std::isnan(val) || std::isinf(val)) return -1e18;
            return val;
        } catch (...) {
            return -1e18;
        }
    }

    bool MetropolisHastingsSampler::areParametersValid(
        const Eigen::VectorXd& params, 
        IParameterManager& pm) const 
    {
        // Check if all parameters are within their defined bounds
        for (int i = 0; i < params.size(); ++i) {
            double lower = pm.getLowerBoundForParamIndex(i);
            double upper = pm.getUpperBoundForParamIndex(i);
            if (params(i) < lower || params(i) > upper) {
                return false;
            }
        }
        return true;
    }

    Eigen::VectorXd MetropolisHastingsSampler::generateProposal(const Eigen::VectorXd& currentParams) {
        // Generate Z ~ N(0, I)
        Eigen::VectorXd z(currentParams.size());
        std::normal_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < currentParams.size(); ++i) {
            z(i) = dist(gen_);
        }

        // Y = X + scale * L * Z
        // If proposal_cholesky_ encodes covariance Sigma = L*L^T, then scale*L*Z ~ N(0, scale^2*Sigma)
        return currentParams + global_scale_ * (proposal_cholesky_ * z);
    }

    void MetropolisHastingsSampler::adaptGlobalScale(bool accepted, int step) {
        // Robbins-Monro stochastic approximation: log(scale) += gamma * (alpha - target)
        if (!adapt_scale_) return;
        
        recent_accepts_.push_back(accepted ? 1 : 0);
        if (recent_accepts_.size() > 1000) {
            recent_accepts_.pop_front();
        }
        
        double recent_acc_rate = 0.0;
        if (!recent_accepts_.empty()) {
            int sum = 0;
            for (int a : recent_accepts_) sum += a;
            recent_acc_rate = static_cast<double>(sum) / recent_accepts_.size();
        }
        
        // Emergency heuristics for critically low acceptance
        if (recent_accepts_.size() >= 1000 && recent_acc_rate < 0.001) {
            log_scale_ -= 0.7;  // Halve the scale
            emergency_shrink_count_++;
            if (emergency_shrink_count_ % 10 == 1) {
                Logger::getInstance().warning(LOG_SOURCE, 
                    "Emergency scale shrink #" + std::to_string(emergency_shrink_count_) + 
                    " at step " + std::to_string(step) + ". New scale: " + 
                    std::to_string(std::exp(log_scale_)));
            }
        }
        else if (recent_acc_rate < 0.02 && recent_accepts_.size() >= 500) {
            // Aggressive shrinking for very low acceptance
            double gamma_fast = 5.0 / std::sqrt(static_cast<double>(step) + 1.0);
            gamma_fast = std::min(gamma_fast, 0.3);
            log_scale_ += gamma_fast * (0.0 - target_acceptance_rate_);
        }
        else {
            // Standard Robbins-Monro update
            double gamma = 1.0 / std::sqrt(static_cast<double>(step) + 1.0);
            gamma = std::min(gamma, 0.1);
            double accept_indicator = accepted ? 1.0 : 0.0;
            log_scale_ += gamma * (accept_indicator - target_acceptance_rate_);
        }
        
        // Allow scale recovery if stuck at floor but acceptance improving
        if (global_scale_ <= 0.011 && recent_acc_rate > 0.15 && recent_acc_rate < 0.30) {
            log_scale_ += 0.01;
        }
        
        log_scale_ = std::max(std::min(log_scale_, 2.3), -6.9);
        global_scale_ = std::exp(log_scale_);
    }

    void MetropolisHastingsSampler::updateCovarianceRank1(int step, int /* n_params */) {
        // Welford-style rank-1 update with adaptation rate gamma = 10.0 / (t + 100)
        if (chain_history_.empty()) return;
        
        const Eigen::VectorXd& new_sample = chain_history_.back();
        
        double gamma = 10.0 / (step + 100.0);
        
        Eigen::VectorXd diff = new_sample - running_mean_;
        running_mean_ += gamma * diff;
        
        current_covariance_ = (1.0 - gamma) * current_covariance_ + gamma * (diff * diff.transpose());
    }

    void MetropolisHastingsSampler::recomputeFullCovariance(int n_params) {
        // Full covariance recomputation to correct accumulated numerical errors
        if (chain_history_.size() < static_cast<size_t>(n_params) + 10) return;

        Eigen::VectorXd mean = Eigen::VectorXd::Zero(n_params);
        for (const auto& vec : chain_history_) {
            mean += vec;
        }
        mean /= static_cast<double>(chain_history_.size());
        running_mean_ = mean;

        Eigen::MatrixXd centered(chain_history_.size(), n_params);
        for (size_t i = 0; i < chain_history_.size(); ++i) {
            centered.row(i) = chain_history_[i] - mean;
        }

        Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(chain_history_.size() - 1);

        // Optimal scaling: (2.38^2) / d (Roberts & Rosenthal 2001)
        double scaling_factor = (2.38 * 2.38) / static_cast<double>(n_params);

        current_covariance_ = (scaling_factor * cov) + 
                              (regularization_epsilon_ * Eigen::MatrixXd::Identity(n_params, n_params));

        Eigen::LLT<Eigen::MatrixXd> llt(current_covariance_);
        if (llt.info() == Eigen::Success) {
            proposal_cholesky_ = llt.matrixL();
            logger.debug(LOG_SOURCE, "Full covariance recomputation successful.");
        } else {
            logger.warning(LOG_SOURCE, "Full covariance recomputation failed (singular). Keeping previous kernel.");
        }
    }

    OptimizationResult MetropolisHastingsSampler::optimize(
        const Eigen::VectorXd& initialParameters,
        IObjectiveFunction& objectiveFunction,
        IParameterManager& parameterManager) 
    {
        // Enable MCMC Reflection Mode for valid Bayesian sampling
        if (auto* casted_pm = dynamic_cast<SEPAIHRDParameterManager*>(&parameterManager)) {
            casted_pm->setConstraintMode(ConstraintMode::MCMC_REFLECT);
            logger.info(LOG_SOURCE, "Constraint mode set to MCMC_REFLECT for valid Bayesian sampling.");
        }

        OptimizationResult result;
        int n_params = static_cast<int>(initialParameters.size());

        // --- Initialization ---
        Eigen::VectorXd current_x = initialParameters;
        
        // Initialize covariance matrix
        if (hasInitialCovariance_ && 
            initialCovariance_.rows() == n_params && 
            initialCovariance_.cols() == n_params) {
            logger.info(LOG_SOURCE, "Warm-starting with covariance from Phase 1.");
            current_covariance_ = initialCovariance_;
        } else {
            // Initialize with diagonal covariance from proposal sigmas
            current_covariance_ = Eigen::MatrixXd::Identity(n_params, n_params);
            for (int i = 0; i < n_params; ++i) {
                double s = parameterManager.getSigmaForParamIndex(i);
                current_covariance_(i, i) = (s > 0 ? s * s : 1e-6);
            }
            // Apply initial scaling factor
            double scaling_factor = (2.38 * 2.38) / static_cast<double>(n_params);
            current_covariance_ *= scaling_factor;
        }
        
        // Add regularization
        current_covariance_ += regularization_epsilon_ * Eigen::MatrixXd::Identity(n_params, n_params);
        
        // Compute initial Cholesky decomposition
        Eigen::LLT<Eigen::MatrixXd> llt(current_covariance_);
        if (llt.info() != Eigen::Success) {
            logger.warning(LOG_SOURCE, "Initial covariance Cholesky failed. Using identity.");
            proposal_cholesky_ = Eigen::MatrixXd::Identity(n_params, n_params) * 0.1;
        } else {
            proposal_cholesky_ = llt.matrixL();
        }

        // Initialize running mean
        running_mean_ = initialParameters;
        
        // Initialize global scale for adaptive scaling
        log_scale_ = 0.0;
        global_scale_ = 1.0;

        // Evaluate initial state
        // IMPORTANT: objectiveFunction.calculate() returns Log-Posterior (or Log-Likelihood)
        double current_log_post = safeEvaluate(objectiveFunction, current_x);
        
        // Storage setup
        chain_history_.clear();
        chain_history_.reserve(iterations_);
        chain_history_.push_back(current_x);

        // Result container
        if (store_samples_) {
            result.samples.reserve(iterations_ / thinning_);
            result.sampleObjectiveValues.reserve(iterations_ / thinning_);
            result.samples.push_back(current_x);
            result.sampleObjectiveValues.push_back(current_log_post);
        }

        // Track best (for MAP estimate)
        result.bestParameters = current_x;
        result.bestObjectiveValue = current_log_post;

        int accepted_count = 0;
        std::uniform_real_distribution<double> u_dist(0.0, 1.0);

        logger.info(LOG_SOURCE, "Starting Bayesian AM-MCMC. Params: " + std::to_string(n_params) +
                    " | Initial LogPost: " + std::to_string(current_log_post));

        // --- MCMC Loop ---
        for (int t = 1; t < iterations_; ++t) {
            
            // 1. Adaptation Step (only after burn-in and periodically)
            if (t > burn_in_) {
                // Rank-1 update every iteration after burn-in
                updateCovarianceRank1(t, n_params);
                
                // Full recomputation every adaptation_period_ iterations
                if (t % adaptation_period_ == 0) {
                    recomputeFullCovariance(n_params);
                    
                    // Recompute Cholesky
                    Eigen::MatrixXd cov_stable = current_covariance_ + 
                                                 regularization_epsilon_ * Eigen::MatrixXd::Identity(n_params, n_params);
                    Eigen::LLT<Eigen::MatrixXd> llt_update(cov_stable);
                    if (llt_update.info() == Eigen::Success) {
                        proposal_cholesky_ = llt_update.matrixL();
                    }
                }
            }

            // 2. Propose new state: X' ~ N(X, scale^2 * Sigma)
            Eigen::VectorXd proposed_x_raw = generateProposal(current_x);
            
            // 2b. Apply constraints (reflection for MCMC mode) to keep proposals in bounds
            // This is CRITICAL in high dimensions - raw proposals almost always violate bounds
            Eigen::VectorXd proposed_x = parameterManager.applyConstraints(proposed_x_raw);

            // 3. Evaluate likelihood (constraints already applied, so proposal is valid)
            double proposed_log_post = safeEvaluate(objectiveFunction, proposed_x);

            // 4. Metropolis-Hastings Acceptance Ratio (Log Space)
            // log_alpha = log_pi(x') - log_pi(x) + log_q(x|x') - log_q(x'|x)
            // For symmetric proposal N(x, Sigma): q(x|x') = q(x'|x), so:
            // log_alpha = log_pi(x') - log_pi(x)
            double log_ratio = proposed_log_post - current_log_post;

            // 5. Accept / Reject
            bool accept = false;
            if (log_ratio >= 0.0) {
                // Always accept if proposal is better
                accept = true;
            } else {
                // Accept with probability exp(log_ratio)
                if (std::log(u_dist(gen_)) < log_ratio) {
                    accept = true;
                }
            }

            // 6. Update state
            if (accept) {
                current_x = proposed_x;
                current_log_post = proposed_log_post;
                accepted_count++;

                // Update MAP estimate
                if (current_log_post > result.bestObjectiveValue) {
                    result.bestObjectiveValue = current_log_post;
                    result.bestParameters = current_x;
                }
            }
            // Else: we keep current_x (repeat current state in chain)

            // 6b. Adapt global scale based on acceptance
            // Scale adaptation runs from the start to avoid wasting burn-in with 0% acceptance
            // The adaptation rate naturally diminishes with t via the 1/(t+1) factor
            if (adapt_scale_) {
                adaptGlobalScale(accept, t);
            }

            // 7. Store sample in chain history (always, for covariance estimation)
            chain_history_.push_back(current_x);
            
            // Store in result (with thinning)
            if (store_samples_ && (t % thinning_ == 0)) {
                result.samples.push_back(current_x);
                result.sampleObjectiveValues.push_back(current_log_post);
            }

            // 8. Logging & Checkpointing
            if ((t + 1) % report_interval_ == 0) {
                double acc_rate = static_cast<double>(accepted_count) / (t + 1);
                std::stringstream ss;
                ss << "Iter: " << std::setw(6) << (t + 1) 
                   << " | LogPost: " << std::fixed << std::setprecision(2) << current_log_post 
                   << " | Best: " << std::setprecision(2) << result.bestObjectiveValue
                   << " | AccRate: " << std::setprecision(1) << (acc_rate * 100.0) << "%"
                   << " | Scale: " << std::setprecision(3) << global_scale_;
                logger.info(LOG_SOURCE, ss.str());
                
                // Warn if acceptance rate is critically low (optimal is ~23.4% for RWM)
                if (t > burn_in_ && acc_rate < 0.05) {
                    logger.warning(LOG_SOURCE, "Acceptance rate low (<5%). Scale adaptation working to correct.");
                } else if (t > burn_in_ && acc_rate > 0.50) {
                    logger.warning(LOG_SOURCE, "Acceptance rate high (>50%). Scale adaptation working to correct.");
                }
                
                if (write_checkpoints_ && store_samples_) {
                    saveCheckpoint(result, parameterManager, false);
                }
            }
        }

        // --- Finalization ---
        double final_acc_rate = static_cast<double>(accepted_count) / iterations_;
        logger.info(LOG_SOURCE, "Sampling Complete. Final Acceptance Rate: " + 
                    std::to_string(final_acc_rate * 100.0) + "% | Final Scale: " +
                    std::to_string(global_scale_));
        
        // Store final covariance for potential use in future runs
        result.finalCovariance = current_covariance_;
        result.additionalStats["acceptance_rate"] = final_acc_rate;
        result.additionalStats["final_scale"] = global_scale_;
        result.additionalStats["burn_in"] = static_cast<double>(burn_in_);
        result.additionalStats["total_iterations"] = static_cast<double>(iterations_);

        if (write_checkpoints_ && store_samples_) {
            saveCheckpoint(result, parameterManager, true);
        }

        if (write_trace_ && store_samples_ && !result.samples.empty()) {
            std::string dir = FileUtils::joinPaths(FileUtils::getProjectRoot(), "data/mcmc_samples");
            FileUtils::ensureDirectoryExists(dir);
            std::string path = FileUtils::joinPaths(dir, "posterior_trace.csv");
            saveSamplesToCSV(result.samples, result.sampleObjectiveValues, 
                             parameterManager.getParameterNames(), path);
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

        file << "iter,log_posterior";
        for (const auto& name : parameterNames) file << "," << name;
        file << "\n";

        for (size_t i = 0; i < samples.size(); ++i) {
            file << i << "," << std::scientific << std::setprecision(6) << objectiveValues[i];
            for (int j = 0; j < samples[i].size(); ++j) {
                file << "," << samples[i][j];
            }
            file << "\n";
        }
        logger.info(LOG_SOURCE, "Full posterior trace saved to: " + filepath);
    }

    void MetropolisHastingsSampler::saveCheckpoint(const OptimizationResult& res, 
                                                   IParameterManager& pm, 
                                                   bool final) {
        if (res.samples.empty()) return;

        std::string dir = FileUtils::joinPaths(FileUtils::getProjectRoot(), "data/mcmc_samples");
        FileUtils::ensureDirectoryExists(dir);
        std::string fname = final ? "posterior_trace_final.csv" : "posterior_trace_checkpoint.csv";
        std::string path = FileUtils::joinPaths(dir, fname);
        
        std::ofstream file(path);
        if (file.is_open()) {
            file << "iter,log_posterior";
            auto names = pm.getParameterNames();
            for (const auto& n : names) file << "," << n;
            file << "\n";
            
            // For checkpoint: save last 5000 samples; for final: save everything
            size_t start_idx = final ? 0 : (res.samples.size() > 5000 ? res.samples.size() - 5000 : 0);

            for (size_t i = start_idx; i < res.samples.size(); ++i) {
                file << i << "," << std::scientific << std::setprecision(6) << res.sampleObjectiveValues[i];
                for (int j = 0; j < res.samples[i].size(); ++j) {
                    file << "," << res.samples[i][j];
                }
                file << "\n";
            }
        }
        if (final) logger.info(LOG_SOURCE, "Full posterior trace saved to: " + path);
    }

} // namespace epidemic