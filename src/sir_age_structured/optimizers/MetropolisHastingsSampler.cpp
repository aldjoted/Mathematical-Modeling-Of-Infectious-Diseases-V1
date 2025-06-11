#include "sir_age_structured/optimizers/MetropolisHastingsSampler.hpp"
#include "utils/Logger.hpp"
#include "utils/FileUtils.hpp"
#include <numeric>
#include <cmath>
#include <limits>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <immintrin.h>

namespace epidemic {
    static Logger& logger = Logger::getInstance();
    static const std::string LOG_SOURCE = "MHS"; 

    MetropolisHastingsSampler::MetropolisHastingsSampler()
        : gen_(std::random_device{}()), unif_(0.0, 1.0) {}
    
    void MetropolisHastingsSampler::configure(const std::map<std::string, double>& settings) {
        auto get_setting = [&](const std::string& name, double default_val) {
            auto it = settings.find(name);
            return (it != settings.end()) ? it->second : default_val;
        };
        
        burn_in_ = static_cast<int>(get_setting("burn_in", static_cast<double>(burn_in_)));
        mcmc_iterations_ = static_cast<int>(get_setting("mcmc_iterations", static_cast<double>(mcmc_iterations_)));
        thinning_ = static_cast<int>(get_setting("thinning", static_cast<double>(thinning_)));
        step_size_ = get_setting("mcmc_step_size", step_size_);
        calculate_posterior_mean_ = static_cast<bool>(get_setting("calculate_posterior_mean", 1.0));
        report_interval_ = static_cast<int>(get_setting("report_interval", static_cast<double>(report_interval_)));
        enable_refinement_ = static_cast<bool>(get_setting("mcmc_enable_refinement", 0.0));
        refinement_steps_ = static_cast<int>(get_setting("mcmc_refinement_steps", 0.0));
        
        // New settings
        enable_parallel_ = static_cast<bool>(get_setting("mcmc_enable_parallel", 0.0));
        num_threads_ = static_cast<int>(get_setting("mcmc_num_threads", static_cast<double>(num_threads_)));
        enable_vectorized_proposals_ = static_cast<bool>(get_setting("mcmc_enable_vectorization", 1.0));
        vector_batch_size_ = static_cast<int>(get_setting("mcmc_vector_batch_size", 64.0));
        enable_adaptive_step_ = static_cast<bool>(get_setting("mcmc_adaptive_step", 1.0));
        target_acceptance_rate_ = get_setting("mcmc_target_acceptance", 0.234);
        adaptation_rate_ = get_setting("mcmc_adaptation_rate", 0.05);
    }

    OptimizationResult MetropolisHastingsSampler::optimize(
        const Eigen::VectorXd& initialParameters,
        IObjectiveFunction& objectiveFunction,
        IParameterManager& parameterManager) {
            
        OptimizationResult result;
        result.bestParameters = initialParameters;
        result.bestObjectiveValue = objectiveFunction.calculate(initialParameters);

        Eigen::VectorXd current_params = initialParameters;
        double current_logL = result.bestObjectiveValue;
        
        if (!std::isfinite(current_logL)) {
            current_logL = -std::numeric_limits<double>::infinity();
        }

        int total_iterations = burn_in_ + mcmc_iterations_;
        result.samples.reserve(mcmc_iterations_ / thinning_ + 1);
        result.sampleObjectiveValues.reserve(mcmc_iterations_ / thinning_ + 1);

        accepted_count_ = 0;
        total_evaluations_ = 0;
        
        double adaptive_step_size = step_size_;
        int acceptance_window = 100;
        std::vector<bool> recent_accepts;
        recent_accepts.reserve(acceptance_window);

        for (int iter = 0; iter < total_iterations; ++iter) {
            // Adaptive proposal strategy
            bool use_all_params = (iter % 2 == 0) || (iter < burn_in_);
            double current_step = use_all_params ? adaptive_step_size : adaptive_step_size * 2.0;
            
            // Generate proposal
            Eigen::VectorXd proposal;
            if (enable_vectorized_proposals_ && use_all_params) {
                // Vectorized proposal generation for all parameters
                Eigen::MatrixXd proposals = generateVectorizedProposals(
                    current_params, 1, current_step, parameterManager);
                proposal = proposals.col(0);
            } else {
                // Single parameter update
                proposal = current_params;
                size_t param_idx = std::uniform_int_distribution<size_t>(0, 
                    parameterManager.getParameterCount() - 1)(gen_);
                double sigma = parameterManager.getSigmaForParamIndex(param_idx);
                std::normal_distribution<> dist(0.0, sigma * current_step);
                proposal[param_idx] += dist(gen_);
                proposal = parameterManager.applyConstraints(proposal);
            }
            
            double proposal_logL = objectiveFunction.calculate(proposal);
            total_evaluations_++;
            
            // Enhanced refinement with vectorization
            if (enable_refinement_ && std::isfinite(proposal_logL)) {
                Eigen::VectorXd step_direction = proposal - current_params;
                
                // Try reverse direction
                Eigen::VectorXd reverse_proposal = current_params - step_direction;
                reverse_proposal = parameterManager.applyConstraints(reverse_proposal);
                double reverse_logL = objectiveFunction.calculate(reverse_proposal);
                total_evaluations_++;
                
                Eigen::VectorXd best_proposal = proposal;
                double best_logL = proposal_logL;
                
                if (std::isfinite(reverse_logL) && reverse_logL > proposal_logL) {
                    best_proposal = reverse_proposal;
                    best_logL = reverse_logL;
                    step_direction = -step_direction;
                }
                
                if (refinement_steps_ > 0 && step_direction.squaredNorm() > 1e-12) {
                    auto [refined_params, refined_logL] = performVectorizedRefinement(
                        best_proposal, best_logL, step_direction,
                        refinement_steps_, objectiveFunction, parameterManager);
                    
                    if (std::isfinite(refined_logL) && refined_logL > best_logL) {
                        proposal = refined_params;
                        proposal_logL = refined_logL;
                    }
                }
            }
            
            // Metropolis-Hastings acceptance
            bool accepted = false;
            if (std::isfinite(proposal_logL)) {
                double log_ratio = proposal_logL - current_logL;
                if (log_ratio >= 0.0 || std::log(unif_(gen_)) < log_ratio) {
                    current_params = proposal;
                    current_logL = proposal_logL;
                    accepted = true;
                    accepted_count_++;
                }
            }
            
            // Track acceptance for adaptation
            if (enable_adaptive_step_ && iter >= burn_in_) {
                recent_accepts.push_back(accepted);
                if (static_cast<int>(recent_accepts.size()) > acceptance_window) {
                    recent_accepts.erase(recent_accepts.begin());
                }
                
                if (iter % acceptance_window == 0 && !recent_accepts.empty()) {
                    double recent_rate = std::accumulate(recent_accepts.begin(), 
                        recent_accepts.end(), 0.0) / static_cast<int>(recent_accepts.size());
                    updateStepSize(recent_rate);
                    adaptive_step_size = step_size_;
                }
            }
            
            // Sample collection
            if (iter >= burn_in_ && (iter - burn_in_) % thinning_ == 0) {
                result.samples.push_back(current_params);
                result.sampleObjectiveValues.push_back(current_logL);
                
                if (current_logL > result.bestObjectiveValue) {
                    result.bestObjectiveValue = current_logL;
                    result.bestParameters = current_params;
                }
            }
            
            // Progress reporting
            if ((iter + 1) % report_interval_ == 0 || iter == total_iterations - 1) {
                double acceptance_rate = (iter >= burn_in_) ? 
                    (static_cast<double>(accepted_count_) / (iter - burn_in_ + 1)) * 100.0 : 0.0;
                    
                std::ostringstream oss;
                oss << "Iter: " << std::setw(7) << (iter + 1) << "/" << total_iterations
                    << " | Current Obj: " << std::fixed << std::setprecision(4) << current_logL
                    << " | Accept Rate: " << std::setprecision(2) << acceptance_rate << "%"
                    << " | Step Size: " << std::scientific << std::setprecision(3) << adaptive_step_size
                    << " | Evaluations: " << total_evaluations_.load();
                logger.info(LOG_SOURCE, oss.str());
            }
        }
        
        // Calculate posterior mean with vectorization
        if (calculate_posterior_mean_ && !result.samples.empty()) {
            Eigen::VectorXd posterior_mean = Eigen::VectorXd::Zero(current_params.size());
            
            // Vectorized accumulation
            #ifdef _OPENMP
            #pragma omp simd reduction(+:posterior_mean)
            #endif
            for (const auto& sample : result.samples) {
                posterior_mean += sample;
            }
            posterior_mean /= static_cast<double>(result.samples.size());
            
            posterior_mean = parameterManager.applyConstraints(posterior_mean);
            double posterior_mean_logL = objectiveFunction.calculate(posterior_mean);
            
            if (std::isfinite(posterior_mean_logL) && posterior_mean_logL > result.bestObjectiveValue) {
                result.bestObjectiveValue = posterior_mean_logL;
                result.bestParameters = posterior_mean;
            }
        }
        
        logger.info(LOG_SOURCE, "MCMC Finished. Samples: " + std::to_string(result.samples.size()) +
                    ", Total evaluations: " + std::to_string(total_evaluations_.load()));
        
        if (!result.samples.empty()) {
            std::string samples_dir = FileUtils::joinPaths(FileUtils::getProjectRoot(), "data/mcmc_samples");
            FileUtils::ensureDirectoryExists(samples_dir);
            auto now = std::chrono::system_clock::now();
            auto now_time = std::chrono::system_clock::to_time_t(now);
            std::stringstream timestamp_ss;
            timestamp_ss << std::put_time(std::localtime(&now_time), "%Y%m%d_%H%M%S");
            std::string samples_file = FileUtils::joinPaths(samples_dir, "mcmc_samples_" + timestamp_ss.str() + ".csv");
            saveSamplesToCSV(result.samples, result.sampleObjectiveValues, parameterManager.getParameterNames(), samples_file);
        }
        
        return result;
    }

    Eigen::MatrixXd MetropolisHastingsSampler::generateVectorizedProposals(
        const Eigen::VectorXd& current,
        int num_proposals,
        double step_coef,
        IParameterManager& paramManager) {
        
        size_t n_params = current.size();
        Eigen::MatrixXd proposals(n_params, num_proposals);
        
        // Generate random perturbations in batch
        for (int j = 0; j < num_proposals; ++j) {
            Eigen::VectorXd perturbation(n_params);
            for (size_t i = 0; i < n_params; ++i) {
                double sigma = paramManager.getSigmaForParamIndex(i) * step_coef;
                if (sigma <= 0) sigma = 0.01 * step_coef;
                std::normal_distribution<> dist(0.0, sigma);
                perturbation[i] = dist(gen_);
            }
            proposals.col(j) = current + perturbation;
            proposals.col(j) = paramManager.applyConstraints(proposals.col(j));
        }
        
        return proposals;
    }

    std::vector<double> MetropolisHastingsSampler::evaluateProposalsParallel(
        const Eigen::MatrixXd& proposals,
        IObjectiveFunction& objectiveFunction) {
        
        int n_proposals = proposals.cols();
        std::vector<double> results(n_proposals);
        
        if (enable_parallel_ && n_proposals > 4) {
             #ifdef _OPENMP
            #pragma omp parallel for num_threads(num_threads_)
            #endif
            for (int i = 0; i < n_proposals; ++i) {
                results[i] = objectiveFunction.calculate(proposals.col(i));
                total_evaluations_++;
            }
        } else {
            for (int i = 0; i < n_proposals; ++i) {
                results[i] = objectiveFunction.calculate(proposals.col(i));
                total_evaluations_++;
            }
        }
        
        return results;
    }

    std::pair<Eigen::VectorXd, double> MetropolisHastingsSampler::performVectorizedRefinement(
        const Eigen::VectorXd& initial,
        double initial_logL,
        const Eigen::VectorXd& step_direction,
        int num_refinement_steps,
        IObjectiveFunction& objectiveFunction,
        IParameterManager& parameterManager) {
        
        Eigen::VectorXd best_point = initial;
        double best_logL = initial_logL;
        
        if (!std::isfinite(initial_logL) || step_direction.squaredNorm() < 1e-18) {
            return {initial, initial_logL};
        }
        
        // Elongation phase with vectorized evaluation
        std::vector<double> factors = {1.0, 2.0, 4.0, 8.0};
        Eigen::MatrixXd elongation_points(initial.size(), factors.size());
        
        for (size_t i = 0; i < factors.size(); ++i) {
            elongation_points.col(i) = initial + factors[i] * step_direction;
            elongation_points.col(i) = parameterManager.applyConstraints(elongation_points.col(i));
        }
        
        std::vector<double> elongation_logLs = evaluateProposalsParallel(elongation_points, objectiveFunction);
        
        for (size_t i = 0; i < factors.size(); ++i) {
            if (std::isfinite(elongation_logLs[i]) && elongation_logLs[i] > best_logL) {
                best_point = elongation_points.col(i);
                best_logL = elongation_logLs[i];
            } else {
                break; // Stop elongation if no improvement
            }
        }
        
        // Binary refinement with batch evaluation
        if (num_refinement_steps > 0) {
            double alpha = 1.0;
            Eigen::VectorXd base_point = best_point;
            
            for (int k = 0; k < num_refinement_steps; ++k) {
                alpha *= 0.5;
                
                // Evaluate both directions simultaneously
                Eigen::MatrixXd refinement_points(initial.size(), 2);
                refinement_points.col(0) = base_point - alpha * step_direction;
                refinement_points.col(1) = base_point + alpha * step_direction;
                
                refinement_points.col(0) = parameterManager.applyConstraints(refinement_points.col(0));
                refinement_points.col(1) = parameterManager.applyConstraints(refinement_points.col(1));
                
                std::vector<double> refinement_logLs = evaluateProposalsParallel(
                    refinement_points, objectiveFunction);
                
                for (int i = 0; i < 2; ++i) {
                    if (std::isfinite(refinement_logLs[i]) && refinement_logLs[i] > best_logL) {
                        best_point = refinement_points.col(i);
                        best_logL = refinement_logLs[i];
                        base_point = best_point;
                    }
                }
            }
        }
        
        return {best_point, best_logL};
    }

    void MetropolisHastingsSampler::updateStepSize(double acceptance_rate) {
        // Robbins-Monro stochastic approximation
        double log_ratio = std::log(acceptance_rate / target_acceptance_rate_);
        step_size_ *= std::exp(adaptation_rate_ * log_ratio);
        
        // Keep step size in reasonable bounds
        step_size_ = std::max(1e-6, std::min(10.0, step_size_));
    }

    void MetropolisHastingsSampler::saveSamplesToCSV(
        const std::vector<Eigen::VectorXd>& samples,
        const std::vector<double>& objectiveValues,
        const std::vector<std::string>& parameterNames,
        const std::string& filepath) {
        
        std::ofstream file(filepath);
        if (!file.is_open()) {
            logger.error(LOG_SOURCE, "Failed to open CSV: " + filepath);
            return;
        }
        
        file << "sample_id,objective_value";
        for (const auto& name : parameterNames) {
            file << "," << name;
        }
        file << "\n";
        
        for (size_t i = 0; i < samples.size(); ++i) {
            file << i << "," << objectiveValues[i];
            for (int j = 0; j < samples[i].size(); ++j) {
                file << "," << std::scientific << std::setprecision(10) << samples[i][j];
            }
            file << "\n";
        }
        
        file.close();
        logger.info(LOG_SOURCE, "Saved " + std::to_string(samples.size()) + " samples to: " + filepath);
    }

} // namespace epidemic