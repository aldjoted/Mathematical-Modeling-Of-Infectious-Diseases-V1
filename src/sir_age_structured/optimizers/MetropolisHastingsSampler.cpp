#include "sir_age_structured/optimizers/MetropolisHastingsSampler.hpp"
#include "utils/Logger.hpp"
#include <numeric>
#include <cmath>
#include <limits>
#include <iostream>
#include <iomanip>
#include "utils/FileUtils.hpp"
#include <chrono>
#include <fstream>
#include <sstream>

namespace epidemic {
    static Logger& logger = Logger::getInstance();
    static const std::string LOG_SOURCE = "MHS"; 

    MetropolisHastingsSampler::MetropolisHastingsSampler()
        :gen_(std::random_device{}()), unif_(0.0, 1.0) {}
    
    void MetropolisHastingsSampler::configure(const std::map<std::string, double>& settings) {
        auto get_setting = [&](const std::string& name, double default_val) {
            auto it = settings.find(name);
            return (it != settings.end()) ? it->second : default_val;
        };
        burn_in_ = static_cast<int>(get_setting("burn_in", static_cast<double>(burn_in_)));
        mcmc_iterations_ = static_cast<int>(get_setting("mcmc_iterations", static_cast<double>(mcmc_iterations_)));
        thinning_ = static_cast<int>(get_setting("thinning", static_cast<double>(thinning_)));
        step_size_ = get_setting("mcmc_step_size", step_size_);
        calculate_posterior_mean_ = static_cast<bool>(get_setting("calculate_posterior_mean", calculate_posterior_mean_ ? 1.0 : 0.0));
        report_interval_ = static_cast<int>(get_setting("report_interval", static_cast<double>(report_interval_)));
        enable_refinement_ = static_cast<bool>(get_setting("mcmc_enable_refinement", enable_refinement_ ? 1.0 : 0.0));
        refinement_steps_ = static_cast<int>(get_setting("mcmc_refinement_steps", static_cast<double>(refinement_steps_)));

        if (burn_in_ < 0) burn_in_ = 5000;
        if (mcmc_iterations_ <= 0) mcmc_iterations_ = 10000;
        if (thinning_ <= 0) thinning_ = 10;
        if (step_size_ <= 0) step_size_ = 0.1;
        if (report_interval_ <= 0) report_interval_ = 200;
        if (refinement_steps_ < 0) refinement_steps_ = 0;

        std::ostringstream oss;
        oss << "Configured: burn_in=" << burn_in_
            << ", mcmc_iterations=" << mcmc_iterations_
            << ", thinning=" << thinning_
            << ", step_size=" << step_size_
            << ", calculate_posterior_mean=" << calculate_posterior_mean_
            << ", report_interval=" << report_interval_
            << ", mcmc_enable_refinement=" << enable_refinement_
            << ", mcmc_refinement_steps=" << refinement_steps_;
        logger.info(LOG_SOURCE, oss.str());
    }

    OptimizationResult MetropolisHastingsSampler::optimize(
        const Eigen::VectorXd& initialParameters,
        IObjectiveFunction& objectiveFunction,
        IParameterManager& parameterManager) {
            
            OptimizationResult result;
            result.bestParameters = initialParameters;
            result.bestObjectiveValue = objectiveFunction.calculate(initialParameters);

            Eigen::VectorXd P0_current_params_vec = initialParameters;
            double current_logL = result.bestObjectiveValue;

            if (std::isinf(current_logL) || std::isnan(current_logL)) {
                current_logL = -std::numeric_limits<double>::infinity();
            }
            
            logger.info(LOG_SOURCE, "Starting MCMC Sampling. Initial Obj: " + std::to_string(current_logL));

            int accepted_post_burnin_count = 0;
            int samples_collected = 0;
            int total_iterations = burn_in_ + mcmc_iterations_;

            result.samples.reserve(mcmc_iterations_ / thinning_ + 1);
            result.sampleObjectiveValues.reserve(mcmc_iterations_ / thinning_ + 1);

            for (int iter = 0; iter < total_iterations; ++iter) {
                Eigen::VectorXd P1_proposal_fwd = P0_current_params_vec;
                if (iter % 2 == 0 || iter < burn_in_) {
                    randomStepAll(P1_proposal_fwd, step_size_, parameterManager);
                } else { 
                    randomStepOne(P1_proposal_fwd, step_size_ * 2.0, parameterManager);
                }
                double P1_proposal_fwd_logL = objectiveFunction.calculate(P1_proposal_fwd);
                Eigen::VectorXd fwd_step_direction = P1_proposal_fwd - P0_current_params_vec;

                Eigen::VectorXd chosen_initial_proposal_for_refinement = P1_proposal_fwd;
                double chosen_initial_logL_for_refinement = P1_proposal_fwd_logL;
                Eigen::VectorXd chosen_step_direction_for_refinement = fwd_step_direction;

                if (enable_refinement_) {
                    Eigen::VectorXd P1_proposal_bwd = P0_current_params_vec - fwd_step_direction;
                    P1_proposal_bwd = parameterManager.applyConstraints(P1_proposal_bwd);
                    double P1_proposal_bwd_logL = objectiveFunction.calculate(P1_proposal_bwd);

                    if (std::isfinite(P1_proposal_bwd_logL) && 
                        (!std::isfinite(P1_proposal_fwd_logL) || P1_proposal_bwd_logL > P1_proposal_fwd_logL)) {
                        chosen_initial_proposal_for_refinement = P1_proposal_bwd;
                        chosen_initial_logL_for_refinement = P1_proposal_bwd_logL;
                        chosen_step_direction_for_refinement = P1_proposal_bwd - P0_current_params_vec;
                    }
                }
                
                Eigen::VectorXd final_proposal_vec = chosen_initial_proposal_for_refinement;
                double final_proposal_logL = chosen_initial_logL_for_refinement;

                if (enable_refinement_ && refinement_steps_ >= 0 &&
                    std::isfinite(chosen_initial_logL_for_refinement) && chosen_initial_logL_for_refinement != -std::numeric_limits<double>::infinity()) {
                    if (chosen_step_direction_for_refinement.squaredNorm() > 1e-12) { 
                        std::pair<Eigen::VectorXd, double> refined_output = perform_refinement(
                            chosen_initial_proposal_for_refinement,
                            chosen_initial_logL_for_refinement,
                            chosen_step_direction_for_refinement,
                            refinement_steps_,
                            objectiveFunction,
                            parameterManager
                        );

                        if (std::isfinite(refined_output.second) && refined_output.second > final_proposal_logL) {
                            final_proposal_vec = refined_output.first;
                            final_proposal_logL = refined_output.second;
                        }
                    }
                }
                
                bool accepted_this_mcmc_step = false;
                if (!std::isnan(final_proposal_logL) && !std::isinf(final_proposal_logL)) {
                    if (!std::isinf(current_logL) && !std::isnan(current_logL)) {
                        double log_ratio = final_proposal_logL - current_logL;
                        if (log_ratio >= 0.0 || std::log(unif_(gen_)) < log_ratio) {
                            P0_current_params_vec = final_proposal_vec;
                            current_logL = final_proposal_logL;
                            accepted_this_mcmc_step = true;
                        }
                    } else { 
                        P0_current_params_vec = final_proposal_vec;
                        current_logL = final_proposal_logL;
                        accepted_this_mcmc_step = true;
                    }
                }

                if (iter >= burn_in_) {
                    if (accepted_this_mcmc_step) {
                        accepted_post_burnin_count++;
                    }
                    if ((iter - burn_in_) % thinning_ == 0) {
                        result.samples.push_back(P0_current_params_vec);
                        result.sampleObjectiveValues.push_back(current_logL);
                        samples_collected++;
                        if (!std::isnan(current_logL) && !std::isinf(current_logL) &&
                        current_logL > result.bestObjectiveValue) {
                            result.bestObjectiveValue = current_logL;
                            result.bestParameters = P0_current_params_vec;
                        }
                    }
                }

                if ((iter + 1) % report_interval_ == 0 || iter == total_iterations - 1) {
                    double acceptance_rate = 0.0;
                    int post_burnin_steps = iter - burn_in_ + 1;
                    if (iter >= burn_in_ && post_burnin_steps > 0) {
                        acceptance_rate = (static_cast<double>(accepted_post_burnin_count) / post_burnin_steps) * 100.0;
                    }
                    std::ostringstream report_oss;
                    report_oss  << "Iter: " << std::setw(7) << (iter + 1) << "/" << total_iterations
                                << " | Stage: " << (iter < burn_in_ ? "Burn-in " : "Sampling")
                                << " | Current Obj: " << std::fixed << std::setprecision(4) << std::setw(12) << current_logL
                                << " | Accept Rate (post-burn): " << std::fixed << std::setprecision(2) << std::setw(6) << acceptance_rate << "%"
                                << " | Samples: " << std::setw(6) << samples_collected;
                    logger.info(LOG_SOURCE, report_oss.str());
                }
            } 
            
            if (calculate_posterior_mean_ && !result.samples.empty()) {
                Eigen::VectorXd posterior_mean = Eigen::VectorXd::Zero(P0_current_params_vec.size());
                int valid_samples_for_mean = 0;
                for (const auto& sample : result.samples) {
                    if (!sample.hasNaN() && sample.allFinite()) {
                        posterior_mean += sample;
                        valid_samples_for_mean++;
                    }
                }
                if (valid_samples_for_mean > 0) {
                    posterior_mean /= static_cast<double>(valid_samples_for_mean);
                    posterior_mean = parameterManager.applyConstraints(posterior_mean);
                    double posterior_mean_logL = objectiveFunction.calculate(posterior_mean);
                    if (!std::isnan(posterior_mean_logL) && !std::isinf(posterior_mean_logL) &&
                    posterior_mean_logL > result.bestObjectiveValue) {
                        result.bestObjectiveValue = posterior_mean_logL;
                        result.bestParameters = posterior_mean;
                    }
                }
            }
            
            logger.info(LOG_SOURCE, "MCMC Finished. Samples collected: " + std::to_string(result.samples.size()));
            logger.info(LOG_SOURCE, "Final Best Objective Value: " + std::to_string(result.bestObjectiveValue));


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

    std::pair<Eigen::VectorXd, double> MetropolisHastingsSampler::perform_refinement(
        const Eigen::VectorXd& P1_initial,
        double P1_initial_logL,
        const Eigen::VectorXd& step_direction,
        int num_binary_refinement_steps,
        IObjectiveFunction& objectiveFunction,
        IParameterManager& parameterManager) {
        
        Eigen::VectorXd best_refined_pt = P1_initial;
        double best_refined_logL = P1_initial_logL;

        if (!std::isfinite(P1_initial_logL) || step_direction.squaredNorm() < 1e-18) {
            return {P1_initial, P1_initial_logL};
        }

        auto evaluate = [&](const Eigen::VectorXd& p) {
            Eigen::VectorXd constrained_p = parameterManager.applyConstraints(p);
            return std::make_pair(constrained_p, objectiveFunction.calculate(constrained_p));
        };

        // Step 1: Elongation
        Eigen::VectorXd current_pt = P1_initial;
        double current_logL = P1_initial_logL;
        
        bool improved = true;
        while (improved) {
            auto [next_pt, next_logL] = evaluate(current_pt + step_direction);
            
            if (std::isfinite(next_logL) && next_logL > current_logL) {
                current_pt = next_pt;
                current_logL = next_logL;
                if (current_logL > best_refined_logL) {
                    best_refined_pt = current_pt;
                    best_refined_logL = current_logL;
                }
            } else {
                improved = false;
            }
        }

        // Step 2: Binary refinement
        if (num_binary_refinement_steps > 0) {
            double alpha = 1.0;
            Eigen::VectorXd base_pt = best_refined_pt;
            
            for (int k = 0; k < num_binary_refinement_steps; ++k) {
                alpha *= 0.5;
                
                // Try both directions at current scale
                for (int sign : {-1, 1}) {
                    auto [refined_pt, refined_logL] = evaluate(base_pt + sign * alpha * step_direction);
                    
                    if (std::isfinite(refined_logL) && refined_logL > best_refined_logL) {
                        best_refined_pt = refined_pt;
                        best_refined_logL = refined_logL;
                        // Update base point for next iteration
                        base_pt = best_refined_pt;
                    }
                }
            }
        }

        return {best_refined_pt, best_refined_logL};
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
        logger.info(LOG_SOURCE, "Saved " + std::to_string(samples.size()) + " MCMC samples to: " + filepath);
    }

    void MetropolisHastingsSampler::randomStepAll(Eigen::VectorXd& params_vec, double step_coef, IParameterManager& paramManager) {
        size_t param_count = paramManager.getParameterCount();
        if (static_cast<size_t>(params_vec.size()) != param_count) {
            THROW_INVALID_PARAM("MHS::randomStepAll", "Param vector size mismatch.");
        }
        for (size_t i = 0; i < param_count; ++i) {
            double base_sigma = paramManager.getSigmaForParamIndex(i);
            double effective_stddev = base_sigma * step_coef;
            if (base_sigma <= 0) base_sigma = 0.01;
            effective_stddev = base_sigma * step_coef;
            if (effective_stddev <= 1e-9) effective_stddev = 1e-6;
            std::normal_distribution<> dist(0.0, effective_stddev);
            params_vec[i] += dist(gen_);
        }
        params_vec = paramManager.applyConstraints(params_vec);
    }

    void MetropolisHastingsSampler::randomStepOne(Eigen::VectorXd& params_vec, double step_coef, IParameterManager& paramManager) {
        size_t param_count = paramManager.getParameterCount();
         if (param_count == 0) return;
        if (static_cast<size_t>(params_vec.size()) != param_count) {
            THROW_INVALID_PARAM("MHS::randomStepOne", "Param vector size mismatch.");
        }
        std::uniform_int_distribution<size_t> int_dist(0, param_count - 1);
        size_t selected_param = int_dist(gen_);
        double base_sigma = paramManager.getSigmaForParamIndex(selected_param);
        double effective_stddev = base_sigma * step_coef;
        if (base_sigma <= 0) base_sigma = 0.01;
        effective_stddev = base_sigma * step_coef;
        if (effective_stddev <= 1e-9) effective_stddev = 1e-6;
        std::normal_distribution<> dist(0.0, effective_stddev);
        params_vec[selected_param] += dist(gen_);
        params_vec = paramManager.applyConstraints(params_vec);
    }

} // namespace epidemic