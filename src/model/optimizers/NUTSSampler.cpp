#include "model/optimizers/NUTSSampler.hpp"
#include "model/interfaces/IGradientObjectiveFunction.hpp"
#include "sir_age_structured/interfaces/IParameterManager.hpp"
#include "utils/Logger.hpp"
#include "exceptions/Exceptions.hpp"
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <iomanip>

namespace epidemic {

NUTSSampler::NUTSSampler()
    : num_warmup_(1000),
      num_samples_(1000),
      delta_target_(0.8),
      max_tree_depth_(10),
      rng_(std::random_device{}()) {}

void NUTSSampler::configure(const std::map<std::string, double>& settings) {
    auto get_setting = [&](const std::string& name, double default_val) {
        auto it = settings.find(name);
        return (it != settings.end()) ? it->second : default_val;
    };
    
    num_warmup_ = static_cast<int>(get_setting("nuts_warmup", 1000.0));
    num_samples_ = static_cast<int>(get_setting("nuts_samples", 1000.0));
    delta_target_ = get_setting("nuts_delta_target", 0.8);
    max_tree_depth_ = static_cast<int>(get_setting("nuts_max_tree_depth", 10.0));
    
    Logger::getInstance().info("NUTSSampler", 
        "Configured with warmup=" + std::to_string(num_warmup_) +
        ", samples=" + std::to_string(num_samples_) +
        ", delta_target=" + std::to_string(delta_target_) +
        ", max_tree_depth=" + std::to_string(max_tree_depth_));
}

OptimizationResult NUTSSampler::optimize(
    const Eigen::VectorXd& initialParameters,
    IObjectiveFunction& objectiveFunction,
    IParameterManager& parameterManager) {
    
    Logger& logger = Logger::getInstance();
    logger.info("NUTSSampler", "Starting NUTS sampling.");
    
    auto* grad_obj = dynamic_cast<IGradientObjectiveFunction*>(&objectiveFunction);
    if (!grad_obj) {
        THROW_INVALID_PARAM("NUTSSampler", 
            "Objective function must implement IGradientObjectiveFunction for NUTS.");
    }
    
    OptimizationResult result;
    result.samples.reserve(num_samples_);
    result.sampleObjectiveValues.reserve(num_samples_);
    
    Eigen::VectorXd theta_m = initialParameters;
    
    double epsilon = findReasonableEpsilon(*grad_obj, theta_m, parameterManager);
    logger.info("NUTSSampler", "Initial step size (epsilon): " + std::to_string(epsilon));
    
    // Dual averaging parameters for step size adaptation (Hoffman & Gelman, 2014, Sec 3.2.1)
    double mu = std::log(10.0 * epsilon);
    double epsilon_bar = 1.0;
    double H_bar = 0.0;
    const double gamma = 0.05;
    const double t0 = 10.0;
    const double kappa = 0.75;
    
    int total_iterations = num_warmup_ + num_samples_;
    
    // Main sampling loop (implements Algorithm 6)
    for (int m = 1; m <= total_iterations; ++m) {
        // --- Algorithm 6, Step 1: Resample momentum and slice variable ---
        std::normal_distribution<> normal(0.0, 1.0);
        Eigen::VectorXd r0 = Eigen::VectorXd::NullaryExpr(theta_m.size(), [&](){ return normal(rng_); });
        
        std::vector<double> grad_unused;
        double current_logp = grad_obj->evaluate_with_gradient({theta_m.data(), theta_m.data() + theta_m.size()}, grad_unused);
        if (!std::isfinite(current_logp)) {
            logger.warning("NUTSSampler", "Initial log-probability is non-finite. Aborting iteration " + std::to_string(m) + ". Try a different initial position.");
            if (m > num_warmup_) result.samples.push_back(result.samples.back()); // Re-use last good sample
            continue;
        }

        // Sample slice variable u using a numerically stable trick: log(u) = H0 - Exp(1)
        double H0 = current_logp - 0.5 * r0.dot(r0);
        double log_u_slice = H0 - std::exponential_distribution<>(1.0)(rng_);

        // --- Algorithm 6, Step 2: Initialize the tree ---
        Eigen::VectorXd theta_minus = theta_m, theta_plus = theta_m;
        Eigen::VectorXd r_minus = r0, r_plus = r0;
        int j = 0;         // tree depth
        int n_total = 1;   // total valid points in trajectory
        bool s = true;     // termination flag

        double alpha_sum = 0.0;
        int n_alpha = 0;

        // --- Algorithm 6, Step 3: Build the tree iteratively ---
        while (s && j < max_tree_depth_) {
            int v = (std::uniform_int_distribution<>(0, 1)(rng_) * 2 - 1); // -1 or 1
            
            Tree subtree;
            if (v == -1) {
                buildTree(*grad_obj, theta_minus, r_minus, log_u_slice, v, j, epsilon, H0, parameterManager, subtree);
                theta_minus = subtree.theta_minus;
                r_minus = subtree.r_minus;
            } else {
                buildTree(*grad_obj, theta_plus, r_plus, log_u_slice, v, j, epsilon, H0, parameterManager, subtree);
                theta_plus = subtree.theta_plus;
                r_plus = subtree.r_plus;
            }

            // Probabilistically update the sample for this iteration (memory-efficient version)
            if (subtree.s && subtree.n_valid > 0) {
                if (std::uniform_real_distribution<>(0.0, 1.0)(rng_) < static_cast<double>(subtree.n_valid) / static_cast<double>(n_total)) {
                    theta_m = subtree.theta_prime;
                }
            }
            
            n_total += subtree.n_valid;
            s = subtree.s && checkNoUTurn(theta_minus, theta_plus, r_minus, r_plus);
            alpha_sum += subtree.alpha;
            n_alpha += subtree.n_alpha;
            j++;
        }
        
        // --- Algorithm 6, Step 4: Adapt step size during warmup phase ---
        if (m <= num_warmup_) {
            double eta = 1.0 / (m + t0);
            double avg_metro_prob = (n_alpha > 0) ? alpha_sum / n_alpha : 0.0;
            H_bar = (1.0 - eta) * H_bar + eta * (delta_target_ - avg_metro_prob);
            
            epsilon = std::exp(mu - std::sqrt(m) / gamma * H_bar);
            double eta_power = std::pow(m, -kappa);
            epsilon_bar = std::exp(eta_power * std::log(epsilon) + (1.0 - eta_power) * std::log(epsilon_bar));
            
            if (m == num_warmup_) {
                logger.info("NUTSSampler", "Warmup complete. Final adapted step size (epsilon_bar): " + std::to_string(epsilon_bar));
            }
        } else {
            epsilon = epsilon_bar; // Use the adapted average epsilon post-warmup
        }
        
        // Store sample after warmup
        if (m > num_warmup_) {
            result.samples.push_back(parameterManager.applyConstraints(theta_m));
            double sample_logp = objectiveFunction.calculate(result.samples.back());
            result.sampleObjectiveValues.push_back(sample_logp);
            
            if (sample_logp > result.bestObjectiveValue) {
                result.bestObjectiveValue = sample_logp;
                result.bestParameters = result.samples.back();
            }
        }
        
        // Progress reporting
        if (m % 1 == 0 || m == total_iterations) {
            std::ostringstream msg;
            msg << "Progress: " << m << "/" << total_iterations;
            if (m <= num_warmup_) {
                double avg_prob = (n_alpha > 0) ? alpha_sum / n_alpha : 0.0;
                msg << " (Warmup) | eps=" << std::scientific << std::setprecision(3) << epsilon
                    << " | Avg. Metro. Accept=" << std::fixed << std::setprecision(3) << avg_prob;
            } else {
                msg << " (Sampling) | Best obj=" << std::fixed << std::setprecision(4) << result.bestObjectiveValue;
            }
            logger.info("NUTSSampler", msg.str());
        }
    }
    
    logger.info("NUTSSampler", "NUTS sampling completed. Generated " + 
                std::to_string(result.samples.size()) + " samples.");
    
    return result;
}

double NUTSSampler::findReasonableEpsilon(
    IGradientObjectiveFunction& objective,
    const Eigen::VectorXd& theta,
    IParameterManager& parameterManager) const {
    
    double epsilon = 1.0;
    std::normal_distribution<> normal(0.0, 1.0);
    Eigen::VectorXd r = Eigen::VectorXd::NullaryExpr(theta.size(), [&](){ return normal(rng_); });
    
    std::vector<double> grad_unused;
    double logp = objective.evaluate_with_gradient({theta.data(), theta.data() + theta.size()}, grad_unused);
    double initial_H = logp - 0.5 * r.dot(r);
    
    Eigen::VectorXd theta_new = theta;
    Eigen::VectorXd r_new = r;
    leapfrog(objective, theta_new, r_new, epsilon, parameterManager);
    
    double logp_new = objective.evaluate_with_gradient({theta_new.data(), theta_new.data() + theta_new.size()}, grad_unused);
    double H_new = logp_new - 0.5 * r_new.dot(r_new);
    
    double p_accept = std::exp(H_new - initial_H);
    double a = (p_accept > 0.5) ? 1.0 : -1.0;
    
    // This loop implements the heuristic from Hoffman & Gelman (2014), Algorithm 4.
    // It doubles/halves epsilon until the acceptance probability of a single leapfrog step crosses 0.5.
    while (std::pow(p_accept, a) > std::pow(2.0, -a)) {
        epsilon = std::pow(2.0, a) * epsilon;
        
        theta_new = theta;
        r_new = r;
        leapfrog(objective, theta_new, r_new, epsilon, parameterManager);
        
        logp_new = objective.evaluate_with_gradient({theta_new.data(), theta_new.data() + theta_new.size()}, grad_unused);
        H_new = logp_new - 0.5 * r_new.dot(r_new);
        p_accept = std::exp(H_new - initial_H);

        if (epsilon > 1e4 || epsilon < 1e-7) break; // Safety break for extreme cases
    }
    
    return epsilon;
}

void NUTSSampler::leapfrog(
    IGradientObjectiveFunction& objective,
    Eigen::VectorXd& theta,
    Eigen::VectorXd& r,
    double epsilon,
    IParameterManager& parameterManager) const {
    
    std::vector<double> grad_vec;
    objective.evaluate_with_gradient({theta.data(), theta.data() + theta.size()}, grad_vec);
    Eigen::Map<Eigen::VectorXd> grad_start(grad_vec.data(), grad_vec.size());

    r += 0.5 * epsilon * grad_start;
    theta += epsilon * r;
    theta = parameterManager.applyConstraints(theta);
    
    objective.evaluate_with_gradient({theta.data(), theta.data() + theta.size()}, grad_vec);
    Eigen::Map<Eigen::VectorXd> grad_end(grad_vec.data(), grad_vec.size());
    r += 0.5 * epsilon * grad_end;
}

void NUTSSampler::buildTree(
    IGradientObjectiveFunction& objective,
    const Eigen::VectorXd& theta, const Eigen::VectorXd& r,
    double log_u_slice, int v_direction, int j_depth, double epsilon,
    double H0, IParameterManager& parameterManager,
    Tree& tree) const {
    
    if (j_depth == 0) {
        // Base case: take one leapfrog step in the given direction
        Eigen::VectorXd theta_prime = theta;
        Eigen::VectorXd r_prime = r;
        leapfrog(objective, theta_prime, r_prime, v_direction * epsilon, parameterManager);
        
        std::vector<double> grad_unused;
        double logp = objective.evaluate_with_gradient({theta_prime.data(), theta_prime.data() + theta_prime.size()}, grad_unused);
        
        tree.theta_minus = tree.theta_plus = tree.theta_prime = theta_prime;
        tree.r_minus = tree.r_plus = r_prime;

        if (!std::isfinite(logp)) {
            tree.s = false; // Terminate due to non-finite probability
            return;
        }

        double H_new = logp - 0.5 * r_prime.dot(r_prime);
        tree.n_valid = (log_u_slice <= H_new) ? 1 : 0;
        tree.s = (log_u_slice <= H_new + DELTA_MAX);
        tree.alpha = std::min(1.0, std::exp(H_new - H0));
        tree.n_alpha = 1;
        return;
    }
    
    // Recursion: build left and right subtrees
    Tree left_subtree;
    buildTree(objective, theta, r, log_u_slice, v_direction, j_depth - 1, epsilon, H0, parameterManager, left_subtree);
    if (!left_subtree.s) { tree = left_subtree; return; }
    
    Tree right_subtree;
    if (v_direction == -1) {
        buildTree(objective, left_subtree.theta_minus, left_subtree.r_minus, log_u_slice, v_direction, j_depth - 1, epsilon, H0, parameterManager, right_subtree);
        tree.theta_minus = right_subtree.theta_minus;
        tree.r_minus = right_subtree.r_minus;
        tree.theta_plus = left_subtree.theta_plus;
        tree.r_plus = left_subtree.r_plus;
    } else {
        buildTree(objective, left_subtree.theta_plus, left_subtree.r_plus, log_u_slice, v_direction, j_depth - 1, epsilon, H0, parameterManager, right_subtree);
        tree.theta_minus = left_subtree.theta_minus;
        tree.r_minus = left_subtree.r_minus;
        tree.theta_plus = right_subtree.theta_plus;
        tree.r_plus = right_subtree.r_plus;
    }
    
    if (!right_subtree.s) { tree = right_subtree; return; }

    // Combine results from subtrees
    tree.n_valid = left_subtree.n_valid + right_subtree.n_valid;
    if (tree.n_valid > 0) {
        if (std::uniform_real_distribution<>(0.0, 1.0)(rng_) < static_cast<double>(right_subtree.n_valid) / static_cast<double>(tree.n_valid)) {
            tree.theta_prime = right_subtree.theta_prime;
        } else {
            tree.theta_prime = left_subtree.theta_prime;
        }
    } else {
        tree.theta_prime = left_subtree.theta_prime; // Propagate a value even if invalid
    }
    
    tree.alpha = left_subtree.alpha + right_subtree.alpha;
    tree.n_alpha = left_subtree.n_alpha + right_subtree.n_alpha;
    tree.s = checkNoUTurn(tree.theta_minus, tree.theta_plus, tree.r_minus, tree.r_plus);
}

bool NUTSSampler::checkNoUTurn(
    const Eigen::VectorXd& theta_minus,
    const Eigen::VectorXd& theta_plus,
    const Eigen::VectorXd& r_minus,
    const Eigen::VectorXd& r_plus) const {
    
    Eigen::VectorXd delta_theta = theta_plus - theta_minus;
    return (delta_theta.dot(r_minus) >= 0) && (delta_theta.dot(r_plus) >= 0);
}

} // namespace epidemic