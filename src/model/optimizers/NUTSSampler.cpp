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
#include <limits>

namespace epidemic {

NUTSSampler::NUTSSampler()
    : num_iterations_(2000),
      adaptation_window_(500),
      delta_target_(0.8),
      max_tree_depth_(10),
      rng_(std::random_device{}()) {}

void NUTSSampler::configure(const std::map<std::string, double>& settings) {
    auto get_setting = [&](const std::string& name, double default_val) {
        auto it = settings.find(name);
        return (it != settings.end()) ? it->second : default_val;
    };
    
    num_iterations_ = static_cast<int>(get_setting("nuts_iterations", 2000.0));
    adaptation_window_ = static_cast<int>(get_setting("nuts_adaptation_window", 500.0));
    delta_target_ = get_setting("nuts_delta_target", 0.8);
    max_tree_depth_ = static_cast<int>(get_setting("nuts_max_tree_depth", 10.0));
    
    Logger::getInstance().info("NUTSSampler", 
        "Configured (single-phase): iterations=" + std::to_string(num_iterations_) +
        ", adaptation_window=" + std::to_string(adaptation_window_) +
        ", delta_target=" + std::to_string(delta_target_) +
        ", max_tree_depth=" + std::to_string(max_tree_depth_));
}

OptimizationResult NUTSSampler::optimize(
    const Eigen::VectorXd& initialParameters,
    IObjectiveFunction& objectiveFunction,
    IParameterManager& parameterManager) {
    
    Logger& logger = Logger::getInstance();
    logger.info("NUTSSampler", "Starting single-phase NUTS calibration.");
    
    auto* grad_obj = dynamic_cast<IGradientObjectiveFunction*>(&objectiveFunction);
    if (!grad_obj) {
        THROW_INVALID_PARAM("NUTSSampler", 
            "Objective function must implement IGradientObjectiveFunction for NUTS.");
    }
    
    OptimizationResult result;
    result.samples.reserve(num_iterations_);
    result.sampleObjectiveValues.reserve(num_iterations_);
    
    Eigen::VectorXd theta_m = initialParameters;
    
    // === Initialization ===
    double epsilon = findReasonableEpsilon(*grad_obj, theta_m, parameterManager);
    logger.info("NUTSSampler", "Initial epsilon found: " + std::to_string(epsilon));
    
    // Dual averaging parameters for continuous adaptation
    double mu = std::log(10.0 * epsilon);
    double epsilon_bar = epsilon;  // Initialize to current epsilon
    double H_bar = 0.0;
    const double gamma = 0.05;
    const double t0 = 10.0;
    const double kappa = 0.75;
    
    for (int m = 1; m <= num_iterations_; ++m) {
        // --- Resample momentum r0 ~ N(0, I) ---
        std::normal_distribution<> normal(0.0, 1.0);
        Eigen::VectorXd r0 = Eigen::VectorXd::NullaryExpr(theta_m.size(), [&](){ return normal(rng_); });
        
        // Evaluate L(theta) and gradient
        Eigen::VectorXd grad(theta_m.size());
        double log_p = grad_obj->evaluate_with_gradient(theta_m, grad);
        
        // Scale the gradient to prevent numerical issues
        // Large gradients cause the Hamiltonian to diverge
        double grad_norm = grad.norm();
        double max_grad_norm = 1000.0;  // Cap the gradient norm
        if (grad_norm > max_grad_norm) {
            grad *= max_grad_norm / grad_norm;
            if (m <= 5) {
                logger.info("NUTSSampler", "Scaled gradient from " + std::to_string(grad_norm) + 
                            " to " + std::to_string(max_grad_norm));
            }
        }
        
        if (m <= 5) {
            logger.info("NUTSSampler", "Iteration " + std::to_string(m) + 
                        " - Gradient norm (after scaling): " + std::to_string(grad.norm()) +
                        ", log_p: " + std::to_string(log_p));
        }

        if (!std::isfinite(log_p)) {
             logger.warning("NUTSSampler", "Log probability non-finite at iteration " + std::to_string(m));
             // Store previous sample if available, skip otherwise
             if (!result.samples.empty()) {
                 result.samples.push_back(result.samples.back());
                 result.sampleObjectiveValues.push_back(result.sampleObjectiveValues.back());
             }
             continue;
        }
        
        // Check for degenerate gradient
        if (grad_norm < 1e-10) {
            logger.warning("NUTSSampler", "Gradient near zero at iteration " + std::to_string(m) + 
                          ". This may indicate numerical issues with gradient computation.");
        }

        // Hamiltonian: H = log_p - 0.5 * r.dot(r)
        double H0 = log_p - 0.5 * r0.dot(r0);

        // Sample slice variable u: log_u = H0 - Exp(1)
        double log_u_slice = H0 - std::exponential_distribution<>(1.0)(rng_);

        // Initialize Tree Variables
        Eigen::VectorXd theta_minus = theta_m;
        Eigen::VectorXd theta_plus = theta_m;
        Eigen::VectorXd r_minus = r0;
        Eigen::VectorXd r_plus = r0;
        
        int j = 0;              // Depth
        int n = 1;              // Number of valid points in current tree
        bool s = true;          // Stopping condition
        
        double alpha = 0.0;
        int n_alpha = 0;

        Eigen::VectorXd theta_next = theta_m;

        // --- Build Tree Loop ---
        while (s && j < max_tree_depth_) {
            int v = (std::uniform_int_distribution<>(0, 1)(rng_) * 2) - 1;

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

            if (subtree.s) {
                if (checkNoUTurn(theta_minus, theta_plus, r_minus, r_plus)) {
                    double acceptance_prob = static_cast<double>(subtree.n_valid) / static_cast<double>(n + subtree.n_valid);
                    if (std::uniform_real_distribution<>(0.0, 1.0)(rng_) < acceptance_prob) {
                        theta_next = subtree.theta_prime;
                    }
                    
                    n += subtree.n_valid;
                    alpha += subtree.alpha;
                    n_alpha += subtree.n_alpha;
                    j++;
                } else {
                    s = false;
                }
            } else {
                s = false;
            }
        }
        
        theta_m = theta_next;

        // --- Continuous Step-Size Adaptation (during adaptation window) ---
        if (m <= adaptation_window_) {
            double avg_alpha = (n_alpha > 0) ? alpha / n_alpha : 0.0;
            
            double eta = 1.0 / (m + t0);
            H_bar = (1.0 - eta) * H_bar + eta * (delta_target_ - avg_alpha);

            double log_epsilon = mu - (std::sqrt(m) / gamma) * H_bar;
            epsilon = std::exp(log_epsilon);

            double m_kappa = std::pow(m, -kappa);
            double log_epsilon_bar = m_kappa * log_epsilon + (1.0 - m_kappa) * std::log(epsilon_bar);
            epsilon_bar = std::exp(log_epsilon_bar);
        } else {
            // After adaptation window: use fixed smoothed epsilon
            epsilon = epsilon_bar;
        }

        // --- Store ALL Samples (single-phase: no warmup exclusion) ---
        Eigen::VectorXd constrained_theta = parameterManager.applyConstraints(theta_m);
        result.samples.push_back(constrained_theta);
        
        double final_obj = objectiveFunction.calculate(constrained_theta);
        result.sampleObjectiveValues.push_back(final_obj);
        
        if (final_obj > result.bestObjectiveValue) {
            result.bestObjectiveValue = final_obj;
            result.bestParameters = constrained_theta;
        }

        // Logging - more frequent for debugging
        if (m % 1 == 0 || m == num_iterations_) {
            std::ostringstream msg;
            msg << "Iteration " << m << "/" << num_iterations_;
            double avg_alpha = (n_alpha > 0) ? alpha / n_alpha : 0.0;
            msg << " | eps: " << std::scientific << std::setprecision(3) << epsilon
                << " | alpha: " << std::fixed << std::setprecision(3) << avg_alpha
                << " | tree_depth: " << j
                << " | best_logp: " << std::fixed << std::setprecision(4) << result.bestObjectiveValue;
            if (m <= adaptation_window_) {
                msg << " [adapting]";
            }
            logger.info("NUTSSampler", msg.str());
        }
    }

    logger.info("NUTSSampler", "Single-phase calibration finished. Total samples: " + 
                std::to_string(result.samples.size()));
    return result;
}

// --- Algorithm 4: FindReasonableEpsilon ---
// Simplified version that uses a heuristic based on parameter scales
// to avoid expensive gradient evaluations during initialization
double NUTSSampler::findReasonableEpsilon(
    IGradientObjectiveFunction& objective,
    const Eigen::VectorXd& theta,
    IParameterManager& parameterManager) const {
    
    Logger& logger = Logger::getInstance();
    
    // Start with a small epsilon based on parameter scales
    // Use proposal sigmas as a proxy for reasonable step sizes
    double avg_scale = 0.0;
    for (int i = 0; i < theta.size(); ++i) {
        double sigma = parameterManager.getSigmaForParamIndex(i);
        avg_scale += sigma;
    }
    avg_scale /= theta.size();
    
    // Initial epsilon as a fraction of average parameter scale
    double epsilon = avg_scale * 0.1;
    
    // Clamp to reasonable range
    epsilon = std::max(1e-6, std::min(epsilon, 0.1));
    
    logger.info("NUTSSampler", "Using heuristic initial epsilon based on parameter scales: " + std::to_string(epsilon));
    
    // Optional: Do a quick acceptance probability check with limited iterations
    std::normal_distribution<> normal(0.0, 1.0);
    Eigen::VectorXd r = Eigen::VectorXd::NullaryExpr(theta.size(), [&](){ return normal(rng_); });
    
    Eigen::VectorXd grad(theta.size());
    double log_p = objective.evaluate_with_gradient(theta, grad);
    
    if (!std::isfinite(log_p)) {
        logger.warning("NUTSSampler", "Initial log probability is non-finite, using default epsilon");
        return epsilon;
    }
    
    double H0 = log_p - 0.5 * r.dot(r);
    
    // Take one leapfrog step to test
    Eigen::VectorXd theta_prime = theta;
    Eigen::VectorXd r_prime = r;
    leapfrog(objective, theta_prime, r_prime, epsilon, parameterManager);
    
    double log_p_prime = objective.evaluate_with_gradient(theta_prime, grad);
    double H_prime = log_p_prime - 0.5 * r_prime.dot(r_prime);
    
    double accept_prob = std::exp(std::min(0.0, H_prime - H0));
    logger.info("NUTSSampler", "Initial acceptance probability: " + std::to_string(accept_prob));
    
    // Simple adjustment: if acceptance is too low, reduce epsilon; if too high, increase
    // Limited to just a few iterations to avoid getting stuck
    int max_adjust_iters = 5;
    for (int iter = 0; iter < max_adjust_iters; ++iter) {
        if (accept_prob < 0.1 && epsilon > 1e-8) {
            epsilon *= 0.5;
        } else if (accept_prob > 0.9 && epsilon < 1.0) {
            epsilon *= 1.5;
        } else {
            break;  // Acceptable range
        }
        
        theta_prime = theta;
        r_prime = r;
        leapfrog(objective, theta_prime, r_prime, epsilon, parameterManager);
        
        log_p_prime = objective.evaluate_with_gradient(theta_prime, grad);
        if (!std::isfinite(log_p_prime)) {
            epsilon *= 0.5;  // Reduce if we hit numerical issues
            continue;
        }
        
        H_prime = log_p_prime - 0.5 * r_prime.dot(r_prime);
        accept_prob = std::exp(std::min(0.0, H_prime - H0));
    }
    
    logger.info("NUTSSampler", "Final epsilon after adjustment: " + std::to_string(epsilon) + 
                " (accept_prob: " + std::to_string(accept_prob) + ")");
    
    return epsilon;
}

// Standard Leapfrog Integrator with gradient scaling
void NUTSSampler::leapfrog(
    IGradientObjectiveFunction& objective,
    Eigen::VectorXd& theta,
    Eigen::VectorXd& r,
    double epsilon,
    IParameterManager& parameterManager) const {
    
    // Maximum gradient norm to prevent numerical instability
    static constexpr double MAX_GRAD_NORM = 1000.0;
    
    // 1. Half step for momentum
    Eigen::VectorXd grad(theta.size());
    objective.evaluate_with_gradient(theta, grad);
    
    // Scale gradient if too large
    double grad_norm = grad.norm();
    if (grad_norm > MAX_GRAD_NORM) {
        grad *= MAX_GRAD_NORM / grad_norm;
    }
    
    r += 0.5 * epsilon * grad;
    
    // 2. Full step for position
    theta += epsilon * r;
    
    // Apply constraints (e.g., non-negativity)
    theta = parameterManager.applyConstraints(theta);
    
    // 3. Half step for momentum using gradient at new position
    objective.evaluate_with_gradient(theta, grad);
    
    // Scale gradient if too large
    grad_norm = grad.norm();
    if (grad_norm > MAX_GRAD_NORM) {
        grad *= MAX_GRAD_NORM / grad_norm;
    }
    
    r += 0.5 * epsilon * grad;
}

// --- Algorithm 6: BuildTree ---
void NUTSSampler::buildTree(
    IGradientObjectiveFunction& objective,
    const Eigen::VectorXd& theta,
    const Eigen::VectorXd& r,
    double log_u_slice,
    int v,
    int j,
    double epsilon,
    double H0, // Initial Hamiltonian of the iteration (for alpha calc)
    IParameterManager& parameterManager,
    Tree& tree) const {
    
    if (j == 0) {
        // Base case: Take one leapfrog step
        Eigen::VectorXd theta_prime = theta;
        Eigen::VectorXd r_prime = r;
        
        // v * epsilon (direction * step size)
        leapfrog(objective, theta_prime, r_prime, v * epsilon, parameterManager);
        
        Eigen::VectorXd grad(theta_prime.size());
        double log_p = objective.evaluate_with_gradient(theta_prime, grad);
        double H_prime = log_p - 0.5 * r_prime.dot(r_prime);
        
        // n' = I[u <= exp(H')] => I[log_u <= H']
        tree.n_valid = (log_u_slice <= H_prime) ? 1 : 0;
        
        // s' = I[log_u < H' + Delta_max]
        tree.s = (log_u_slice < H_prime + DELTA_MAX);
        
        // Set leaf values
        tree.theta_minus = theta_prime;
        tree.theta_plus = theta_prime;
        tree.r_minus = r_prime;
        tree.r_plus = r_prime;
        tree.theta_prime = theta_prime;
        
        // alpha' = min(1, exp(H' - H0))
        // Note: H = log_prob - kinetic. 
        // If H' is much smaller than H0 (energy lost/prob decreased), alpha < 1.
        tree.alpha = std::min(1.0, std::exp(H_prime - H0));
        tree.n_alpha = 1;
        
    } else {
        // Recursion
        Tree left_subtree;
        // Implicitly build left/right subtrees (Algorithm 6 recursion logic)
        buildTree(objective, theta, r, log_u_slice, v, j - 1, epsilon, H0, parameterManager, left_subtree);
        
        if (left_subtree.s) {
            Tree right_subtree;
            if (v == -1) {
                // Extending backward from theta_minus
                buildTree(objective, left_subtree.theta_minus, left_subtree.r_minus, log_u_slice, v, j - 1, epsilon, H0, parameterManager, right_subtree);
                
                // Merging: The "minus" comes from the new extension (right_subtree in code logic, but conceptually further left)
                tree.theta_minus = right_subtree.theta_minus;
                tree.r_minus = right_subtree.r_minus;
                tree.theta_plus = left_subtree.theta_plus;
                tree.r_plus = left_subtree.r_plus;
            } else {
                // Extending forward from theta_plus
                buildTree(objective, left_subtree.theta_plus, left_subtree.r_plus, log_u_slice, v, j - 1, epsilon, H0, parameterManager, right_subtree);
                
                // Merging: The "plus" comes from the new extension
                tree.theta_minus = left_subtree.theta_minus;
                tree.r_minus = left_subtree.r_minus;
                tree.theta_plus = right_subtree.theta_plus;
                tree.r_plus = right_subtree.r_plus;
            }
            
            if (right_subtree.s) {
                // Both subtrees valid. Merge logic.
                
                // 1. Update n'
                tree.n_valid = left_subtree.n_valid + right_subtree.n_valid;
                
                // 2. Select theta' with bias
                double prob = (tree.n_valid > 0) 
                    ? static_cast<double>(right_subtree.n_valid) / static_cast<double>(tree.n_valid) 
                    : 0.0;
                
                if (std::uniform_real_distribution<>(0.0, 1.0)(rng_) < prob) {
                    tree.theta_prime = right_subtree.theta_prime;
                } else {
                    tree.theta_prime = left_subtree.theta_prime;
                }
                
                // 3. Update alpha and n_alpha
                tree.alpha = left_subtree.alpha + right_subtree.alpha;
                tree.n_alpha = left_subtree.n_alpha + right_subtree.n_alpha;
                
                // 4. Update s' (Stop condition)
                // s' = s'_left * s'_right * NoUTurn(Total Tree)
                bool no_uturn = checkNoUTurn(tree.theta_minus, tree.theta_plus, tree.r_minus, tree.r_plus);
                tree.s = left_subtree.s && right_subtree.s && no_uturn;
                
            } else {
                // Right subtree invalid, propagate failure
                tree.theta_prime = left_subtree.theta_prime; // Keep valid sample just in case
                tree.n_valid = left_subtree.n_valid;
                tree.s = false;
                tree.alpha = left_subtree.alpha; // approximate return
                tree.n_alpha = left_subtree.n_alpha;
            }
        } else {
            // Left subtree invalid, return it
            tree = left_subtree;
        }
    }
}

// Equation 9: Checks if trajectory is doubling back
bool NUTSSampler::checkNoUTurn(
    const Eigen::VectorXd& theta_minus,
    const Eigen::VectorXd& theta_plus,
    const Eigen::VectorXd& r_minus,
    const Eigen::VectorXd& r_plus) const {
    
    Eigen::VectorXd delta_theta = theta_plus - theta_minus;
    
    // Stop if distance between ends is decreasing
    // (theta+ - theta-) . r- >= 0  AND  (theta+ - theta-) . r+ >= 0
    double dot_minus = delta_theta.dot(r_minus);
    double dot_plus = delta_theta.dot(r_plus);
    
    return (dot_minus >= 0) && (dot_plus >= 0);
}

} // namespace epidemic