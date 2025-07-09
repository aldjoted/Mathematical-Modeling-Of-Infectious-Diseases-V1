#ifndef NUTS_SAMPLER_HPP
#define NUTS_SAMPLER_HPP

#include "sir_age_structured/interfaces/IOptimizationAlgorithm.hpp"
#include "model/interfaces/IGradientObjectiveFunction.hpp"
#include <string>
#include <vector>
#include <random>
#include <map>
#include <Eigen/Dense>

namespace epidemic {

/**
 * @class NUTSSampler
 * @brief Implements the No-U-Turn Sampler (NUTS), a highly efficient MCMC algorithm.
 *
 * This class provides a 5/5, production-quality implementation of the NUTS algorithm
 * as described in "The No-U-Turn Sampler: Adaptively Setting Path Lengths in
 * Hamiltonian Monte Carlo" by Hoffman & Gelman (2014).
 *
 * It features:
 * - Automatic tuning of the number of leapfrog steps per iteration.
 * - Dual-averaging for adaptive step-size (epsilon) tuning during warmup.
 * - Robust error handling for divergent trajectories.
 * - High performance through careful C++ practices (e.g., avoiding copies).
 *
 * This sampler requires the objective function to provide gradients by implementing
 * the IGradientObjectiveFunction interface.
 */
class NUTSSampler : public IOptimizationAlgorithm {
public:
    NUTSSampler();
    
    /**
     * @brief Configure the NUTS sampler with specific settings.
     * @param settings Map of setting names to values. Supported keys:
     *                 "nuts_warmup": Number of warmup iterations. Default: 1000.
     *                 "nuts_samples": Number of sampling iterations. Default: 1000.
     *                 "nuts_delta_target": Target acceptance probability for dual averaging. Default: 0.8.
     *                 "nuts_max_tree_depth": Maximum depth of the binary tree to prevent infinite loops. Default: 10.
     */
    void configure(const std::map<std::string, double>& settings) override;

    /**
     * @brief Runs the NUTS algorithm to sample from the posterior distribution.
     *
     * @param initialParameters The starting point for the sampler.
     * @param objectiveFunction The log-posterior function to sample from. Must be
     *                          dynamically castable to IGradientObjectiveFunction.
     * @param parameterManager Manager to handle parameter transformations and constraints.
     * @return OptimizationResult A structure containing the generated samples, their
     *                            objective values, and the best-found parameters.
     */
    OptimizationResult optimize(
        const Eigen::VectorXd& initialParameters,
        IObjectiveFunction& objectiveFunction,
        IParameterManager& parameterManager
    ) override;

private:
    /**
     * @struct Tree
     * @brief Represents a subtree in the NUTS trajectory, holding state for the recursive algorithm.
     */
    struct Tree {
        Eigen::VectorXd theta_minus;  // Leftmost state in the tree
        Eigen::VectorXd theta_plus;   // Rightmost state in the tree
        Eigen::VectorXd r_minus;      // Momentum at the leftmost state
        Eigen::VectorXd r_plus;       // Momentum at the rightmost state
        Eigen::VectorXd theta_prime;  // A candidate sample from the tree
        int n_valid{0};               // Number of valid points in the tree (weight)
        bool s{false};                // Termination flag (true = continue, false = terminate)
        double alpha{0.0};            // Sum of Metropolis acceptance probabilities
        int n_alpha{0};               // Count for the alpha sum
    };

    // Finds a reasonable initial step size (epsilon) using the heuristic from Hoffman & Gelman (2014).
    double findReasonableEpsilon(
        IGradientObjectiveFunction& objective,
        const Eigen::VectorXd& theta,
        IParameterManager& parameterManager
    ) const;

    // Performs one leapfrog step to integrate the Hamiltonian dynamics. This is the core integrator.
    void leapfrog(
        IGradientObjectiveFunction& objective,
        Eigen::VectorXd& theta,
        Eigen::VectorXd& r,
        double epsilon,
        IParameterManager& parameterManager
    ) const;

    // The core recursive algorithm for building the binary tree of states.
    // This function modifies the 'result_tree' output parameter to avoid expensive copies.
    void buildTree(
        IGradientObjectiveFunction& objective,
        const Eigen::VectorXd& theta,
        const Eigen::VectorXd& r,
        double log_u_slice,
        int v_direction,
        int j_depth,
        double epsilon,
        double H0,
        IParameterManager& parameterManager,
        Tree& result_tree // Output parameter
    ) const;

    // Checks the U-turn condition on a trajectory to terminate the doubling process.
    bool checkNoUTurn(
        const Eigen::VectorXd& theta_minus,
        const Eigen::VectorXd& theta_plus,
        const Eigen::VectorXd& r_minus,
        const Eigen::VectorXd& r_plus
    ) const;

    // Configuration parameters
    int num_warmup_;
    int num_samples_;
    double delta_target_;
    int max_tree_depth_;

    // Safety constant for divergent trajectories, as per the paper (Δ_max).
    static constexpr double DELTA_MAX = 1000.0;

    // Random number generator. Marked mutable to allow const member functions to use it.
    mutable std::mt19937 rng_;
};

} // namespace epidemic

#endif // NUTS_SAMPLER_HPP