#ifndef PARTICLE_SWARM_OPTIMIZER_HPP
#define PARTICLE_SWARM_OPTIMIZER_HPP

#include "sir_age_structured/interfaces/IOptimizationAlgorithm.hpp"
#include "sir_age_structured/interfaces/IObjectiveFunction.hpp"
#include "sir_age_structured/interfaces/IParameterManager.hpp"
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <limits>
#include <string>
#include <stdexcept>


namespace epidemic {

/**
 * @brief Particle Swarm Optimization (PSO) implementation for maximizing an objective function.
 *
 * This class implements an enhanced Particle Swarm Optimization (PSO) algorithm.
 * Its design and features are informed by established PSO principles and improvements
 * as reviewed in "Particle Swarm Optimization Algorithm and Its Applications:
 * A Systematic Review" by Ahmed G. Gad (2022), Archives of Computational
 * Methods in Engineering, 29, 2531–2561, and "Zhan, Zh., Zhang, J. (2008). Adaptive Particle Swarm Optimization.
 * In: Dorigo, M., Birattari, M., Blum, C., Clerc, M., Stützle, T., Winfield, A.F.T. (eds) Ant Colony Optimization and Swarm Intelligence. 
 * ANTS 2008. Lecture Notes in Computer Science, vol 5217. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-540-87527-7_21"
 *
 * The implementation leverages the application's `Logger` for all console and file output,
 * allowing for configurable log levels and detailed tracing of the optimization process.
 *
 * @b Key Features and Enhancements:
 * - @b Core @b PSO @b Mechanics: Utilizes the standard velocity and position update rules.
 * - @b Linearly @b Decreasing @b Inertia @b Weight @b ($\omega$): Allows the inertia weight to
 *   decrease linearly over iterations. This is a common technique to balance global
 *   exploration in early stages with local exploitation in later stages.
 * - @b Time-Varying @b Acceleration @b Coefficients @b (TVAC): Supports linearly changing
 *   cognitive ($c_1$) and social ($c_2$) coefficients. Typically, $c_1$
 *   decreases (reducing self-reliance) and $c_2$ increases (increasing swarm
 *   influence) over time.
 * - @b Optional @b Local @b Best @b (LBest) @b Swarm @b Topology: Provides an option to use a
 *   ring topology where particles are attracted to their local best neighbor,
 *   instead of solely the global best. This can improve swarm diversity and
 *   mitigate premature convergence.
 * - @b Optional @b Constriction @b Factor: Implements the constriction factor method by
 *   Clerc and Kennedy for velocity control. This offers an alternative to the
 *   inertia weight and explicit velocity clamping to help ensure convergence.
 * - @b Uniform @b Initialization: Particle positions and velocities are initialized
 *   uniformly within parameter bounds defined by the `IParameterManager`.
 * - @b Velocity @b Clamping @b and @b Boundary @b Handling: Particle velocities are constrained
 *   to a maximum value ($V_{max}$) to prevent divergence, and positions are kept
 *   within feasible parameter bounds. These are standard PSO practices.
 *
 * The optimizer is designed for @b maximization tasks.
 *
 * @note The `IParameterManager` interface must provide:
 *   - `getParameterCount()`: The number of dimensions (parameters) in the search space.
 *   - `getLowerBoundForParamIndex(int)`: The lower bound for a specific parameter dimension.
 *   - `getUpperBoundForParamIndex(int)`: The upper bound for a specific parameter dimension.
 *
 * @implements IOptimizationAlgorithm
 */
class ParticleSwarmOptimizer : public IOptimizationAlgorithm {
public:
    /**
     * @brief Virtual destructor.
     */
    virtual ~ParticleSwarmOptimizer();

    /**
     * @brief Default constructor. Initializes PSO parameters to their default values.
     *
     * @b Default @b Parameter @b Values:
     *   - iterations: 100
     *   - swarm_size: 30
     *   - omega_start: 0.9
     *   - omega_end: 0.4
     *   - c1_initial: 1.5 (or value from deprecated 'c1')
     *   - c1_final: 1.5   (or value from deprecated 'c1')
     *   - c2_initial: 1.5 (or value from deprecated 'c2')
     *   - c2_final: 1.5   (or value from deprecated 'c2')
     *   - report_interval: 10
     *   - use_lbest_topology: false
     *   - lbest_neighborhood_k: 1
     *   - use_constriction_factor: false
     *   - log_new_gbest: true
     *   - log_particle_details: false
     *   - particle_log_interval: 10
     *   - particles_to_log_count: 3
     */
    ParticleSwarmOptimizer() = default;

    /**
     * @brief Configures the PSO algorithm parameters from a map of settings.
     *
     * This method allows customization of the PSO algorithm's behavior.
     * Invalid settings will result in an `std::invalid_argument` exception
     * and an error message logged via the application's `Logger`.
     *
     * @b Configurable @b Parameters @b (key: @b string, @b value: @b double):
     *   - `"iterations"` (int): Total number of optimization iterations. Must be positive. (Default: 100)
     *   - `"swarm_size"` (int): Number of particles in the swarm. Must be positive. (Default: 30)
     *   - `"omega_start"` (double): Initial value of the inertia weight $\omega$. Used if constriction factor is disabled. Must be non-negative. (Default: 0.9)
     *   - `"omega_end"` (double): Final value of the inertia weight $\omega$. Used if constriction factor is disabled. Must be non-negative. (Default: 0.4)
     *   - `"c1_initial"` (double): Initial value for the cognitive (personal best) acceleration coefficient $c_1$. Must be non-negative. (Default: 1.5)
     *   - `"c1_final"` (double): Final value for the cognitive coefficient $c_1$. Must be non-negative. (Default: 1.5)
     *   - `"c2_initial"` (double): Initial value for the social (global/local best) acceleration coefficient $c_2$. Must be non-negative. (Default: 1.5)
     *   - `"c2_final"` (double): Final value for the social coefficient $c_2$. Must be non-negative. (Default: 1.5)
     *   - `"report_interval"` (int): Iteration interval for logging general progress reports. Must be positive. (Default: 10)
     *   - `"use_lbest_topology"` (bool, 0.0 for false, non-zero for true): If true, particles are attracted to their local best within a ring neighborhood. Otherwise, attracted to the global best. (Default: false)
     *   - `"lbest_neighborhood_k"` (int): If LBest topology is used, this defines the number of neighbors on each side of a particle (e.g., k=1 means a total neighborhood of 2k+1 = 3 particles: self + 1 left + 1 right). Must be non-negative. (Default: 1)
     *   - `"use_constriction_factor"` (bool, 0.0 for false, non-zero for true): If true, uses the constriction factor method for velocity updates, ignoring inertia weight. (Default: false)
     *   - `"log_new_gbest"` (bool, 0.0 for false, non-zero for true): If true, logs a message every time a new global best solution is found. (Default: true)
     *   - `"log_particle_details"` (bool, 0.0 for false, non-zero for true): If true, logs detailed information (position, velocity, fitness) for a subset of particles at specified intervals. (Default: false)
     *   - `"particle_log_interval"` (int): Iteration interval for logging particle details, if `log_particle_details` is true. Must be positive. (Default: 10)
     *   - `"particles_to_log_count"` (int): Number of particles (from the start of the swarm vector) for which to log details, if `log_particle_details` is true. Must be non-negative. (Default: 3)
     *
     * @note @b Deprecated @b Parameters:
     *   - `"c1"` (double): If provided, sets both `c1_initial` and `c1_final`. A warning will be logged.
     *   - `"c2"` (double): If provided, sets both `c2_initial` and `c2_final`. A warning will be logged.
     *   - `"omega"` (double): Ignored. Use `omega_start` and `omega_end`. A warning will be logged.
     *
     * @param settings A map where keys are parameter names and values are their settings.
     * @throws std::invalid_argument If any parameter value is outside its valid range or type.
     */
    void configure(const std::map<std::string,double>& settings) override;

    /**
     * @brief Runs the Particle Swarm Optimization algorithm to find the parameters that maximize the given objective function.
     *
     * The optimization process involves:
     * 1. Initializing a swarm of particles with random positions and velocities within the bounds defined by `parameterManager`.
     * 2. Iteratively updating each particle's velocity and position based on its personal best experience (`pbest`) and the best experience of its neighborhood (either global best `gbest` or local best `lbest`).
     * 3. Evaluating the objective function for each particle's new position.
     * 4. Updating `pbest` and `gbest` (or `lbest`) if better solutions are found.
     * 5. Repeating until the maximum number of iterations is reached.
     * All significant events and progress are logged using the application's `Logger`.
     *
     * @param initialParameters If provided and matches the parameter count, these are used as the initial position 
     *                          for the first particle in the swarm. Otherwise, or if the size mismatches, 
     *                          all particles are initialized uniformly (or the initial guess is ignored for the first particle
     *                          if size mismatches but is non-zero)
     * @param objectiveFunction An object implementing the `IObjectiveFunction` interface, which calculates the fitness of a given set of parameters.
     * @param parameterManager An object implementing the `IParameterManager` interface, providing parameter dimensionality and bounds.
     * @return An `OptimizationResult` struct containing the best set of parameters found (`bestParameters`) and their corresponding objective function value (`bestObjectiveValue`).
     */
    OptimizationResult optimize(
        const Eigen::VectorXd& initialParameters,
        IObjectiveFunction& objectiveFunction,
        IParameterManager& parameterManager) override;

private:
    // PSO hyperparameters
    /** @brief Total number of optimization iterations. */
    int iterations_ = 100;
    /** @brief Number of particles in the swarm. */
    int swarm_size_ = 30;
    /** @brief Initial value of the inertia weight omega. */
    double omega_start_ = 0.9;
    /** @brief Final value of the inertia weight omega. */
    double omega_end_ = 0.4;

    /** @brief Initial value for the cognitive acceleration coefficient c1. */
    double c1_initial_ = 1.5;
    /** @brief Final value for the cognitive acceleration coefficient c1. */
    double c1_final_   = 1.5;
    /** @brief Initial value for the social acceleration coefficient c2. */
    double c2_initial_ = 1.5;
    /** @brief Final value for the social acceleration coefficient c2. */
    double c2_final_   = 1.5;

    /** @brief Iteration interval for logging general progress reports. */
    int report_interval_ = 10;

    /** @brief Flag to use LBest (local best) topology instead of GBest (global best). */
    bool use_lbest_topology_ = false;
    /** @brief Neighborhood size (k) for LBest topology (k neighbors on each side). */
    int lbest_neighborhood_k_ = 1;

    /** @brief Flag to use the constriction factor method for velocity updates. */
    bool use_constriction_factor_ = false;

    // Logging control members
    /** @brief Flag to enable logging when a new global best is found. */
    bool log_new_gbest_ = true;
    /** @brief Flag to enable detailed logging for a subset of particles. */
    bool log_particle_details_ = false;
    /** @brief Iteration interval for logging particle details if enabled. */
    int particle_log_interval_ = 10;
    /** @brief Number of particles for which to log details if enabled. */
    int particles_to_log_count_ = 3;

    /** @brief Identifier string used for logging messages from this class. */
    const std::string logger_source_id_ = "ParticleSwarmOptimizer";

    /** @brief Internal structure to represent each particle in the swarm. */
    struct Particle {
        /** @brief Current position of the particle in the search space. */
        Eigen::VectorXd position;
        /** @brief Current velocity of the particle. */
        Eigen::VectorXd velocity;
        /** @brief Personal best position found by this particle so far. */
        Eigen::VectorXd pbest_position;
        /** @brief Objective function value at the personal best position. */
        double pbest_value = -std::numeric_limits<double>::infinity(); // For maximization
    };

    /**
     * @brief Initializes the swarm of particles.
     *
     * Sets random initial positions and velocities for each particle within the
     * bounds defined by the `parameterManager`. Optionally uses a provided initial
     * parameter vector for the first particle.
     *
     * @param swarm The vector of particles to be initialized (output).
     * @param objectiveFunction The objective function to evaluate particle positions.
     * @param parameterManager The parameter manager providing bounds for initialization.
     * @param gbest_position The global best position found after initialization (output).
     * @param gbest_value The global best objective function value after initialization (output).
     * @param initial_params Optional initial parameter vector to use for first particle.
     */
    void initializeSwarm(
        std::vector<Particle>& swarm,
        IObjectiveFunction& objectiveFunction,
        IParameterManager& parameterManager,
        Eigen::VectorXd& gbest_position,
        double& gbest_value,
        const Eigen::VectorXd* initial_params = nullptr);

    /**
     * @brief Finds the best position in a particle's local neighborhood (LBest topology).
     *
     * For a given particle, this method identifies the particle with the best
     * personal best (`pbest_value`) within its ring neighborhood of size `k`.
     * The neighborhood includes `k` particles to the left and `k` particles to
     * the right in the swarm array (with wrap-around).
     *
     * @param swarm The current swarm of particles.
     * @param particle_idx The index of the particle for which to find the local best.
     * @param neighborhood_k The number of neighbors on each side to consider (k).
     * @return The personal best position of the best particle in the local neighborhood.
     */
    Eigen::VectorXd get_lbest_position_for_particle(
        const std::vector<Particle>& swarm,
        int particle_idx,
        int neighborhood_k);

    /**
     * @brief Helper function to convert an Eigen::VectorXd to a string representation.
     *
     * Useful for logging vector contents. The format is "[v1, v2, ..., vn]".
     *
     * @param vec The vector to convert.
     * @return A string representation of the vector.
     */
    std::string vector_to_string(const Eigen::VectorXd& vec);
};

} // namespace epidemic

#endif // PARTICLE_SWARM_OPTIMIZER_HPP