#ifndef PARTICLE_SWARM_OPTIMIZER_HPP
#define PARTICLE_SWARM_OPTIMIZER_HPP

#include "sir_age_structured/interfaces/IOptimizationAlgorithm.hpp"
#include "sir_age_structured/interfaces/IObjectiveFunction.hpp"
#include "sir_age_structured/interfaces/IParameterManager.hpp"
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <memory>
#include <omp.h>

namespace epidemic {

/**
 * @brief Particle Swarm Optimization algorithm with enhancements
 * 
 * This implementation incorporates multiple PSO improvements based on recent
 * research in swarm intelligence. The algorithm adaptively switches between exploration
 * and exploitation phases, maintains population diversity, and includes multiple strategies
 * for escaping local optima.
 * 
 * Key features:
 * - Adaptive PSO (APSO) with Evolutionary State Estimation (Zhan et al., 2008)
 * - Quantum-behaved PSO for enhanced global search (Sun et al., 2004)
 * - Lévy flight mechanism for long-range exploration (Zhang et al., 2018)
 * - Opposition-based learning for intelligent initialization (Tizhoosh, 2005)
 * - Multiple swarm topologies for different problem characteristics
 * - Automatic restart strategies with elite preservation
 * - Parallel computation support via OpenMP
 * 
 * The algorithm is designed for continuous optimization problems and automatically
 * adapts its behavior based on the search progress and swarm diversity.
 * 
 * @implements IOptimizationAlgorithm
 */
class ParticleSwarmOptimization : public IOptimizationAlgorithm {
public:
    /**
     * @brief PSO variant types available for optimization
     * 
     * Different variants are suited for different problem types:
     * - STANDARD: Best for smooth, unimodal problems
     * - QUANTUM: Excellent for multimodal problems with many local optima
     * - ADAPTIVE: Automatically adjusts strategy based on search progress
     * - LEVY_FLIGHT: Good for problems requiring long-range exploration
     * - HYBRID: Combines multiple strategies dynamically
     */
    enum class PSOVariant {
        STANDARD,     ///< Classical PSO with inertia weight and acceleration coefficients
        QUANTUM,      ///< Quantum-behaved PSO using wave function collapse
        ADAPTIVE,     ///< PSO with evolutionary state estimation and adaptive parameters
        LEVY_FLIGHT,  ///< PSO enhanced with Lévy flight for escaping local optima
        HYBRID        ///< Dynamically switches between strategies based on particle success
    };

    /**
     * @brief Evolutionary states for adaptive parameter control
     * 
     * The algorithm estimates which phase the swarm is in and adjusts
     * parameters accordingly for optimal performance.
     */
    enum class EvolutionaryState {
        EXPLORATION,   ///< Particles are scattered, exploring the search space broadly
        EXPLOITATION,  ///< Particles are converging toward promising regions
        CONVERGENCE,   ///< Particles are fine-tuning near the optimum
        JUMPING_OUT    ///< Particles need to escape from local optima
    };

    /**
     * @brief Swarm topology types for particle communication
     * 
     * Different topologies affect information flow and convergence speed:
     * - GLOBAL_BEST: Fast convergence but prone to premature convergence
     * - LOCAL_BEST: Slower but more thorough exploration
     * - VON_NEUMANN: Good balance between exploration and exploitation
     * - RANDOM_DYNAMIC: Prevents stagnation through changing connections
     */
    enum class TopologyType {
        GLOBAL_BEST,     ///< Fully connected - all particles share information
        LOCAL_BEST,      ///< Ring topology - particles communicate with k neighbors
        VON_NEUMANN,     ///< Grid topology - 2D lattice communication structure
        RANDOM_DYNAMIC   ///< Dynamic random connections that change each iteration
    };

    /**
     * @brief Default constructor
     * 
     * Initializes the PSO algorithm with default parameters optimized for
     * general continuous optimization problems. Uses adaptive variant with
     * global topology as default configuration.
     */
    ParticleSwarmOptimization() = default;

    /**
     * @brief Virtual destructor
     * 
     * Ensures proper cleanup of resources in derived classes.
     */
    virtual ~ParticleSwarmOptimization() = default;

    /**
     * @brief Configure the PSO algorithm with custom parameters
     * 
     * Allows fine-tuning of all algorithm parameters through a key-value map.
     * Invalid parameters will throw std::invalid_argument with descriptive messages.
     * 
     * @param settings Map of configuration parameters:
     *   Standard PSO parameters:
     *   - "iterations": Maximum number of iterations (default: 100)
     *   - "swarm_size": Number of particles in the swarm (default: 30)
     *   - "omega_start": Initial inertia weight (default: 0.9)
     *   - "omega_end": Final inertia weight (default: 0.4)
     *   - "c1_initial": Initial cognitive coefficient (default: 2.5)
     *   - "c1_final": Final cognitive coefficient (default: 0.5)
     *   - "c2_initial": Initial social coefficient (default: 0.5)
     *   - "c2_final": Final social coefficient (default: 2.5)
     *   - "report_interval": Iterations between progress reports (default: 10)
     *   
     *   Advanced parameters:
     *   - "variant": PSO variant to use, 0-4 (default: 2 for ADAPTIVE)
     *   - "topology": Topology type, 0-3 (default: 0 for GLOBAL_BEST)
     *   - "use_opposition_learning": Enable OBL initialization, 0/1 (default: 1)
     *   - "use_parallel": Enable parallel computation, 0/1 (default: 1)
     *   - "use_adaptive_parameters": Enable adaptive control, 0/1 (default: 1)
     *   - "diversity_threshold": Minimum diversity before action (default: 0.1)
     *   - "restart_threshold": Improvement threshold for restart (default: 1e-6)
     *   - "quantum_beta": Contraction-expansion coefficient for QPSO (default: 1.0)
     *   - "levy_alpha": Stability parameter for Lévy flights (default: 1.5)
     *   - "max_stagnation": Iterations without improvement before restart (default: 20)
     *   - "log_evolutionary_state": Log state transitions, 0/1 (default: 1)
     * 
     * @throws std::invalid_argument If any parameter value is invalid
     */
    void configure(const std::map<std::string, double>& settings) override;

    /**
     * @brief Execute the PSO optimization algorithm
     * 
     * Runs the complete optimization process, adaptively adjusting parameters
     * and strategies based on search progress. Supports warm starts through
     * initial parameters.
     * 
     * @param initialParameters Optional starting point for optimization. If provided
     *                         and has correct dimension, the first particle will be
     *                         initialized at this position. Can be empty for random start.
     * @param objectiveFunction The function to maximize. Will be called many times
     *                         during optimization, so should be efficient.
     * @param parameterManager Provides parameter bounds and constraints. All generated
     *                        positions will respect these bounds.
     * 
     * @return OptimizationResult containing:
     *         - bestParameters: The optimized parameter vector
     *         - bestObjectiveValue: The maximum objective value found
     * 
     * @throws std::runtime_error If optimization fails due to numerical issues
     */
    OptimizationResult optimize(
        const Eigen::VectorXd& initialParameters,
        IObjectiveFunction& objectiveFunction,
        IParameterManager& parameterManager) override;

private:
    // ===== Core PSO Parameters =====
    
    /// @brief Maximum number of iterations for the optimization process
    int iterations_ = 100;
    
    /// @brief Number of particles in the swarm (population size)
    int swarm_size_ = 30;
    
    /// @brief Initial inertia weight for velocity update (exploration emphasis)
    double omega_start_ = 0.9;
    
    /// @brief Final inertia weight for velocity update (exploitation emphasis)
    double omega_end_ = 0.4;
    
    /// @brief Initial cognitive acceleration coefficient (personal best influence)
    double c1_initial_ = 2.5;
    
    /// @brief Final cognitive acceleration coefficient
    double c1_final_ = 0.5;
    
    /// @brief Initial social acceleration coefficient (global best influence)
    double c2_initial_ = 0.5;
    
    /// @brief Final social acceleration coefficient
    double c2_final_ = 2.5;
    
    /// @brief Number of iterations between progress reports
    int report_interval_ = 10;
    
    // ===== Advanced Algorithm Parameters =====
    
    /// @brief PSO variant to use for optimization
    PSOVariant variant_ = PSOVariant::ADAPTIVE;
    
    /// @brief Swarm topology for particle communication
    TopologyType topology_ = TopologyType::GLOBAL_BEST;
    
    /// @brief Enable opposition-based learning for intelligent initialization
    bool use_opposition_learning_ = true;
    
    /// @brief Enable parallel computation using OpenMP
    bool use_parallel_ = false;
    
    /// @brief Enable adaptive parameter control based on evolutionary state
    bool use_adaptive_parameters_ = true;
    
    /// @brief Minimum swarm diversity threshold (0-1) before triggering actions
    double diversity_threshold_ = 0.1;
    
    /// @brief Minimum improvement threshold to reset stagnation counter
    double restart_threshold_ = 1e-6;
    
    /// @brief Contraction-expansion coefficient for quantum PSO (typically 0.5-1.5)
    double quantum_beta_ = 1.0;
    
    /// @brief Stability parameter for Lévy distribution (typically 1.0-2.0)
    double levy_alpha_ = 1.5;
    
    /// @brief Current number of iterations without significant improvement
    int stagnation_counter_ = 0;
    
    /// @brief Maximum stagnation iterations before triggering restart
    int max_stagnation_ = 50;
    
    // ===== Logging Configuration =====
    
    /// @brief Whether to log evolutionary state transitions
    bool log_evolutionary_state_ = true;
    
    /// @brief Logger source identifier for this algorithm
    const std::string logger_source_id_ = "PSO";

    /**
     * @brief Internal particle representation with extended attributes
     * 
     * Each particle maintains its position, velocity, and performance history
     * to support adaptive strategies and success-based behavior switching.
     */
    struct particle {
        /// @brief Current position in the search space
        Eigen::VectorXd position;
        
        /// @brief Current velocity vector
        Eigen::VectorXd velocity;
        
        /// @brief Best position found by this particle
        Eigen::VectorXd pbest_position;
        
        /// @brief Objective value at personal best position
        double pbest_value = -std::numeric_limits<double>::infinity();
        
        /// @brief Current objective value at current position
        double current_fitness = -std::numeric_limits<double>::infinity();
        
        /// @brief Quantum position for QPSO variant
        Eigen::VectorXd quantum_position;
        
        /// @brief Success rate (0-1) for adaptive behavior in hybrid variant
        double success_rate = 0.0;
        
        /// @brief Number of successful updates (improvements)
        int success_count = 0;
        
        /// @brief Total number of position updates
        int total_updates = 0;
    };

    // ===== Core Optimization Methods =====

    /**
     * @brief Initialize the particle swarm with random or guided positions
     * 
     * Creates and evaluates initial particle positions using various strategies
     * including random initialization, opposition-based learning, and warm starts.
     * 
     * @param[out] swarm Vector of particles to be initialized
     * @param[in] objectiveFunction Function to evaluate particle fitness
     * @param[in] parameterManager Provides parameter bounds for initialization
     * @param[out] gbest_position Will contain the best position found during init
     * @param[out] gbest_value Will contain the best objective value found
     * @param[in] initial_params Optional starting position for first particle
     */
    void initializeSwarm(
        std::vector<particle>& swarm,
        IObjectiveFunction& objectiveFunction,
        IParameterManager& parameterManager,
        Eigen::VectorXd& gbest_position,
        double& gbest_value,
        const Eigen::VectorXd* initial_params = nullptr);

    /**
     * @brief Update all particle positions and velocities for one iteration
     * 
     * Applies the selected PSO variant update rules to all particles, handles
     * boundary constraints, and tracks particle success for adaptive strategies.
     * 
     * @param[in,out] swarm Vector of particles to update
     * @param[in] gbest_position Current global best position
     * @param[in] objectiveFunction Function to evaluate new positions
     * @param[in] parameterManager Provides parameter bounds for constraints
     * @param[in] iteration Current iteration number for adaptive parameters
     */
    void updateParticles(
        std::vector<particle>& swarm,
        const Eigen::VectorXd& gbest_position,
        IObjectiveFunction& objectiveFunction,
        IParameterManager& parameterManager,
        int iteration);

    // ===== Evolutionary State Estimation Methods =====

    /**
     * @brief Estimate the current evolutionary state of the swarm
     * 
     * Analyzes swarm distribution and fitness landscape to determine whether
     * particles are exploring, exploiting, converging, or need to escape.
     * Based on the APSO paper by Zhan et al., 2008.
     * 
     * @param[in] swarm Current particle swarm
     * @param[in] gbest_position Current global best position
     * @return Estimated evolutionary state
     */
    EvolutionaryState estimateEvolutionaryState(
        const std::vector<particle>& swarm,
        const Eigen::VectorXd& gbest_position);

    /**
     * @brief Calculate the evolutionary factor for state estimation
     * 
     * Computes a normalized factor (0-1) representing swarm convergence state.
     * Higher values indicate exploration, lower values indicate convergence.
     * 
     * @param[in] swarm Current particle swarm
     * @param[in] gbest_position Current global best position
     * @return Evolutionary factor in range [0, 1]
     */
    double calculateEvolutionaryFactor(
        const std::vector<particle>& swarm,
        const Eigen::VectorXd& gbest_position);

    // ===== Adaptive Control Methods =====

    /**
     * @brief Adaptively adjust PSO parameters based on evolutionary state
     * 
     * Modifies inertia weight and acceleration coefficients to match the
     * current search phase for optimal performance.
     * 
     * @param[in] state Current evolutionary state
     * @param[in] iteration Current iteration for time-based adjustments
     * @param[out] omega Adapted inertia weight
     * @param[out] c1 Adapted cognitive coefficient
     * @param[out] c2 Adapted social coefficient
     */
    void adaptParameters(
        EvolutionaryState state,
        int iteration,
        double& omega,
        double& c1,
        double& c2);

    // ===== Initialization Strategy Methods =====

    /**
     * @brief Apply opposition-based learning to improve initial population
     * 
     * Generates opposite positions for all particles and selects the best
     * combination of original and opposite particles for the initial swarm.
     * 
     * @param[in,out] swarm Particle swarm to be enhanced
     * @param[in] lb Lower bounds for all parameters
     * @param[in] ub Upper bounds for all parameters
     */
    void oppositionBasedInitialization(
        std::vector<particle>& swarm,
        const std::vector<double>& lb,
        const std::vector<double>& ub);

    // ===== PSO Variant Update Methods =====

    /**
     * @brief Standard PSO velocity and position update
     * 
     * Implements the classical PSO update equations with velocity clamping
     * and reflective boundary handling.
     * 
     * @param[in,out] p Particle to update
     * @param[in] gbest_position Global best position
     * @param[in] lbest_position Local best position (from neighborhood)
     * @param[in] omega Inertia weight
     * @param[in] c1 Cognitive acceleration coefficient
     * @param[in] c2 Social acceleration coefficient
     * @param[in] lb Lower bounds for all parameters
     * @param[in] ub Upper bounds for all parameters
     */
    void standardPSOUpdate(
        particle& p,
        const Eigen::VectorXd& lbest_position,
        double omega, double c1, double c2,
        const std::vector<double>& lb,
        const std::vector<double>& ub);

    /**
     * @brief Quantum-behaved PSO position update
     * 
     * Updates particle position using quantum mechanics principles with
     * wave function collapse around attractors. No velocity is used.
     * 
     * @param[in,out] p Particle to update
     * @param[in] gbest_position Global best position
     * @param[in] mean_best_position Mean of all personal best positions
     * @param[in] iteration Current iteration for beta calculation
     * @param[in] lb Lower bounds for all parameters
     * @param[in] ub Upper bounds for all parameters
     */
    void quantumPSOUpdate(
        particle& p,
        const Eigen::VectorXd& gbest_position,
        const Eigen::VectorXd& mean_best_position,
        int iteration,
        const std::vector<double>& lb,
        const std::vector<double>& ub);

    /**
     * @brief PSO update enhanced with Lévy flights
     * 
     * Performs standard PSO update followed by occasional Lévy flight
     * jumps for escaping local optima and long-range exploration.
     * 
     * @param[in,out] p Particle to update
     * @param[in] gbest_position Global best position
     * @param[in] omega Inertia weight
     * @param[in] c1 Cognitive acceleration coefficient
     * @param[in] c2 Social acceleration coefficient
     * @param[in] lb Lower bounds for all parameters
     * @param[in] ub Upper bounds for all parameters
     */
    void levyFlightUpdate(
        particle& p,
        const Eigen::VectorXd& gbest_position,
        double omega, double c1, double c2,
        const std::vector<double>& lb,
        const std::vector<double>& ub);

    // ===== Diversity and Restart Methods =====

    /**
     * @brief Calculate normalized swarm diversity metric
     * 
     * Measures how spread out particles are in the search space.
     * Returns value in [0,1] where 0 means fully converged.
     * 
     * @param[in] swarm Current particle swarm
     * @param[in] centroid Reference point (usually global best)
     * @return Normalized diversity measure
     */
    double calculateSwarmDiversity(
        const std::vector<particle>& swarm);

    /**
     * @brief Apply elitist learning to the best particle
     * 
     * Performs local search around the best particle using Gaussian
     * perturbations with adaptive step size based on success rate.
     * 
     * @param[in,out] best_particle Best particle to improve
     * @param[in] objectiveFunction Function to evaluate trial positions
     * @param[in] lb Lower bounds for all parameters
     * @param[in] ub Upper bounds for all parameters
     */
    void applyElitistLearningStrategy(
        particle& best_particle,
        IObjectiveFunction& objectiveFunction,
        const std::vector<double>& lb,
        const std::vector<double>& ub);

    /**
     * @brief Restart swarm while preserving elite particles
     * 
     * Reinitializes the swarm when stagnation is detected, keeping the
     * best particles and generating new ones around them or randomly.
     * 
     * @param[in,out] swarm Particle swarm to restart
     * @param[in] objectiveFunction Function to evaluate new particles
     * @param[in] parameterManager Provides parameter bounds
     * @param[in,out] gbest_position Global best position (preserved)
     * @param[in,out] gbest_value Global best value (preserved)
     * @param[in] keep_best_count Number of elite particles to preserve
     */
    void restartSwarm(
        std::vector<particle>& swarm,
        IObjectiveFunction& objectiveFunction,
        IParameterManager& parameterManager,
        Eigen::VectorXd& gbest_position,
        double& gbest_value,
        int keep_best_count = 3);

    // ===== Topology Management Methods =====

    /**
     * @brief Find the best position in a particle's neighborhood
     * 
     * Returns the best position among all neighbors according to the
     * current topology configuration.
     * 
     * @param[in] swarm Current particle swarm
     * @param[in] particle_idx Index of the particle to find neighbors for
     * @return Best position in the neighborhood
     */
    Eigen::VectorXd getNeighborhoodBest(
        const std::vector<particle>& swarm,
        int particle_idx);

    /**
     * @brief Get indices of neighboring particles based on topology
     * 
     * Returns different neighbor sets based on the configured topology:
     * ring, grid, fully connected, or random.
     * 
     * @param[in] particle_idx Index of the particle
     * @return Vector of neighbor indices (including self)
     */
    std::vector<int> getNeighbors(int particle_idx);

    // ===== Utility Methods =====

    /**
     * @brief Generate a vector of Lévy-distributed random numbers
     * 
     * Creates random jumps following a heavy-tailed Lévy distribution
     * for long-range exploration moves.
     * 
     * @param[in] dimension Size of the vector to generate
     * @return Vector of Lévy-distributed values
     */
    Eigen::VectorXd generateLevyVector(int dimension);

    /**
     * @brief Generate a single Lévy-distributed random number
     * 
     * Uses Mantegna's algorithm to generate Lévy stable distribution
     * samples with parameter alpha.
     * 
     * @return Single Lévy-distributed value
     */
    double generateLevyNumber();

    /**
     * @brief Calculate mean of all personal best positions
     * 
     * Used by quantum PSO to determine the mean best position for
     * quantum position updates.
     * 
     * @param[in] swarm Current particle swarm
     * @return Mean position vector
     */
    Eigen::VectorXd calculateMeanBestPosition(
        const std::vector<particle>& swarm);
    
    // ===== Random Number Generation =====
    
    /// @brief Mersenne Twister random number generator
    std::mt19937 rng_{std::random_device{}()};
    
    /// @brief Uniform distribution for random numbers in [0,1]
    std::uniform_real_distribution<> uniform_dist_{0.0, 1.0};
    
    /// @brief Standard normal distribution for Gaussian perturbations
    std::normal_distribution<> normal_dist_{0.0, 1.0};
};

} // namespace epidemic

#endif // PARTICLE_SWARM_OPTIMIZER_HPP