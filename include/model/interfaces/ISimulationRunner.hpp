#ifndef I_SIMULATION_RUNNER_HPP
#define I_SIMULATION_RUNNER_HPP

#include "model/parameters/SEPAIHRDParameters.hpp"
#include "sir_age_structured/SimulationResult.hpp"
#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace epidemic {

/**
 * @brief Interface for running simulations with caching
 */
class ISimulationRunner {
public:
    virtual ~ISimulationRunner() = default;
    
    /**
     * @brief Run simulation with given parameters and initial state
     * @param params Model parameters
     * @param initial_state Initial state vector
     * @param time_points Time points for simulation
     * @return Simulation result (may be cached)
     */
    virtual SimulationResult runSimulation(
        const SEPAIHRDParameters& params,
        const Eigen::VectorXd& initial_state,
        const std::vector<double>& time_points
    ) = 0;
    
    /**
     * @brief Clear the simulation cache
     */
    virtual void clearCache() = 0;
    
    /**
     * @brief Get cache statistics
     * @return Pair of (cache_hits, total_calls)
     */
    virtual std::pair<size_t, size_t> getCacheStats() const = 0;
};

} // namespace epidemic

#endif // I_SIMULATION_RUNNER_HPP
