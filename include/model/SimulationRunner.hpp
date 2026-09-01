#ifndef SIMULATION_RUNNER_HPP
#define SIMULATION_RUNNER_HPP

#include "model/interfaces/ISimulationRunner.hpp"
#include "model/AgeSEPAIHRDModel.hpp"
#include "sir_age_structured/interfaces/IOdeSolverStrategy.hpp"
#include <cstddef>
#include <deque>
#include <memory>
#include <map>
#include <vector>

namespace epidemic {

/**
 * @brief Concrete implementation of ISimulationRunner with caching
 * 
 * This class uses memoization to prevent redundant simulations of identical parameter sets.
 * The cache key is a hash of the parameter vector.
 */
class SimulationRunner : public ISimulationRunner {
public:
    /**
     * @brief Construct a new Simulation Runner
     * @param model_template Shared pointer to model template for cloning
     * @param solver Shared pointer to ODE solver strategy
     * @param max_cache_entries Maximum number of trajectories retained by the memo cache.
     *        Each entry holds a full trajectory, so an unbounded cache grows without limit
     *        when the runner is driven over distinct posterior samples (which essentially
     *        never repeat). Oldest entries are evicted first.
     */
    SimulationRunner(
        std::shared_ptr<AgeSEPAIHRDModel> model_template,
        std::shared_ptr<IOdeSolverStrategy> solver,
        std::size_t max_cache_entries = DEFAULT_MAX_CACHE_ENTRIES
    );

    /** @brief Default upper bound on retained trajectories. */
    static constexpr std::size_t DEFAULT_MAX_CACHE_ENTRIES = 64;
    
    SimulationResult runSimulation(
        const SEPAIHRDParameters& params,
        const Eigen::VectorXd& initial_state,
        const std::vector<double>& time_points
    ) override;
    
    void clearCache() override;
    
    std::pair<size_t, size_t> getCacheStats() const override;
    
private:
    std::shared_ptr<AgeSEPAIHRDModel> model_template_;
    std::shared_ptr<IOdeSolverStrategy> solver_;
    
    // Cache structure: hash of parameters -> simulation result.
    // Bounded by max_cache_entries_ with FIFO eviction; cache_order_ records insertion order.
    std::map<size_t, SimulationResult> cache_;
    std::deque<size_t> cache_order_;
    std::size_t max_cache_entries_;
    
    // Statistics
    mutable size_t cache_hits_ = 0;
    mutable size_t total_calls_ = 0;
    
    /**
     * @brief Generate a hash key from parameter vector
     * @param param_vec Parameter vector to hash
     * @return Hash value for use as cache key
     */
    size_t hashParameterVector(const Eigen::VectorXd& param_vec) const;
    
    /**
     * @brief Convert parameters to a vector for hashing
     * @param params Model parameters
     * @return Flat vector representation
     */
    Eigen::VectorXd parametersToVector(const SEPAIHRDParameters& params) const;
};

} // namespace epidemic

#endif // SIMULATION_RUNNER_HPP
