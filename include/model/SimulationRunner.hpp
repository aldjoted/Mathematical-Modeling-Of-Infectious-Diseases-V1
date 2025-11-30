#ifndef SIMULATION_RUNNER_HPP
#define SIMULATION_RUNNER_HPP

#include "model/interfaces/ISimulationRunner.hpp"
#include "model/AgeSEPAIHRDModel.hpp"
#include "sir_age_structured/interfaces/IOdeSolverStrategy.hpp"
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
     */
    SimulationRunner(
        std::shared_ptr<AgeSEPAIHRDModel> model_template,
        std::shared_ptr<IOdeSolverStrategy> solver
    );
    
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
    
    // Cache structure: hash of parameters -> simulation result
    std::map<size_t, SimulationResult> cache_;
    
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
