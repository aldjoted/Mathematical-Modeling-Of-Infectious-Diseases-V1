#ifndef SIMULATION_CACHE_HPP
#define SIMULATION_CACHE_HPP

#include "sir_age_structured/interfaces/ISimulationCache.hpp"
#include <vector>
#include <string>
#include <optional>
#include <cstdint>
#include <Eigen/Dense>
#include <mutex>

namespace epidemic {

/**
 * @brief High-Performance SimulationCache using Structure of Arrays (SoA) and Open Addressing.
 * Eliminates pointer chasing and dynamic allocations (std::list, std::unordered_map nodes).
 * Optimized for cache locality and CPU pipelining.
 * Thread-safe using a mutex.
 */
class SimulationCache : public ISimulationCache {
public:
    /**
     * @brief Constructor.
     * @param max_size Maximum number of entries in the cache.
     */
    explicit SimulationCache(size_t max_size = 1000);

    // Implementation of the ISimulationCache interface
    std::optional<double> get(const Eigen::VectorXd& parameters) override;
    void set(const Eigen::VectorXd& parameters, double result) override;
    void clear() override;
    size_t size() const override;

    std::string createCacheKey(const Eigen::VectorXd& parameters) const override;
    
    // Internal optimized hash computation
    size_t computeHash(const Eigen::VectorXd& parameters) const;

    bool getLikelihood(const std::string& key, double& value) override;
    void storeLikelihood(const std::string& key, double value) override;

private:
    // Structure of Arrays (SoA) layout for cache locality
    std::vector<size_t> keys_;           // Stores hash keys
    std::vector<double> values_;         // Stores cached results
    std::vector<uint32_t> frequencies_;  // Stores access frequencies (for LFU)
    std::vector<uint32_t> timestamps_;   // Stores access timestamps (for LRU tie-breaking)
    std::vector<bool> occupied_;         // Validity mask

    size_t capacity_;
    size_t count_;
    mutable uint32_t current_tick_; // Global counter for LRU
    mutable std::mutex mutex_;      // Mutex for thread safety

    // Open addressing parameters
    static constexpr size_t EMPTY_KEY = 0;
    
    // Internal helper to find index
    size_t findIndex(size_t key) const;
    // Internal helper to handle eviction
    size_t evict();
};

} // namespace epidemic

#endif // SIMULATION_CACHE_HPP
