#ifndef SIMULATION_CACHE_HPP
#define SIMULATION_CACHE_HPP

#include "sir_age_structured/interfaces/ISimulationCache.hpp"
#include <unordered_map>
#include <string>
#include <map>
#include <list>
#include <vector>
#include <optional>
#include <Eigen/Dense>
#include <mutex>
#include <atomic>

namespace epidemic {

/**
 * @brief An implementation of ISimulationCache using LFU eviction with LRU tie-breaking.
 *        Optimized with Zobrist-style hashing for Eigen::VectorXd keys.
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
    
    // Note: This method name is kept for interface compatibility but now returns a stringified hash
    std::string createCacheKey(const Eigen::VectorXd& parameters) const override;
    
    // Overloaded for direct hash access (internal optimization)
    size_t computeHash(const Eigen::VectorXd& parameters) const;

    bool getLikelihood(const std::string& key, double& value) override;
    void storeLikelihood(const std::string& key, double value) override;

    // Fast-path overloads for callers that already have a numeric hash key.
    // These bypass string conversions and are useful in tight calibration loops.
    bool getLikelihood(size_t key, double& value);
    void storeLikelihood(size_t key, double value);

    // Lightweight counters for benchmarking/diagnostics (thread-safe).
    size_t getLikelihoodCalls() const { return likelihood_get_calls_.load(std::memory_order_relaxed); }
    size_t getLikelihoodHits() const { return likelihood_get_hits_.load(std::memory_order_relaxed); }
    size_t storeLikelihoodCalls() const { return likelihood_store_calls_.load(std::memory_order_relaxed); }
    void resetLikelihoodStats() {
        likelihood_get_calls_.store(0, std::memory_order_relaxed);
        likelihood_get_hits_.store(0, std::memory_order_relaxed);
        likelihood_store_calls_.store(0, std::memory_order_relaxed);
    }

private:
    // Structure of Arrays (SoA) for cache locality and SIMD friendliness
    size_t capacity_;
    size_t count_;
    uint32_t current_tick_;

    std::vector<size_t> keys_;
    std::vector<double> values_;
    std::vector<uint32_t> frequencies_;
    std::vector<uint32_t> timestamps_;
    std::vector<uint8_t> occupied_; // Changed from vector<bool> to vector<uint8_t> for raw pointer access and performance

    static constexpr size_t EMPTY_KEY = 0;

    // Helper methods for Open Addressing
    size_t findIndex(size_t key) const;
    size_t evict();

    mutable std::mutex mutex_;

    // Stats (do not affect cache behavior)
    mutable std::atomic<size_t> likelihood_get_calls_{0};
    mutable std::atomic<size_t> likelihood_get_hits_{0};
    mutable std::atomic<size_t> likelihood_store_calls_{0};
};

} // namespace epidemic

#endif // SIMULATION_CACHE_HPP