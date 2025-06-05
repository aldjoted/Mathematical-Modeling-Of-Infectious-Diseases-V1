#ifndef SIMULATION_CACHE_HPP
#define SIMULATION_CACHE_HPP

#include "sir_age_structured/interfaces/ISimulationCache.hpp"
#include <unordered_map>
#include <string>
#include <sstream>
#include <map>
#include <list>
#include <iomanip>
#include <optional>
#include <stdexcept>

namespace epidemic {

/**
 * @brief An implementation of ISimulationCache using LFU (Least Frequently Used)
 *        eviction with LRU tie-breaking.
 */
class SimulationCache : public ISimulationCache {
public:
    /**
     * @brief Constructor.
     * @param max_size Maximum number of entries in the cache (must be > 0).
     * @param hash_precision Precision used when converting parameter vectors to keys.
     */
    explicit SimulationCache(size_t max_size = 1000, int hash_precision = 8);

    // Implementation of the ISimulationCache interface
    std::optional<double> get(const Eigen::VectorXd& parameters) override;
    void set(const Eigen::VectorXd& parameters, double result) override;
    void clear() override;
    size_t size() const override;
    std::string createCacheKey(const Eigen::VectorXd& parameters) const override;
    bool getLikelihood(const std::string& key, double& value) override;
    void storeLikelihood(const std::string& key, double value) override;

private:
    /**
     * @brief Internal structure for storing cache node data.
     * 
     * Contains the cached value along with metadata for LFU cache management.
     */
    struct CacheNode {
        /** @brief The cached simulation result value. */
        double value;
        /** @brief Access frequency counter for LFU eviction policy. */
        int frequency;
        /** @brief Iterator to this key's position in the frequency list. */
        std::list<std::string>::iterator freq_list_iter;
    };

    /** @brief Main cache storage mapping string keys to cache nodes. */
    std::unordered_map<std::string, CacheNode> cache_;

    /** @brief Frequency map organizing keys by access frequency for LFU eviction. */
    std::map<int, std::list<std::string>> freq_map_;

    /** @brief Maximum number of entries allowed in the cache. */
    size_t max_size_;
    
    /** @brief Decimal precision used for parameter-to-key conversion. */
    int hash_precision_;
    
    /** @brief Current minimum frequency value for efficient LFU eviction. */
    int min_frequency_;

    /**
     * @brief Updates the access frequency for a cache key.
     * @param key The cache key whose frequency should be incremented.
     */
    void updateFrequency(const std::string& key);

    /**
     * @brief Converts parameter vector to string key for cache lookup.
     * @param params The parameter vector to convert.
     * @return String representation of the parameters for use as cache key.
     */
    std::string createStringKey(const Eigen::VectorXd& params) const;
};

} // namespace epidemic

#endif // SIMULATION_CACHE_HPP