#include "sir_age_structured/caching/SimulationCache.hpp"
#include <sstream>
#include <iomanip>
#include <stdexcept>

namespace epidemic {

SimulationCache::SimulationCache(size_t max_size, int hash_precision)
    : max_size_(max_size), hash_precision_(hash_precision), min_frequency_(0)
{
    if (max_size_ == 0) {
        throw std::invalid_argument("SimulationCache (LFU): max_size must be greater than 0.");
    }
}

// Private helper to generate a unique string key from parameter values,
// considering the specified precision.
std::string SimulationCache::createStringKey(const Eigen::VectorXd& params) const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(hash_precision_);
    for (int i = 0; i < params.size(); ++i) {
        oss << params[i];
        if (i < params.size() - 1) {
            oss << "_";
        }
    }
    return oss.str();
}

std::string SimulationCache::createCacheKey(const Eigen::VectorXd& parameters) const {
    return createStringKey(parameters);
}

// Helper to update frequency and recency for a key access.
void SimulationCache::updateFrequency(const std::string& key) {
    auto& node = cache_.at(key);
    int old_freq = node.frequency;

    freq_map_[old_freq].erase(node.freq_list_iter);
    
    // If that frequency list is empty and was the minimum, update min_frequency_.
    if (freq_map_[old_freq].empty()) {
        freq_map_.erase(old_freq);
        if (old_freq == min_frequency_) {
            // Next available frequency becomes the new minimum.
            min_frequency_ = freq_map_.empty() ? 0 : freq_map_.begin()->first;
        }
    }

    node.frequency++;
    int new_freq = node.frequency;

    // Add key to the most recent position in the new frequency list.
    freq_map_[new_freq].push_back(key);
    node.freq_list_iter = std::prev(freq_map_[new_freq].end());
}

std::optional<double> SimulationCache::get(const Eigen::VectorXd& parameters) {
    std::string key = createStringKey(parameters);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        return std::nullopt; // Cache miss
    }

    // Cache hit: Update frequency and return value
    updateFrequency(key);
    return it->second.value;
}

void SimulationCache::set(const Eigen::VectorXd& parameters, double result) {
    std::string key = createStringKey(parameters);
    // If key exists, update value and frequency.
    if (cache_.find(key) != cache_.end()) {
        cache_[key].value = result;
        updateFrequency(key);
    } else {
        // Evict an element if the cache is full.
        if (cache_.size() >= max_size_) {
            // Evict the LRU item among those with the minimum frequency.
            auto it = freq_map_.find(min_frequency_);
            if (it != freq_map_.end() && !it->second.empty()) {
                std::string key_to_evict = it->second.front();   // Get least recently used key at min frequency
                it->second.pop_front();
                cache_.erase(key_to_evict);
                if (it->second.empty()) {
                    freq_map_.erase(min_frequency_);
                }
            }
        }
        // Insert new element with frequency 1.
        int initial_frequency = 1;
        freq_map_[initial_frequency].push_back(key);
        auto list_it = std::prev(freq_map_[initial_frequency].end());
        cache_[key] = {result, initial_frequency, list_it};
        min_frequency_ = 1; // New entries always have frequency 1.
    }
}

void SimulationCache::clear() {
    cache_.clear();
    freq_map_.clear();
    min_frequency_ = 0;
}

// Returns the current cache size.
size_t SimulationCache::size() const {
    return cache_.size();
}

// Attempts to retrieve a cached likelihood value using a key.
bool SimulationCache::getLikelihood(const std::string& key, double& value) {
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        updateFrequency(key);
        value = it->second.value;
        return true;
    }
    return false;
}

// Store a likelihood result in the cache using a key.
void SimulationCache::storeLikelihood(const std::string& key, double value) {
    // If key exists, update value and frequency.
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        it->second.value = value;
        updateFrequency(key);
    } else {
        // Evict if cache is full.
        if (cache_.size() >= max_size_) {
            auto freq_it = freq_map_.find(min_frequency_);
            if (freq_it != freq_map_.end() && !freq_it->second.empty()) {
                std::string key_to_evict = freq_it->second.front();
                freq_it->second.pop_front();
                cache_.erase(key_to_evict);
                if (freq_it->second.empty()) {
                    freq_map_.erase(min_frequency_);
                }
            }
        }
        // Insert the new key with initial frequency.
        int initial_frequency = 1;
        freq_map_[initial_frequency].push_back(key);
        auto list_it = std::prev(freq_map_[initial_frequency].end());
        cache_[key] = {value, initial_frequency, list_it};
        min_frequency_ = 1;
    }
}

} // namespace epidemic