#include "sir_age_structured/caching/SimulationCache.hpp"
#include <functional>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <mutex>

namespace epidemic {

// Optimized MurmurHash3-like mixer
inline size_t mix_hash(size_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccd;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53;
    k ^= k >> 33;
    return k;
}

SimulationCache::SimulationCache(size_t max_size)
    : capacity_(max_size), count_(0), current_tick_(0)
{
    if (max_size == 0) {
        throw std::invalid_argument("SimulationCache: max_size must be > 0.");
    }
    // Pre-allocate SoA vectors to ensure contiguous memory
    keys_.resize(capacity_, EMPTY_KEY);
    values_.resize(capacity_, 0.0);
    frequencies_.resize(capacity_, 0);
    timestamps_.resize(capacity_, 0);
    occupied_.resize(capacity_, 0);
}

size_t SimulationCache::computeHash(const Eigen::VectorXd& params) const {
    size_t seed = 0;
    const double* ptr = params.data();
    const int size = params.size();
    
    constexpr double precision_scale = 1e8;

    // Unrolled loop for hashing
    for (int i = 0; i < size; ++i) {
        // Fast quantization
        long long quantized = static_cast<long long>(ptr[i] * precision_scale + 0.5);
        
        // Combine hash (SplitMix64-like variant)
        size_t k = static_cast<size_t>(quantized);
        seed ^= mix_hash(k) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

std::string SimulationCache::createCacheKey(const Eigen::VectorXd& parameters) const {
    return std::to_string(computeHash(parameters));
}

// Open Addressing with Linear Probing
size_t SimulationCache::findIndex(size_t key) const {
    size_t idx = key % capacity_;
    size_t start_idx = idx;
    
    while (occupied_[idx]) {
        if (keys_[idx] == key) {
            return idx; // Found
        }
        idx++;
        if (idx == capacity_) idx = 0;
        if (idx == start_idx) break; // Should not happen if managed correctly
    }
    return -1; // Not found
}

size_t SimulationCache::evict() {
    // O(N) scan for eviction candidate (LFU with LRU tie-breaker)
    // Since cache size is small (~1000), this linear scan is faster than maintaining heap/list structures
    // due to vectorization and prefetching.
    
    size_t victim_idx = 0;
    uint32_t min_freq = std::numeric_limits<uint32_t>::max();
    uint32_t min_time = std::numeric_limits<uint32_t>::max();

    const uint32_t* __restrict__ freqs = frequencies_.data();
    const uint32_t* __restrict__ times = timestamps_.data();
    const uint8_t* __restrict__ occ = occupied_.data();
    
    for (size_t i = 0; i < capacity_; ++i) {
        if (occ[i]) {
            if (freqs[i] < min_freq) {
                min_freq = freqs[i];
                min_time = times[i];
                victim_idx = i;
            } else if (freqs[i] == min_freq) {
                if (times[i] < min_time) {
                    min_time = times[i];
                    victim_idx = i;
                }
            }
        }
    }
    
    occupied_[victim_idx] = 0;
    count_--;
    return victim_idx;
}

std::optional<double> SimulationCache::get(const Eigen::VectorXd& parameters) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t key = computeHash(parameters);
    size_t idx = findIndex(key);

    if (idx != static_cast<size_t>(-1)) {
        frequencies_[idx]++;
        timestamps_[idx] = ++current_tick_;
        return values_[idx];
    }
    return std::nullopt;
}

void SimulationCache::set(const Eigen::VectorXd& parameters, double result) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t key = computeHash(parameters);
    size_t idx = findIndex(key);

    if (idx != static_cast<size_t>(-1)) {
        // Update existing
        values_[idx] = result;
        frequencies_[idx]++;
        timestamps_[idx] = ++current_tick_;
    } else {
        // Insert new
        if (count_ >= capacity_) {
            evict(); // Make space
        }
        
        // Find empty slot (Linear Probing)
        size_t insert_idx = key % capacity_;
        while (occupied_[insert_idx]) {
            insert_idx++;
            if (insert_idx == capacity_) insert_idx = 0;
        }

        keys_[insert_idx] = key;
        values_[insert_idx] = result;
        frequencies_[insert_idx] = 1;
        timestamps_[insert_idx] = ++current_tick_;
        occupied_[insert_idx] = true;
        count_++;
    }
}

void SimulationCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::fill(occupied_.begin(), occupied_.end(), 0);
    std::fill(frequencies_.begin(), frequencies_.end(), 0);
    count_ = 0;
    current_tick_ = 0;
}

size_t SimulationCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return count_;
}

bool SimulationCache::getLikelihood(const std::string& keyStr, double& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        size_t key = std::stoull(keyStr);
        size_t idx = findIndex(key);
        if (idx != static_cast<size_t>(-1)) {
            frequencies_[idx]++;
            timestamps_[idx] = ++current_tick_;
            value = values_[idx];
            return true;
        }
    } catch (...) {}
    return false;
}

void SimulationCache::storeLikelihood(const std::string& keyStr, double value) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        size_t key = std::stoull(keyStr);
        size_t idx = findIndex(key);
        
        if (idx != static_cast<size_t>(-1)) {
            values_[idx] = value;
            frequencies_[idx]++;
            timestamps_[idx] = ++current_tick_;
        } else {
            if (count_ >= capacity_) evict();
            
            size_t insert_idx = key % capacity_;
            while (occupied_[insert_idx]) {
                insert_idx++;
                if (insert_idx == capacity_) insert_idx = 0;
            }

            keys_[insert_idx] = key;
            values_[insert_idx] = value;
            frequencies_[insert_idx] = 1;
            timestamps_[insert_idx] = ++current_tick_;
            occupied_[insert_idx] = true;
            count_++;
        }
    } catch (...) {}
}

} // namespace epidemic