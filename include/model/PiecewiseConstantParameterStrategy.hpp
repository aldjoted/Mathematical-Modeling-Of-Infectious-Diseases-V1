#pragma once

#include "model/interfaces/INpiStrategy.hpp"
#include "exceptions/Exceptions.hpp"
#include "model/ModelConstants.hpp"
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <limits>

namespace epidemic {

/**
 * @brief Strategy for piecewise constant parameter evolution over time.
 * 
 * This class manages time-varying parameters (e.g., transmission rate beta)
 * that change at discrete time points. It efficiently returns the correct
 * parameter value for any given simulation time using cached interval lookup.
 */
class PiecewiseConstantParameterStrategy {
public:
    /**
     * @brief Constructs a piecewise constant parameter strategy.
     * 
     * @param param_name Identifier for this parameter (for logging/debugging).
     * @param end_times End times for each period after baseline (strictly increasing).
     * @param values Parameter values for each period corresponding to end_times.
     * @param baseline_value Parameter value during the initial baseline period.
     * @param baseline_end_time End time of the baseline period.
     * @throws InvalidParameterException if end_times and values have mismatched sizes.
     */
    PiecewiseConstantParameterStrategy(
        const std::string& param_name,
        const std::vector<double>& end_times,
        const std::vector<double>& values,
        double baseline_value,
        double baseline_end_time);

    /**
     * @brief Gets the parameter value at the specified time.
     * 
     * Uses cached interval lookup for O(1) average-case performance when
     * times are queried in mostly non-decreasing order (typical for ODE solvers).
     * 
     * @param time The simulation time at which to retrieve the parameter value.
     * @return The parameter value at the given time.
     */
    double getValue(double time) const;

private:
    std::string parameter_name_;
    std::vector<double> period_end_times_;
    std::vector<double> parameter_values_;
    double baseline_value_;
    double baseline_period_end_time_;

    /** @brief Cached period index for O(1) lookups when time advances monotonically. */
    mutable size_t cached_period_index_ = 0;
    /** @brief Last queried time for cache invalidation on time reversal. */
    mutable double cached_time_ = std::numeric_limits<double>::lowest();
};

} // namespace epidemic