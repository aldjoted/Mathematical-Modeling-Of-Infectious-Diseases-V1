#include "model/PiecewiseConstantParameterStrategy.hpp"
#include <algorithm>


namespace epidemic {

    PiecewiseConstantParameterStrategy::PiecewiseConstantParameterStrategy(
        const std::string& param_name,
        const std::vector<double>& end_times,
        const std::vector<double>& values,
        double baseline_value,
        double baseline_end_time)
        : parameter_name_(param_name),
          period_end_times_(end_times),
          parameter_values_(values),
          baseline_value_(baseline_value),
          baseline_period_end_time_(baseline_end_time) {
        if (period_end_times_.size() != parameter_values_.size()) {
            THROW_INVALID_PARAM("PiecewiseConstantParameterStrategy::PiecewiseConstantParameterStrategy", "End times and values vectors must have the same size for parameter " + parameter_name_);
        }

        if (!std::is_sorted(period_end_times_.begin(), period_end_times_.end())) {
            THROW_INVALID_PARAM("PiecewiseConstantParameterStrategy::PiecewiseConstantParameterStrategy",
                                "End times must be sorted for parameter " + parameter_name_);
        }

        double prev = baseline_period_end_time_;
        for (double t : period_end_times_) {
            if (t <= prev) {
                THROW_INVALID_PARAM("PiecewiseConstantParameterStrategy::PiecewiseConstantParameterStrategy",
                                    "Each end time must be strictly greater than the baseline period end time and previous end time for parameter " + parameter_name_);
            }
            prev = t;
        }
    }

    double PiecewiseConstantParameterStrategy::getValue(double time) const {
        if (time <= baseline_period_end_time_) {
            return baseline_value_;
        }

        if (period_end_times_.empty()) {
            return baseline_value_;
        }

        // Fast path: most ODE solvers evaluate derivatives at non-decreasing times.
        // If time goes backwards, fall back to binary search.
        if (time < cached_time_) {
            auto it = std::lower_bound(period_end_times_.begin(), period_end_times_.end(), time);
            if (it == period_end_times_.end()) {
                cached_period_index_ = period_end_times_.size();
                cached_time_ = time;
                return parameter_values_.back();
            }
            cached_period_index_ = static_cast<size_t>(std::distance(period_end_times_.begin(), it));
            cached_time_ = time;
            return parameter_values_[cached_period_index_];
        }

        size_t idx = cached_period_index_;
        if (idx > period_end_times_.size()) {
            idx = 0;
        }
        while (idx < period_end_times_.size() && time > period_end_times_[idx]) {
            ++idx;
        }
        cached_period_index_ = idx;
        cached_time_ = time;

        if (idx >= parameter_values_.size()) {
            return parameter_values_.back();
        }
        return parameter_values_[idx];
    }

}