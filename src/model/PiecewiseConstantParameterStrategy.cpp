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
    }

    double PiecewiseConstantParameterStrategy::getValue(double time) const {
        if (time <= baseline_period_end_time_) {
            return baseline_value_;
        }

        auto it = std::upper_bound(period_end_times_.begin(), period_end_times_.end(), time);
        
        if (it == period_end_times_.begin()) {
            return baseline_value_;
        }

        size_t index = std::distance(period_end_times_.begin(), it);

        if (index < parameter_values_.size()) {
            return parameter_values_[index];
        }

        return parameter_values_.back();
    }

}