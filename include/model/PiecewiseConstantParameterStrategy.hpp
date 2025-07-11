#pragma once

#include "model/interfaces/INpiStrategy.hpp"
#include "exceptions/Exceptions.hpp"
#include "model/ModelConstants.hpp"
#include <vector>
#include <string>
#include <map>
#include <memory>

namespace epidemic {

class PiecewiseConstantParameterStrategy {
public:
    PiecewiseConstantParameterStrategy(
        const std::string& param_name,
        const std::vector<double>& end_times,
        const std::vector<double>& values,
        double baseline_value,
        double baseline_end_time);

    double getValue(double time) const;

private:
    std::string parameter_name_;
    std::vector<double> period_end_times_;
    std::vector<double> parameter_values_;
    double baseline_value_;
    double baseline_period_end_time_;
};

} // namespace epidemic