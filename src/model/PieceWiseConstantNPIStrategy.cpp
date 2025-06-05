#include "model/PieceWiseConstantNPIStrategy.hpp"
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace epidemic {

PiecewiseConstantNpiStrategy::PiecewiseConstantNpiStrategy(
    const std::vector<double>& npi_end_times_after_baseline,
    const std::vector<double>& npi_values_after_baseline,
    const std::map<std::string, std::pair<double, double>>& param_specific_bounds,
    double baseline_kappa,
    double baseline_period_end_time,
    bool fixed_baseline,
    const std::vector<std::string>& param_names_for_npi_values)
    : baseline_kappa_value_(baseline_kappa),
      is_baseline_fixed_(fixed_baseline),
      baseline_period_end_time_(baseline_period_end_time),
      npi_period_end_times_(npi_end_times_after_baseline),
      npi_kappa_values_(npi_values_after_baseline),
      npi_param_names_(param_names_for_npi_values),
      param_bounds_map_(param_specific_bounds) {

    if (baseline_period_end_time_ < 0.0) {
        THROW_INVALID_PARAM("PiecewiseConstantNpiStrategy", "Baseline period end time must be non-negative.");
    }

    if (npi_period_end_times_.size() != npi_kappa_values_.size()) {
        THROW_INVALID_PARAM("PiecewiseConstantNpiStrategy", "NPI end times vector size must match NPI values vector size.");
    }
    if (!npi_param_names_.empty() && npi_param_names_.size() != npi_kappa_values_.size()) {
        THROW_INVALID_PARAM("PiecewiseConstantNpiStrategy", "Explicit NPI parameter names vector size must match NPI values vector size if provided.");
    }

    if (!std::is_sorted(npi_period_end_times_.begin(), npi_period_end_times_.end())) {
        THROW_INVALID_PARAM("PiecewiseConstantNpiStrategy", "NPI end times must be sorted.");
    }

    double previous_time = baseline_period_end_time_;
    for (double t : npi_period_end_times_) {
        if (t <= previous_time) {
            THROW_INVALID_PARAM("PiecewiseConstantNpiStrategy", "Each NPI end time must be strictly greater than the baseline period end time and any preceding NPI end time.");
        }
        previous_time = t;
    }

    if (baseline_kappa_value_ < 0.0) {
        THROW_INVALID_PARAM("PiecewiseConstantNpiStrategy", "Baseline kappa value must be non-negative.");
    }
    for (double k_val : npi_kappa_values_) {
        if (k_val < 0.0) {
            THROW_INVALID_PARAM("PiecewiseConstantNpiStrategy", "NPI kappa values must be non-negative.");
        }
    }

    if (npi_param_names_.empty() && !npi_kappa_values_.empty()) {
        npi_param_names_.resize(npi_kappa_values_.size());
        for(size_t i=0; i < npi_kappa_values_.size(); ++i) {
            // This naming convention assumes kappa_1 is the fixed baseline,
            npi_param_names_[i] = "kappa_" + std::to_string(i + 2);
        }
    }

    for(const auto& pair : param_bounds_map_){
        bool found = false;
        for(const auto& name : npi_param_names_){
            if(pair.first == name){
                found = true;
                break;
            }
        }
        if (!is_baseline_fixed_ && pair.first == "kappa_baseline") {
            found = true;
        }
        if(!found){
            std::cerr << "[PiecewiseConstantNpiStrategy Warning] Bound provided for unknown/unused NPI parameter name: " << pair.first << std::endl;
        }
    }
}

bool PiecewiseConstantNpiStrategy::isBaselineFixed() const {
    return is_baseline_fixed_;
}

double PiecewiseConstantNpiStrategy::getReductionFactor(double time) const {
    if (time < 0) {
        return baseline_kappa_value_;
    }

    if (time <= baseline_period_end_time_) {
        return baseline_kappa_value_;
    }

    double previous_npi_end_time = baseline_period_end_time_;
    for (size_t i = 0; i < npi_period_end_times_.size(); ++i) {
        if (time > previous_npi_end_time && time <= npi_period_end_times_[i]) {
            return npi_kappa_values_[i];
        }
        previous_npi_end_time = npi_period_end_times_[i];
    }

    if (!npi_kappa_values_.empty()) {
        return npi_kappa_values_.back();
    } else {
        return baseline_kappa_value_;
    }
}

const std::vector<double>& PiecewiseConstantNpiStrategy::getEndTimes() const {
    return npi_period_end_times_;
}

std::vector<double> PiecewiseConstantNpiStrategy::getValues() const {
    std::vector<double> all_kappa_values;
    all_kappa_values.push_back(baseline_kappa_value_);
    all_kappa_values.insert(all_kappa_values.end(), npi_kappa_values_.begin(), npi_kappa_values_.end());
    return all_kappa_values;
}

double PiecewiseConstantNpiStrategy::getBaselineKappa() const {
    return baseline_kappa_value_;
}

double PiecewiseConstantNpiStrategy::getBaselinePeriodEndTime() const {
    return baseline_period_end_time_;
}

void PiecewiseConstantNpiStrategy::setValues(const std::vector<double>& new_npi_values) {
    if (new_npi_values.size() != npi_kappa_values_.size()) {
        THROW_INVALID_PARAM("PiecewiseConstantNpiStrategy::setValues",
                            "New NPI values vector size must match existing number of changeable NPI periods.");
    }
    for (double k_val : new_npi_values) {
        if (k_val < 0.0) {
            THROW_INVALID_PARAM("PiecewiseConstantNpiStrategy::setValues", "NPI kappa values must be non-negative.");
        }
    }
    npi_kappa_values_ = new_npi_values;
}

std::shared_ptr<INpiStrategy> PiecewiseConstantNpiStrategy::clone() const {
    return std::make_shared<PiecewiseConstantNpiStrategy>(
        npi_period_end_times_,
        npi_kappa_values_,
        param_bounds_map_,
        baseline_kappa_value_,
        baseline_period_end_time_,
        is_baseline_fixed_,
        npi_param_names_
    );
}

size_t PiecewiseConstantNpiStrategy::getNumCalibratableNpiParams() const {
    size_t count = npi_kappa_values_.size();
    if (!is_baseline_fixed_) {
        count++;
    }
    return count;
}

std::string PiecewiseConstantNpiStrategy::getNpiParamName(int calibratable_idx) const {
    if (calibratable_idx < 0 || static_cast<size_t>(calibratable_idx) >= getNumCalibratableNpiParams()) {
         THROW_INVALID_PARAM("PiecewiseConstantNpiStrategy::getNpiParamName", "calibratable_idx out of range.");
    }

    if (!is_baseline_fixed_) {
        if (calibratable_idx == 0) {
            return "kappa_baseline";
        }
        if (static_cast<size_t>(calibratable_idx - 1) < npi_param_names_.size()) {
             return npi_param_names_[calibratable_idx - 1];
        } else {
            THROW_INVALID_PARAM("PiecewiseConstantNpiStrategy::getNpiParamName", "Internal indexing error for NPI parameter names.");
        }
    } else {
        if (static_cast<size_t>(calibratable_idx) < npi_param_names_.size()){
            return npi_param_names_[calibratable_idx];
        } else {
             THROW_INVALID_PARAM("PiecewiseConstantNpiStrategy::getNpiParamName", "Internal indexing error for NPI parameter names (fixed baseline).");
        }
    }
}

double PiecewiseConstantNpiStrategy::getLowerBoundForParamIndex(int calibratable_idx) const {
    if (calibratable_idx < 0 || static_cast<size_t>(calibratable_idx) >= getNumCalibratableNpiParams()) {
         THROW_INVALID_PARAM("PiecewiseConstantNpiStrategy::getLowerBoundForParamIndex", "calibratable_idx out of range.");
    }
    const std::string& param_name = getNpiParamName(calibratable_idx);
    auto it = param_bounds_map_.find(param_name);
    if (it != param_bounds_map_.end()) {
        return it->second.first;
    }
    return DEFAULT_LOWER_BOUND;
}

double PiecewiseConstantNpiStrategy::getUpperBoundForParamIndex(int calibratable_idx) const {
    if (calibratable_idx < 0 || static_cast<size_t>(calibratable_idx) >= getNumCalibratableNpiParams()) {
         THROW_INVALID_PARAM("PiecewiseConstantNpiStrategy::getUpperBoundForParamIndex", "calibratable_idx out of range.");
    }
    const std::string& param_name = getNpiParamName(calibratable_idx);
    auto it = param_bounds_map_.find(param_name);
    if (it != param_bounds_map_.end()) {
        return it->second.second;
    }
    return DEFAULT_UPPER_BOUND;
}

} // namespace epidemic
