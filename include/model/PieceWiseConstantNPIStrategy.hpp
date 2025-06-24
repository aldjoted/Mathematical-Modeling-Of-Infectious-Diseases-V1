#ifndef PIECEWISE_CONSTANT_NPI_STRATEGY_HPP
#define PIECEWISE_CONSTANT_NPI_STRATEGY_HPP

#include "interfaces/INpiStrategy.hpp"
#include "exceptions/Exceptions.hpp"
#include "model/ModelConstants.hpp"
#include <vector>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <string>
#include <map>

namespace epidemic {

/**
 * @brief Implements an NPI strategy with piecewise constant reduction factors (kappa).
 * This strategy defines an initial baseline period with its own kappa value,
 * which, in the current `getReductionFactor` implementation, ends at a specific time (e.g., day 13.0).
 * Subsequent NPI periods, each with a constant NPI effect (kappa value), are defined by `npi_end_times`
 * and their corresponding `npi_values`. The baseline kappa can be fixed or variable.
 */
class PiecewiseConstantNpiStrategy : public INpiStrategy {
public:
    /**
     * @brief Constructs a PiecewiseConstantNpiStrategy object.
     * @param npi_end_times_after_baseline Vector of absolute end times for NPI periods that occur *after* an initial baseline period.
     *                      These times must be sorted, non-negative, and strictly greater than `baseline_period_end_time`.
     *                      `npi_end_times_after_baseline[0]` is the end time for the NPI period immediately following the initial baseline,
     *                      which uses the reduction factor `npi_values_after_baseline[0]`.
     * @param npi_values_after_baseline Vector of reduction factors (kappa values) for the NPI periods defined by `npi_end_times_after_baseline`.
     *                   `npi_values_after_baseline[i]` corresponds to the period ending at `npi_end_times_after_baseline[i]`.
     *                   The size must match `npi_end_times_after_baseline`. Values must be non-negative.
     * @param param_specific_bounds A map defining specific lower and upper bounds for named NPI parameters.
     *                              Keys are parameter names (e.g., "kappa_2", "kappa_baseline"), values are pairs of (lower, upper) bounds.
     * @param baseline_kappa The reduction factor for the initial baseline period. Must be non-negative. Defaults to 1.0.
     * @param baseline_period_end_time The end time of the initial baseline period. Must be non-negative.
     *                                 Subsequent `npi_end_times_after_baseline` must be strictly greater than this value.
     * @param fixed_baseline If true, the `baseline_kappa` is considered fixed and not a calibratable parameter. Defaults to true.
     * @param param_names_for_npi_values Optional vector of names for the NPI kappa values in `npi_values_after_baseline`.
     *                               If provided, its size must match `npi_values_after_baseline`. If empty, names like "kappa_2", "kappa_3", etc., will be generated.
     *                               "kappa_baseline" is used for the baseline kappa if it's not fixed.
     * @throws InvalidParameterException if inputs are invalid.
     */
    PiecewiseConstantNpiStrategy(
        const std::vector<double>& npi_end_times_after_baseline,
        const std::vector<double>& npi_values_after_baseline,
        const std::map<std::string, std::pair<double, double>>& param_specific_bounds = {},
        double baseline_kappa = constants::DEFAULT_BASELINE_KAPPA,
        double baseline_period_end_time = constants::DEFAULT_BASELINE_PERIOD_END_TIME,
        bool fixed_baseline = true,
        const std::vector<std::string>& param_names_for_npi_values = {}
    );

    /**
     * @brief Gets the NPI reduction factor (kappa) at a specific time.
     * @param time The time at which to evaluate the NPI effect.
     * @return The kappa value. Returns `baseline_kappa_value_` if `time` is negative or falls within the
     *         initial baseline period (e.g., `0 <= time <= 13.0` as per current implementation).
     *         For times after the initial baseline period, it returns the kappa value corresponding to the
     *         NPI period defined by `npi_period_end_times_` and `npi_kappa_values_`.
     *         If `time` is beyond the last defined `npi_period_end_times_`, the last `npi_kappa_values_` is used.
     *         If `npi_period_end_times_` is empty, `baseline_kappa_value_` is returned for times after the initial baseline.
     */
    double getReductionFactor(double time) const override;

    /**
     * @brief Gets the vector of end times for the NPI periods.
     * @return A constant reference to the vector of NPI period end times.
     */
    const std::vector<double>& getEndTimes() const override;

    /**
     * @brief Gets the vector of NPI reduction factors (kappa values).
     * This includes the baseline kappa value followed by kappa values for subsequent periods.
     * @return A vector containing all NPI kappa values.
     */
    std::vector<double> getValues() const override;

    /**
     * @brief Sets the NPI reduction factors (kappa values) for the periods.
     * @param new_values The new vector of kappa values for periods after the baseline. Its size must match the
     *                   existing number of NPI periods defined by `npi_period_end_times_` (i.e., `npi_kappa_values_`).
     *                   All values must be non-negative.
     * @throws InvalidParameterException if new_values size doesn't match or contains negative values.
     */
    void setValues(const std::vector<double>& new_values) override;

    /**
     * @brief Gets the baseline NPI reduction factor (kappa).
     * @return The kappa value for the initial baseline period (e.g., `0 <= time <= 13.0`).
     */
    double getBaselineKappa() const;
    
    /**
     * @brief Gets the end time of the initial baseline period.
     * @return The end time of the baseline period (e.g., 13.0).
     */
    double getBaselinePeriodEndTime() const;

    /**
     * @brief Creates a deep copy (clone) of the current NPI strategy object.
     * @return A shared_ptr to the newly created PiecewiseConstantNpiStrategy object.
     */
    std::shared_ptr<INpiStrategy> clone() const override;

    /**
     * @brief Gets the lower bound for a calibratable NPI parameter by its index.
     * The index corresponds to the position in the `npi_kappa_values_` vector.
     * If the baseline kappa is not fixed, it is considered the first calibratable parameter (index 0),
     * followed by the elements of `npi_kappa_values_`.
     * @param calibratable_idx The index of the calibratable NPI parameter.
     * @return The lower bound for the specified parameter. Uses specific bound if defined, otherwise default.
     * @throws InvalidParameterException if calibratable_idx is out of range.
     */
    double getLowerBoundForParamIndex(int calibratable_idx) const override;

    /**
     * @brief Gets the upper bound for a calibratable NPI parameter by its index.
     * The index corresponds to the position in the `npi_kappa_values_` vector.
     * If the baseline kappa is not fixed, it is considered the first calibratable parameter (index 0),
     * followed by the elements of `npi_kappa_values_`.
     * @param calibratable_idx The index of the calibratable NPI parameter.
     * @return The upper bound for the specified parameter. Uses specific bound if defined, otherwise default.
     * @throws InvalidParameterException if calibratable_idx is out of range.
     */
    double getUpperBoundForParamIndex(int calibratable_idx) const override;

    /**
     * @brief Gets the total number of calibratable NPI parameters.
     * This includes the NPI kappa values for each period. If the baseline kappa is not fixed, it is also counted.
     * @return The number of NPI parameters that can be calibrated.
     */
    size_t getNumCalibratableNpiParams() const;

    /**
     * @brief Gets the name of a calibratable NPI parameter by its index.
     * The index corresponds to the position in the `npi_kappa_values_` vector.
     * If the baseline kappa is not fixed, it is "kappa_baseline" at index 0.
     * Otherwise, names are taken from `npi_param_names_` or generated (e.g., "kappa_2").
     * @param calibratable_idx The index of the calibratable NPI parameter.
     * @return The name of the specified NPI parameter.
     * @throws InvalidParameterException if calibratable_idx is out of range.
     */
    std::string getNpiParamName(int calibratable_idx) const;

    /**
     * @brief Sets all calibratable NPI parameter values.
     * This includes the baseline kappa if it is not fixed, followed by the kappa values
     * for subsequent NPI periods. The order and number of values must match those
     * defined by getNumCalibratableNpiParams() and getNpiParamName().
     * @param calibratable_values A vector containing all calibratable NPI values.
     * @throws InvalidParameterException if the size of calibratable_values is incorrect
     *         or if any value is negative.
     */
    void setCalibratableValues(const std::vector<double>& calibratable_values);

    /**
     * @brief Gets all calibratable NPI parameter values.
     * @return A vector containing the baseline kappa (if not fixed) followed by the
     *         kappa values for subsequent NPI periods.
     */
    std::vector<double> getCalibratableValues() const;

    /**
     * @brief Checks if the baseline NPI parameter is fixed.
     * @return True if the baseline kappa is fixed, false if it is calibratable.
     */
    bool isBaselineFixed() const;


private:
    /** @brief The NPI reduction factor (kappa) for the baseline period. */
    double baseline_kappa_value_;
    /** @brief Flag indicating if the baseline kappa is fixed (true) or calibratable (false). */
    bool is_baseline_fixed_;
    /** @brief The end time of the initial baseline period. */
    double baseline_period_end_time_;
    /** @brief Vector of end times for each NPI period after the baseline. */
    std::vector<double> npi_period_end_times_;
    /** @brief Vector of NPI reduction factors (kappa values) corresponding to each period in npi_period_end_times_. */
    std::vector<double> npi_kappa_values_;
    /** @brief Vector of names for the NPI kappa values in npi_kappa_values_. Used for identifying parameters for bounds. */
    std::vector<std::string> npi_param_names_;
    /** @brief Map storing specific lower and upper bounds for named NPI parameters. */
    std::map<std::string, std::pair<double, double>> param_bounds_map_;
    /** @brief Default lower bound for NPI kappa parameters if not specified in param_bounds_map_. */
    const double DEFAULT_LOWER_BOUND = constants::DEFAULT_NPI_LOWER_BOUND;
    /** @brief Default upper bound for NPI kappa parameters if not specified in param_bounds_map_. */
    const double DEFAULT_UPPER_BOUND = constants::DEFAULT_NPI_UPPER_BOUND;
};

} // namespace epidemic

#endif