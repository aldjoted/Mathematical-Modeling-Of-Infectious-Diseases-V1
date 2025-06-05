#ifndef I_NPI_STRATEGY_HPP
#define I_NPI_STRATEGY_HPP

#include <vector>
#include <memory>

namespace epidemic {

/**
 * @brief Interface for Non-Pharmaceutical Intervention (NPI) strategies.
 *
 * Defines how NPIs affect model parameters (e.g., contact rates) over time.
 */
class INpiStrategy {
public:
    virtual ~INpiStrategy() = default;

    /**
     * @brief Get the reduction factor applied to the baseline contact matrix at a given time.
     *
     * A factor of 1.0 means no reduction, while values between 0 and 1 represent
     * varying levels of contact reduction due to NPIs. Values > 1 could represent
     * increased contact, though less common for NPIs.
     *
     * @param time The current simulation time.
     * @return double The contact reduction factor (kappa equivalent).
     */
    virtual double getReductionFactor(double time) const = 0;

    /**
     * @brief Get the end times defining the NPI periods.
     * @return const std::vector<double>& Vector of end times.
     */
    virtual const std::vector<double>& getEndTimes() const = 0;

    /**
     * @brief Get the reduction factor values for each NPI period.
     * @return const std::vector<double>& Vector of reduction factors (kappa values).
     */
     virtual std::vector<double> getValues() const = 0;

     virtual double getBaselineKappa() const = 0;
     /**
      * @brief Get the baseline NPI reduction factor (kappa).
      * @return double The kappa value for the initial baseline period.
      */
     virtual double getBaselinePeriodEndTime() const = 0;

    /**
     * @brief Set the reduction factor values for calibration/updating.
     * @param new_values Vector of new reduction factors.
     * @throws InvalidParameterException if size mismatches or values are invalid.
     */
    virtual void setValues(const std::vector<double>& new_values) = 0;

    /**
     * @brief Creates a deep copy of the strategy.
     * @return std::shared_ptr<INpiStrategy> A shared pointer to the new copy.
     */
    virtual std::shared_ptr<INpiStrategy> clone() const = 0;
    
    virtual double getLowerBoundForParamIndex(int idx) const = 0;
    virtual double getUpperBoundForParamIndex(int idx) const = 0;

};

} // namespace epidemic

#endif // I_NPI_STRATEGY_HPP
