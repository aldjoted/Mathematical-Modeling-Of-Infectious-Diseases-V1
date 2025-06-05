#ifndef INTERVENTION_CALLBACK_H
#define INTERVENTION_CALLBACK_H

#include "AgeSIRModel.hpp"
#include "exceptions/Exceptions.hpp"
#include "utils/Logger.hpp"
#include <memory>
#include <map>
#include <vector>
#include <string>
#include <Eigen/Dense>

namespace epidemic {

/**
 * @class InterventionCallback
 * @brief Manages and applies timed interventions to an AgeSIRModel during simulation.
 *
 * This class allows scheduling interventions (like contact reduction or mask mandates)
 * at specific time points. During a simulation run (typically via the `Simulator`),
 * the `applyScheduledInterventions` method should be called at each time step (or output time point)
 * to check for and apply any pending interventions to the associated `AgeSIRModel`.
 * It handles basic parameter validation for known intervention types and logs errors
 * encountered during application without halting the simulation.
 */
class InterventionCallback {
public:
    /**
     * @brief Constructs an intervention callback associated with a specific model.
     *
     * @param model Shared pointer to the AgeSIRModel instance to which interventions will be applied.
     *
     * @throws InvalidParameterException if the provided model pointer is null.
     */
    explicit InterventionCallback(std::shared_ptr<AgeSIRModel> model);

    /**
     * @brief Adds an intervention to the internal schedule.
     *
     * Performs basic validation on parameters for known intervention types (e.g., checking size and range).
     * Unknown intervention types are scheduled but rely on the `AgeSIRModel::applyIntervention` method for validation.
     * Multiple interventions can be scheduled for the same time point.
     *
     * @param time The simulation time at which the intervention should be applied (must be non-negative).
     * @param name A string identifying the type of intervention (e.g., "contact_reduction", "mask_mandate").
     * @param params An Eigen::VectorXd containing the parameters required for the specified intervention type.
     *
     * @throws InvalidParameterException if `time` is negative or if `params` are invalid for known intervention types
     *         (e.g., wrong size, out of range).
     * @throws InterventionException if adding the intervention fails due to other reasons (e.g., memory allocation).
     */
    void addIntervention(double time, const std::string& name, const Eigen::VectorXd& params);

    /**
     * @brief Applies all scheduled interventions up to the given simulation time `t`.
     *
     * Iterates through the schedule and calls `AgeSIRModel::applyIntervention` for any intervention
     * scheduled at a time greater than the last applied time and less than or equal to `t`.
     * Logs errors if the model rejects an intervention but continues processing others.
     * Updates the internal state to track the time up to which interventions have been applied.
     *
     * @param t The current simulation time.
     */
    void applyScheduledInterventions(double t);

    /**
     * @brief Resets the intervention schedule and applied status tracking.
     *
     * Clears all scheduled interventions and resets the last applied time tracker.
     * This is useful before starting a new simulation run using the same `InterventionCallback` instance.
     */
    void reset();

private:
    /** @brief Shared pointer to the model instance. */
    std::shared_ptr<AgeSIRModel> model_;

    /**
     * @brief Schedule of interventions, keyed by time.
     * Uses a multimap to allow multiple interventions at the exact same time.
     * The pair stores the intervention name and its parameters.
     */
    std::multimap<double, std::pair<std::string, Eigen::VectorXd>> intervention_schedule_;

    /**
     * @brief Tracks the simulation time up to which interventions have been applied.
     * Interventions scheduled at times > last_applied_time_ and <= current_time will be applied.
     */
    double last_applied_time_;

    /**
     * @brief Internal helper to validate intervention parameters based on known types.
     *
     * Checks parameter size and range constraints for interventions like "contact_reduction",
     * "social_distancing", "lockdown", "mask_mandate", "transmission_reduction".
     * Logs a warning for unknown types, deferring validation to the model.
     *
     * @param name The name/type of the intervention.
     * @param params The parameters associated with the intervention.
     *
     * @throws InvalidParameterException if parameters are invalid for a known intervention type.
     */
    void validateParameters(const std::string& name, const Eigen::VectorXd& params);
};

} // namespace epidemic

#endif // INTERVENTION_CALLBACK_H