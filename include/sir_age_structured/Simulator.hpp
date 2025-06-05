#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "SimulationResult.hpp"
#include "EpidemicModel.hpp"
#include "exceptions/Exceptions.hpp"
#include "interfaces/IOdeSolverStrategy.hpp"
#include <memory>
#include <vector>
#include <Eigen/Dense>

namespace epidemic {

/**
 * @class Simulator
 * @brief Performs time‐stepping simulations of epidemic models using a configurable ODE solver strategy.
 *
 * This class takes an `EpidemicModel`, initial conditions, time points, and an `IOdeSolverStrategy`.
 * It orchestrates the simulation by calling the solver strategy's `integrate` method.
 * The returned `SimulationResult` contains:
 *  - `time_points`: the recorded times,
 *  - `solution`: the corresponding state vectors,
 *  - `compartment_names`: names for each compartment from the model.
 */
class Simulator {
public:
    /**
     * @brief Construct a new Simulator object.
     *
     * @param model_ Shared pointer to the `EpidemicModel` to be simulated.
     * @param solver_strategy_ Shared pointer to the `IOdeSolverStrategy` to use for integration.
     * @param start_time_ The initial time for the simulation range.
     * @param end_time_ The final time for the simulation range.
     * @param time_step_ A hint for the initial step size for adaptive solvers.
     * @param abs_error_ Absolute error tolerance (default: 1.0e-6).
     * @param rel_error_ Relative error tolerance (default: 1.0e-6).
     *
     * @throws InvalidParameterException If `model_` or `solver_strategy_` is null,
     *         `end_time_ <= start_time_`, `time_step_ <= 0`, or if `abs_error_`/`rel_error_` are negative.
     */
    Simulator(std::shared_ptr<EpidemicModel> model_,
              std::shared_ptr<IOdeSolverStrategy> solver_strategy_,
              double start_time_,
              double end_time_,
              double time_step_,
              double abs_error_ = 1.0e-6,
              double rel_error_ = 1.0e-6);

    /**
     * @brief Set new error tolerances for the adaptive ODE solver.
     *
     * @param abs_error_ New absolute error tolerance.
     * @param rel_error_ New relative error tolerance.
     *
     * @throws InvalidParameterException If either tolerance is negative.
     */
    void setErrorTolerance(double abs_error_, double rel_error_);

    /**
     * @brief Run the simulation and return the results.
     *
     * Integrates the model's ODEs using the configured solver strategy and tolerances,
     * recording the state exactly at each time in `output_time_points`.
     *
     * @param initial_state An `Eigen::VectorXd` of size `model->getStateSize()`.
     * @param output_time_points A strictly increasing vector of times within
     *                           [`start_time_`, `end_time_`].
     *
     * @return A `SimulationResult` with populated `time_points` and `solution`.
     *         The `compartment_names` can be obtained from the model associated with the result.
     *
     * @throws InvalidParameterException If dimensions mismatch, `output_time_points` is empty,
     *         not strictly increasing, or out of `[start_time_, end_time_]`.
     * @throws SimulationException If integration fails or post‐run checks (e.g.
     *         missing initial timepoint, count mismatches) detect an error.
     * @throws ModelException If `model->operator()` throws during integration.
     */
    SimulationResult run(const Eigen::VectorXd& initial_state,
                         const std::vector<double>& output_time_points);

    /**
     * @brief Access the underlying epidemic model.
     * @return Shared pointer to the `EpidemicModel`.
     */
    std::shared_ptr<EpidemicModel> getModel() const;

protected:
    ///< The epidemic model being simulated.
    std::shared_ptr<EpidemicModel>     model;
    ///< The ODE solver strategy.
    std::shared_ptr<IOdeSolverStrategy> solver_strategy;
    ///< Simulation start time.
    double                              start_time;
    ///< Simulation end time.
    double                              end_time;
    ///< Initial step‐size hint.
    double                              time_step;
    ///< Absolute error tolerance.
    double                              abs_error;
    ///< Relative error tolerance.
    double                              rel_error;
};

} // namespace epidemic

#endif // SIMULATOR_H
