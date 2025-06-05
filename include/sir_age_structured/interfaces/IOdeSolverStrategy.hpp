#ifndef I_ODE_SOLVER_STRATEGY_HPP
#define I_ODE_SOLVER_STRATEGY_HPP

#include "../EpidemicModel.hpp"
#include <functional>

namespace epidemic {

// Type alias for the state vector used internally by Boost.Odeint
using state_type = std::vector<double>;

/**
 * @brief Interface for ODE integration strategies.
 *
 * Defines the contract for different numerical integration methods
 * that can be plugged into the Simulator.
 */
class IOdeSolverStrategy {
public:
    virtual ~IOdeSolverStrategy() = default;

    /**
     * @brief Integrates the ODE system over specified time points.
     *
     * @param system The function defining the ODE system (model derivatives).
     * @param initial_state The initial state vector.
     * @param times The vector of time points at which to record the solution.
     * @param dt_hint An initial step size hint for adaptive solvers.
     * @param observer A function called at each output time point to record the state.
     * @param abs_error Absolute error tolerance.
     * @param rel_error Relative error tolerance.
     *
     * @throws SimulationException If integration fails.
     */
    virtual void integrate(
        const std::function<void(const state_type&, state_type&, double)>& system,
        state_type& initial_state,
        const std::vector<double>& times,
        double dt_hint,
        std::function<void(const state_type&, double)> observer,
        double abs_error,
        double rel_error) const = 0;
};

} // namespace epidemic

#endif // I_ODE_SOLVER_STRATEGY_HPP
