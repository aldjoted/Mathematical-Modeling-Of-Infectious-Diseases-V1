#ifndef FEHLBERG_SOLVER_STRATEGY_HPP
#define FEHLBERG_SOLVER_STRATEGY_HPP

#include "sir_age_structured/interfaces/IOdeSolverStrategy.hpp"
#include <boost/numeric/odeint.hpp>

namespace epidemic {

/**
 * @brief Concrete ODE solver strategy using Boost.Odeint's Runge-Kutta Fehlberg 7(8) method.
 * 
 * This class implements the IOdeSolverStrategy interface using the Runge-Kutta-Fehlberg 7(8)
 * method provided by Boost.Odeint. This is a high-order method offering improved accuracy 
 * compared to lower-order methods, making it suitable for problems requiring high precision.
 */
class FehlbergSolverStrategy : public IOdeSolverStrategy {
public:
    /**
     * @brief Integrates the given ODE system over specified time points.
     * 
     * Uses the Runge-Kutta-Fehlberg 7(8) method with adaptive step size control
     * to integrate the system of ordinary differential equations.
     * 
     * @param[in] system The ODE system to integrate, defined as a function that computes 
     *                   the derivatives of the state variables.
     * @param[in,out] initial_state The initial state vector which will be updated with the 
     *                              final state after integration.
     * @param[in] times Vector of time points at which the solution is required.
     * @param[in] dt_hint Suggested initial step size for the integrator.
     * @param[in] observer Function called at each time point with the current state and time.
     * @param[in] abs_error Absolute error tolerance for the adaptive step size control.
     * @param[in] rel_error Relative error tolerance for the adaptive step size control.
     * 
     * @throws SimulationException If the Boost.Odeint integration fails for any reason.
     */
    void integrate(
        const std::function<void(const state_type&, state_type&, double)>& system,
        state_type& initial_state,
        const std::vector<double>& times,
        double dt_hint,
        std::function<void(const state_type&, double)> observer,
        double abs_error,
        double rel_error) const override;
};
    
} // namespace epidemic

#endif // FEHLBERG_SOLVER_STRATEGY_HPP