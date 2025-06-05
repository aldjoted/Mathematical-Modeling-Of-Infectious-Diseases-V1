#ifndef CASH_KARP_SOLVER_STRATEGY_HPP
#define CASH_KARP_SOLVER_STRATEGY_HPP

#include "sir_age_structured/interfaces/IOdeSolverStrategy.hpp"
#include <boost/numeric/odeint.hpp>

namespace epidemic {

    /**
     * @brief Concrete ODE solver strategy using Boost.Odeint's Runge-Kutta Cash-Karp 5(4) method.
     * 
     * This class implements the IOdeSolverStrategy interface using the Runge-Kutta Cash-Karp
     * 5(4) method, which is an adaptive step size control algorithm that provides error 
     * estimation. The implementation uses Boost.Odeint's functionality for numerical integration.
     */
    class CashKarpSolverStrategy : public IOdeSolverStrategy {
    public:
        /**
         * @brief Integrates a system of ordinary differential equations over time.
         * 
         * Uses the Runge-Kutta Cash-Karp 5(4) method with adaptive step size control
         * from the Boost.Odeint library to numerically integrate the provided system.
         * 
         * @param[in] system The function defining the system of ODEs.
         *                   Signature: void(const state_type&, state_type&, double)
         *                   First argument: current state
         *                   Second argument: derivatives (output)
         *                   Third argument: current time
         * @param[in,out] initial_state The initial state vector of the system, which will
         *                              be updated to contain the final state after integration.
         * @param[in] times Vector of time points at which the solution should be calculated.
         * @param[in] dt_hint Initial step size hint for the integrator.
         * @param[in] observer Function called at each output point with the current state and time.
         *                     Signature: void(const state_type&, double)
         * @param[in] abs_error Absolute error tolerance for the adaptive step size algorithm.
         * @param[in] rel_error Relative error tolerance for the adaptive step size algorithm.
         * 
         * @throws SimulationException If the Boost.Odeint integration fails, either with a
         *                             specific error message or with an unknown error.
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

#endif // CASH_KARP_SOLVER_STRATEGY_HPP