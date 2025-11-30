#ifndef DOPRI5_SOLVER_STRATEGY_HPP
#define DOPRI5_SOLVER_STRATEGY_HPP

#include "sir_age_structured/interfaces/IOdeSolverStrategy.hpp"
#include <boost/numeric/odeint.hpp>

namespace epidemic {

/**
 * @brief High-Performance Concrete ODE solver strategy using Boost.Odeint.
 * Marked final to encourage devirtualization.
 */
class Dopri5SolverStrategy final : public IOdeSolverStrategy {
public:
    // Default constructor
    Dopri5SolverStrategy() = default;
    
    /**
     * @brief Integrates the given ODE system.
     * Implementation is optimized to minimize overhead in the loop.
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

#endif // DOPRI5_SOLVER_STRATEGY_HPP