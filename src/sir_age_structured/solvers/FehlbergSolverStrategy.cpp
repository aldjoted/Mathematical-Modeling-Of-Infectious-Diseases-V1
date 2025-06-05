#include "sir_age_structured/solvers/FehlbergSolverStrategy.hpp"
#include "exceptions/Exceptions.hpp"
#include <boost/numeric/odeint/integrate/integrate_times.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_fehlberg78.hpp>

namespace epidemic {

void FehlbergSolverStrategy::integrate(
    const std::function<void(const state_type&, state_type&, double)>& system,
    state_type& initial_state,
    const std::vector<double>& times,
    double dt_hint,
    std::function<void(const state_type&, double)> observer,
    double abs_error,
    double rel_error) const
{
    try {
        boost::numeric::odeint::integrate_times(
            boost::numeric::odeint::make_controlled<boost::numeric::odeint::runge_kutta_fehlberg78<state_type>>(abs_error, rel_error),
            system,
            initial_state,
            times.begin(), times.end(),
            dt_hint,
            observer
        );
    } catch (const std::exception& e) {
        throw SimulationException("FehlbergSolverStrategy::integrate", "Boost.Odeint integration failed: " + std::string(e.what()));
    } catch (...) {
        throw SimulationException("FehlbergSolverStrategy::integrate", "Boost.Odeint integration failed with an unknown error.");
    }
}

} // namespace epidemic