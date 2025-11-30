#include "sir_age_structured/solvers/Dopri5SolverStrategy.hpp"
#include "exceptions/Exceptions.hpp"
#include <boost/numeric/odeint/integrate/integrate_times.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>
#include <boost/numeric/odeint/stepper/generation.hpp>

namespace epidemic {

void Dopri5SolverStrategy::integrate(
    const std::function<void(const state_type&, state_type&, double)>& system,
    state_type& initial_state,
    const std::vector<double>& times,
    double dt_hint,
    std::function<void(const state_type&, double)> observer,
    double abs_error,
    double rel_error) const
{
    using namespace boost::numeric::odeint;

    // Optimization: Avoid re-creating the stepper if possible, but the tolerance implies 
    // we need a controlled stepper specific to this call.
    // We use make_controlled to generate the stepper.
    // The key performance optimization here is ensuring 'system' is passed efficiently.
    // Since the interface forces std::function, we rely on the fact that the underlying 
    // Model::computeDerivatives is now vectorized.

    try {
        auto stepper = make_controlled<runge_kutta_dopri5<state_type>>(abs_error, rel_error);
        
        integrate_times(
            stepper,
            system,
            initial_state,
            times.begin(), times.end(),
            dt_hint,
            observer
        );
    } catch (const std::exception& e) {
        throw SimulationException("Dopri5SolverStrategy::integrate", "Boost.Odeint integration failed: " + std::string(e.what()));
    } catch (...) {
        throw SimulationException("Dopri5SolverStrategy::integrate", "Boost.Odeint integration failed with an unknown error.");
    }
}

} // namespace epidemic