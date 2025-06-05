#ifndef EPIDEMIC_MODEL_H
#define EPIDEMIC_MODEL_H

#include <vector>
#include <string>
#include <Eigen/Dense>
#include "interfaces/IEpidemicModel.hpp"

namespace epidemic {

/**
 * @class EpidemicModel
 * @brief Abstract base class for epidemic models using Boost.Odeint.
 * 
 * This class implements the IEpidemicModel interface and provides the necessary
 * operator() overload required by Boost.Odeint solvers. It delegates the actual
 * computation of the system's derivatives to the pure virtual `computeDerivatives`
 * method, which must be implemented by derived classes (e.g., AgeSIRModel).
 * This structure separates the Odeint integration mechanism from the specific
 * model equations.
 */
class EpidemicModel : public IEpidemicModel {
public:
    /**
     * @brief Virtual destructor for proper cleanup
     */
    virtual ~EpidemicModel() = default;

    /**
     * @brief Overload function call operator for Boost.Odeint compatibility.
     * 
     * This allows instances of derived model classes to be used directly as the
     * system function for Boost.Odeint integration routines (like `integrate_adaptive`).
     * It simply forwards the call to the virtual `computeDerivatives` method.
     * 
     * @param x The current state vector containing all compartment values.
     * @param dxdt The vector where the computed derivatives will be stored.
     * @param t The current simulation time.
     * @throws InvalidParameterException If state or derivative vectors have incorrect sizes.
     * @throws SimulationException If derivative computation encounters numerical issues.
     * @throws ModelException For other model-specific errors during computation.
     */
    virtual void operator()(const std::vector<double>& x, std::vector<double>& dxdt, double t) {
        // Delegate the computation to the virtual computeDerivatives method
        computeDerivatives(x, dxdt, t);
    }

    /**
     * @brief Returns the number of age classes in the model.
     * 
     * Derived classes must implement this to specify the age structure.
     * @return Number of age classes used in the model.
     */
    virtual int getNumAgeClasses() const = 0;

protected:
    /**
     * @brief Protected default constructor.
     * 
     * Ensures that EpidemicModel itself cannot be instantiated directly,
     * enforcing the use of derived classes that implement the model logic.
     */
    EpidemicModel() = default;

    /**
     * @brief Computes the derivatives of the epidemic model system.
     * 
     * This pure virtual function defines the system of ordinary differential equations (ODEs)
     * for a specific epidemic model. Derived classes must implement this method to provide
     * the model-specific equations (e.g., dS/dt, dI/dt, dR/dt for each age group).
     * 
     * @param state The current state vector of the system (e.g., [S0, S1, ..., I0, I1, ..., R0, R1, ...]).
     * @param derivatives The vector where the computed derivatives (rate of change for each state variable) should be stored.
     * @param time The current time t, which might be needed for time-dependent parameters.
     * @throws InvalidParameterException If state or derivatives vectors have incorrect size.
     * @throws SimulationException If derivative computation encounters numerical issues.
     * @throws ModelException For other model-specific errors during computation.
     */
    virtual void computeDerivatives(const std::vector<double>& state, std::vector<double>& derivatives, double time) = 0;
};

} // namespace epidemic
#endif // EPIDEMIC_MODEL_H