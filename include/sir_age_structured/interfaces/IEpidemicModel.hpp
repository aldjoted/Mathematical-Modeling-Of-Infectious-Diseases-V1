#ifndef I_EPIDEMIC_MODEL_H
#define I_EPIDEMIC_MODEL_H

#include <vector>
#include <string>
#include <Eigen/Dense>

namespace epidemic {

/**
 * @class IEpidemicModel
 * @brief Interface for epidemic models with age-structured populations
 *
 * This abstract base class defines the interface that all epidemic models
 * must implement. It provides methods for:
 * - Computing model dynamics (differential equations)
 * - Applying interventions during simulation
 * - Managing model state and parameters
 *
 * Models implementing this interface can be used with the Simulator class
 * and support dynamic interventions during simulation.
 *
 */
class IEpidemicModel {
public:
    /**
     * @brief Virtual destructor for proper cleanup in derived classes
     */
    virtual ~IEpidemicModel() = default;
    
    /**
     * @brief Compute model derivatives at current state and time
     *
     * This method implements the system of ordinary differential equations (ODEs)
     * that define the model dynamics. The derivatives vector must be pre-allocated
     * with size matching the state vector.
     *
     * @param state Current values of all model variables [S0,S1,...,I0,I1,...,R0,R1,...]
     * @param derivatives Pre-allocated vector to store computed derivatives
     * @param time Current simulation time.
     *
     * @throws InvalidParameterException If state or derivatives vectors have incorrect size.
     * @throws SimulationException If derivative computation encounters numerical issues.
     * @throws ModelException For other model-specific errors during computation.
     */
    virtual void computeDerivatives(const std::vector<double>& state,
                                    std::vector<double>& derivatives,
                                    double time) = 0;

    /**
     * @brief Apply a named intervention with parameters at a specific time.
     *
     * @param name Name identifying the intervention type.
     * @param time Time at which the intervention is applied.
     * @param params Vector of parameters specific to the intervention.
     *
     * @throws InterventionException If the intervention name is unknown, parameters are invalid, or application fails.
     * @throws ModelException For other model-specific errors during intervention application.
     */
    virtual void applyIntervention(const std::string& name,
                                   double time,
                                   const Eigen::VectorXd& params) = 0;

    /**
     * @brief Reset model to initial parameter values
     *
     * Resets all model parameters to their initial values, allowing
     * the model to be reused for multiple simulations.
     */
    virtual void reset() = 0;
    
    /**
     * @brief Get the size of the model's state vector
     *
     * @return int Number of state variables in the model
     */
    virtual int getStateSize() const = 0;
    
    /**
     * @brief Get names of all state variables in the model
     *
     * @return std::vector<std::string> Vector of state variable names
     */
    virtual std::vector<std::string> getStateNames() const = 0;

    /**
     * @brief Get the number of age classes in the model.
     *
     * @return int Number of age groups
     */
    virtual int getNumAgeClasses() const = 0;
};

} // namespace epidemic

#endif // I_EPIDEMIC_MODEL_H
