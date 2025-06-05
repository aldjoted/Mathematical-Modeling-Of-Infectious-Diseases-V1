#ifndef AGE_SEPAIHRD_SIMULATOR_HPP
#define AGE_SEPAIHRD_SIMULATOR_HPP

#include "sir_age_structured/interfaces/IOdeSolverStrategy.hpp"
#include "sir_age_structured/Simulator.hpp"
#include "model/AgeSEPAIHRDModel.hpp"
#include <memory>
#include <Eigen/Dense>
#include <string>
namespace epidemic {

    /**
     * @brief Specialized simulator setup for the AgeSEPAIHRDModel.
     *
     * Extends the base Simulator class mainly by ensuring the correct model type
     * is used and potentially providing convenience methods specific to this model type.
     * Result processing is now handled externally by SimulationResultProcessor.
     */
    class AgeSEPAIHRDSimulator : public Simulator {
    public:
        /**
         * @brief Construct a new AgeSEPAIHRD Simulator object
         *
         * @param model Shared pointer to an AgeSEPAIHRDModel instance.
         * @param solver_strategy Shared pointer to the IOdeSolverStrategy to use.
         * @param start_time Simulation start time.
         * @param end_time Simulation end time.
         * @param time_step Output time step hint.
         * @param abs_error Absolute error tolerance (default: 1e-6).
         * @param rel_error Relative error tolerance (default: 1e-6).
         */
        AgeSEPAIHRDSimulator(std::shared_ptr<AgeSEPAIHRDModel> model,
                             std::shared_ptr<IOdeSolverStrategy> solver_strategy,
                             double start_time,
                             double end_time,
                             double time_step,
                             double abs_error = 1.0e-6,
                             double rel_error = 1.0e-6);
    
        // Removed getCompartmentData override - use SimulationResultProcessor::getCompartmentData instead
    
        /**
         * @brief Get the model specifically as an AgeSEPAIHRDModel pointer.
         *
         * Provides type-safe access to the specific model associated with this simulator setup.
         *
         * @return std::shared_ptr<AgeSEPAIHRDModel> Shared pointer to the model.
         *         Returns nullptr if the underlying model is not an AgeSEPAIHRDModel (should not happen with correct construction).
         */
        std::shared_ptr<AgeSEPAIHRDModel> getTypedModel() const;
    
        /**
         * @brief Convenience constant for the number of compartments in this specific model.
         */
        static constexpr int NUM_COMPARTMENTS = 9; // S, E, P, A, I, H, ICU, R, D
        /**
         * @brief Convenience constant for the offset of the D (Deaths) compartment in the state vector.
         * (0=S, 1=E, 2=P, 3=A, 4=I, 5=H, 6=ICU, 7=R, 8=D)
         */
        static constexpr int D_COMPARTMENT_OFFSET = 8; 
    };
    
} // namespace epidemic
    
#endif // AGE_SEPAIHRD_SIMULATOR_HPP