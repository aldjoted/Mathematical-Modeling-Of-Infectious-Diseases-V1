#ifndef SIMULATION_RESULT_PROCESSOR_HPP
#define SIMULATION_RESULT_PROCESSOR_HPP

#include "SimulationResult.hpp"
#include "sir_age_structured/interfaces/IEpidemicModel.hpp"
#include <string>
#include <Eigen/Dense>
#include <vector>

namespace epidemic {

    /**
     * @class SimulationResultProcessor
     * @brief Provides utility functions to process and export SimulationResult data.
     *
     * Contains static methods for tasks like extracting compartment data and saving
     * results to files, decoupling this logic from the Simulator class.
     */
    class SimulationResultProcessor {
    public:
        /** @brief Deleted default constructor to enforce static utility class behavior. */
        SimulationResultProcessor() = delete;
    
        /**
         * @brief Extract the simulation data for a specific named compartment across all time points.
         *
         * Retrieves the values for the specified compartment from the simulation results
         * and returns them as an Eigen::MatrixXd, where each row corresponds to a time point
         * and columns correspond to age groups (or a single column if not age-structured).
         *
         * @param result The SimulationResult object containing the simulation output.
         * @param model A reference to the IEpidemicModel used in the simulation (needed for state names and structure).
         * @param compartment The name of the compartment (e.g., "S", "I", "R", "E", "H", etc.). Case-insensitive matching is attempted.
         * @param num_compartments_per_age The total number of compartments per age group in the model (e.g., 3 for SIR, 9 for SEPAIHRD).
         * @return Eigen::MatrixXd A matrix containing the compartment's data over time (rows) and across age groups (columns).
         *
         * @throws InvalidResultException If the `result` object is invalid or empty.
         * @throws InvalidParameterException If the `compartment` name is not valid for the model structure,
         *         or if the state size in the results is inconsistent with the expected model structure (num_compartments).
         * @throws ModelException If accessing state information from the model fails.
         * @throws SimulationException If there's an inconsistency in the result data (e.g., state vector size mismatch).
         */
        static Eigen::MatrixXd getCompartmentData(const SimulationResult& result,
                                                  const IEpidemicModel& model,
                                                  const std::string& compartment,
                                                  int num_compartments_per_age);
        
        /**
         * @brief Calculate incidence (new infections) from simulation results.
         *
         * Extracts the flow from S to I compartments at each time point based on the
         * model's force of infection calculation, representing new infections.
         * This method is designed specifically for SIR-type models and requires access
         * to the AgeSIRModel's methods including population sizes, transmissibility,
         * and contact matrix.
         *
         * @param result The SimulationResult object containing the simulation output.
         * @param model A reference to the IEpidemicModel used in the simulation, expected to be castable to AgeSIRModel.
         * @param num_compartments_per_age The total number of compartments per age group in the model (not used in current implementation).
         * @return Eigen::MatrixXd A matrix containing incidence data over time (rows) and across age groups (columns).
         *
         * @throws InvalidResultException If the `result` object is invalid or empty.
         * @throws ModelException If the model cannot be cast to AgeSIRModel or if accessing model information fails.
         */
        static Eigen::MatrixXd getIncidenceData(
            const SimulationResult& result,
            const IEpidemicModel& model,
            int num_compartments_per_age);
    
        /**
         * @brief Save the full simulation results (time points and all compartment states) to a CSV file.
         *
         * Writes the recorded time points and the corresponding state vectors from the SimulationResult
         * to the specified file. The first column will be "Time", followed by columns for each state
         * variable using names obtained from the model (e.g., "S_0", "S_1", ..., "I_0", "I_1", ...).
         *
         * @param result The SimulationResult object containing the simulation output.
         * @param model A reference to the IEpidemicModel used in the simulation (needed for state names).
         * @param filename The path and name of the output CSV file.
         *
         * @throws InvalidResultException If the `result` object is invalid or empty.
         * @throws FileIOException If the file specified by `filename` cannot be opened for writing.
         * @note If fetching state names from the model fails, a warning is printed to stderr and generic headers are used.
         */
        static void saveResultsToCSV(const SimulationResult& result,
                                     const IEpidemicModel& model,
                                     const std::string& filename);
    };
    
} // namespace epidemic
    
#endif // SIMULATION_RESULT_PROCESSOR_HPP