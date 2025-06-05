#ifndef SIMULATION_RESULT_HPP
#define SIMULATION_RESULT_HPP

#include <vector>
#include <Eigen/Dense>

namespace epidemic {

    /** @brief Type alias for the state vector, representing the values of all compartments at a single time point. */
    using state_type = std::vector<double>;
    
    /**
     * @brief Holds the results of an epidemic simulation.
     *
     * Contains the time points at which the solution was recorded and the
     * corresponding state vectors.
     */
    struct SimulationResult {
        /** @brief Vector storing the time points at which the solution was recorded. */
        std::vector<double> time_points;
    
        /** @brief Vector storing the state vectors (as std::vector<double>) at each recorded time point. */
        std::vector<state_type> solution;
    
        /** @brief Default constructor */
        SimulationResult() = default;
    
        /**
         * @brief Checks if the result object contains valid data.
         * @return true if time_points and solution are not empty and have matching sizes, false otherwise.
         */
        bool isValid() const {
            return !time_points.empty() && !solution.empty() && time_points.size() == solution.size();
        }
    };
    
    } // namespace epidemic

#endif // SIMULATION_RESULT_HPP