#ifndef I_METRICS_CALCULATOR_HPP
#define I_METRICS_CALCULATOR_HPP

#include "model/AnalysisTypes.hpp"
#include "sir_age_structured/SimulationResult.hpp"
#include "model/AgeSEPAIHRDModel.hpp"
#include "model/parameters/SEPAIHRDParameters.hpp"
#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace epidemic {

/**
 * @brief Interface for calculating metrics from simulation results
 * 
 * This interface provides pure, stateless calculation methods to transform
 * simulation results into specific metrics.
 */
class IMetricsCalculator {
public:
    virtual ~IMetricsCalculator() = default;
    
    /**
     * @brief Calculate essential metrics from simulation result
     * @param sim_result Simulation result to analyze
     * @param model Model instance for R0/Rt calculation
     * @param params Model parameters
     * @param initial_state Initial state vector
     * @param time_points Time points from simulation
     * @return Essential metrics structure
     */
    virtual EssentialMetrics calculateEssentialMetrics(
        const SimulationResult& sim_result,
        std::shared_ptr<AgeSEPAIHRDModel> model,
        const SEPAIHRDParameters& params,
        const Eigen::VectorXd& initial_state,
        const std::vector<double>& time_points
    ) const = 0;
    
    /**
     * @brief Calculate effective reproduction number trajectory
     * @param sim_result Simulation result
     * @param model Model instance for Rt calculation
     * @param time_points Time points from simulation
     * @return Vector of Rt values over time
     */
    virtual std::vector<double> calculateRtTrajectory(
        const SimulationResult& sim_result,
        std::shared_ptr<AgeSEPAIHRDModel> model,
        const std::vector<double>& time_points
    ) const = 0;
    
    /**
     * @brief Calculate seroprevalence trajectory
     * @param sim_result Simulation result
     * @param params Model parameters
     * @param time_points Time points from simulation
     * @param initial_state Initial state vector
     * @return Vector of seroprevalence values over time
     */
    virtual std::vector<double> calculateSeroprevalenceTrajectory(
        const SimulationResult& sim_result,
        const SEPAIHRDParameters& params,
        const std::vector<double>& time_points,
        const Eigen::VectorXd& initial_state
    ) const = 0;
};

} // namespace epidemic

#endif // I_METRICS_CALCULATOR_HPP
