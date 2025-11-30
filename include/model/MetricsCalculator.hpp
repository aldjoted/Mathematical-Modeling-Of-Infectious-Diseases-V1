#ifndef METRICS_CALCULATOR_HPP
#define METRICS_CALCULATOR_HPP

#include "model/interfaces/IMetricsCalculator.hpp"

namespace epidemic {

/**
 * @brief Concrete implementation of IMetricsCalculator
 * 
 * This class performs pure, stateless calculations to transform simulation
 * results into specific metrics. All methods are const.
 */
class MetricsCalculator : public IMetricsCalculator {
public:
    MetricsCalculator() = default;
    
    EssentialMetrics calculateEssentialMetrics(
        const SimulationResult& sim_result,
        std::shared_ptr<AgeSEPAIHRDModel> model,
        const SEPAIHRDParameters& params,
        const Eigen::VectorXd& initial_state,
        const std::vector<double>& time_points
    ) const override;
    
    std::vector<double> calculateRtTrajectory(
        const SimulationResult& sim_result,
        std::shared_ptr<AgeSEPAIHRDModel> model,
        const std::vector<double>& time_points
    ) const override;
    
    std::vector<double> calculateSeroprevalenceTrajectory(
        const SimulationResult& sim_result,
        const SEPAIHRDParameters& params,
        const std::vector<double>& time_points,
        const Eigen::VectorXd& initial_state
    ) const override;
};

} // namespace epidemic

#endif // METRICS_CALCULATOR_HPP
