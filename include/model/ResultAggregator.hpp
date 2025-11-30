#ifndef RESULT_AGGREGATOR_HPP
#define RESULT_AGGREGATOR_HPP

#include "model/interfaces/IResultAggregator.hpp"
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/extended_p_square_quantile.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/accumulators/statistics/count.hpp>

namespace epidemic {

/**
 * @brief Concrete implementation of IResultAggregator
 * 
 * Uses Boost.Accumulators for efficient statistical computations
 * and performs in-memory aggregation where possible.
 */
class ResultAggregator : public IResultAggregator {
public:
    ResultAggregator() = default;
    
    std::map<std::string, AggregatedStats> aggregateBatchMetrics(
        const std::vector<EssentialMetrics>& batch_metrics,
        int num_age_classes
    ) const override;
    
    std::map<std::string, AggregatedStats> aggregateAllBatches(
        const std::vector<std::map<std::string, AggregatedStats>>& all_batch_stats
    ) const override;
    
    PosteriorPredictiveData aggregatePosteriorPredictives(
        const std::vector<Eigen::VectorXd>& param_samples,
        IParameterManager& param_manager,
        ISimulationRunner& runner,
        IMetricsCalculator& calculator,
        int num_samples_for_ppc,
        const std::vector<double>& time_points,
        const Eigen::VectorXd& initial_state,
        const CalibrationData& observed_data,
        std::shared_ptr<AgeSEPAIHRDModel> model_template,
        unsigned int random_seed = 0
    ) const override;
    
    void aggregateTrajectoryFiles(
        const std::string& source_dir,
        const std::string& output_filepath,
        const std::vector<double>& time_points,
        IAnalysisWriter& writer
    ) const override;
    
    std::map<std::string, double> performENECOVIDValidation(
        const std::map<std::string, AggregatedStats>& summary,
        double ene_covid_target_day,
        double ene_covid_mean,
        double ene_covid_lower_ci,
        double ene_covid_upper_ci
    ) const override;
    
private:
    /**
     * @brief Helper to extract metric value by name
     * @param metrics Essential metrics structure
     * @param metric_name Name of metric to extract
     * @return Metric value
     */
    double extractMetricValue(
        const EssentialMetrics& metrics,
        const std::string& metric_name
    ) const;
};

} // namespace epidemic

#endif // RESULT_AGGREGATOR_HPP
