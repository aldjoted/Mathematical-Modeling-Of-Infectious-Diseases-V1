#ifndef I_RESULT_AGGREGATOR_HPP
#define I_RESULT_AGGREGATOR_HPP

#include "model/AnalysisTypes.hpp"
#include "model/interfaces/ISimulationRunner.hpp"
#include "model/interfaces/IMetricsCalculator.hpp"
#include "model/interfaces/IAnalysisWriter.hpp"
#include "sir_age_structured/interfaces/IParameterManager.hpp"
#include "utils/GetCalibrationData.hpp"
#include <string>
#include <vector>
#include <map>
#include <Eigen/Dense>

namespace epidemic {

/**
 * @brief Interface for statistical aggregation of results
 * 
 * This interface provides methods for in-memory aggregation of batch metrics
 * and file-based aggregation of trajectory data.
 */
class IResultAggregator {
public:
    virtual ~IResultAggregator() = default;
    
    /**
     * @brief Aggregate a batch of metrics in memory
     * @param batch_metrics Vector of metrics to aggregate
     * @param num_age_classes Number of age classes
     * @return Map of metric names to aggregated stats
     */
    virtual std::map<std::string, AggregatedStats> aggregateBatchMetrics(
        const std::vector<EssentialMetrics>& batch_metrics,
        int num_age_classes
    ) const = 0;
    
    /**
     * @brief Aggregate statistics from multiple batches
     * @param all_batch_stats Vector of batch-level aggregated stats
     * @return Final aggregated statistics across all batches
     */
    virtual std::map<std::string, AggregatedStats> aggregateAllBatches(
        const std::vector<std::map<std::string, AggregatedStats>>& all_batch_stats
    ) const = 0;
    
    /**
     * @brief Generate posterior predictive checks
     * @param param_samples Parameter samples from MCMC
     * @param param_manager Parameter manager for updating model
     * @param runner Simulation runner (with caching)
     * @param calculator Metrics calculator
     * @param num_samples_for_ppc Number of samples to use
     * @param time_points Time points for simulation
     * @param initial_state Initial state vector
     * @param observed_data Observed calibration data
     * @param model_template Model template for simulation
     * @param random_seed Optional seed for reproducibility (0 = random)
     * @return Posterior predictive data structure
     */
    virtual PosteriorPredictiveData aggregatePosteriorPredictives(
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
    ) const = 0;
    
    /**
     * @brief Aggregate trajectory files from a directory
     * @param source_dir Directory containing trajectory CSV files
     * @param output_filepath Output file for aggregated trajectory
     * @param time_points Time points vector
     * @param writer Writer interface for async output
     */
    virtual void aggregateTrajectoryFiles(
        const std::string& source_dir,
        const std::string& output_filepath,
        const std::vector<double>& time_points,
        IAnalysisWriter& writer
    ) const = 0;
    
    /**
     * @brief Perform ENE-COVID seroprevalence validation
     * @param summary Aggregated summary statistics
     * @param ene_covid_target_day Day for comparison (e.g., 64 for May 4)
     * @param ene_covid_mean ENE-COVID mean seroprevalence
     * @param ene_covid_lower_ci ENE-COVID lower 95% CI
     * @param ene_covid_upper_ci ENE-COVID upper 95% CI
     * @return Map of validation data
     */
    virtual std::map<std::string, double> performENECOVIDValidation(
        const std::map<std::string, AggregatedStats>& summary,
        double ene_covid_target_day,
        double ene_covid_mean,
        double ene_covid_lower_ci,
        double ene_covid_upper_ci
    ) const = 0;
};

} // namespace epidemic

#endif // I_RESULT_AGGREGATOR_HPP
