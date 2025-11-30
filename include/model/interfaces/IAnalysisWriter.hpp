#ifndef I_ANALYSIS_WRITER_HPP
#define I_ANALYSIS_WRITER_HPP

#include "model/AnalysisTypes.hpp"
#include <string>
#include <vector>
#include <map>
#include <Eigen/Dense>

namespace epidemic {

/**
 * @brief Interface for asynchronous file I/O operations
 * 
 * All write operations are asynchronous and return immediately.
 * Use waitForCompletion() to block until all pending I/O is finished.
 */
class IAnalysisWriter {
public:
    virtual ~IAnalysisWriter() = default;
    
    /**
     * @brief Save a vector of doubles to CSV file (async)
     * @param filepath Output file path
     * @param data Vector of values to write
     */
    virtual void saveVectorToCSV(
        const std::string& filepath, 
        const std::vector<double>& data
    ) = 0;
    
    /**
     * @brief Save parameter posterior samples (async)
     * @param output_dir Output directory
     * @param param_samples Parameter sample vectors
     * @param param_names Parameter names
     * @param burn_in Number of samples to skip
     * @param thinning Thinning factor
     */
    virtual void saveParameterPosteriors(
        const std::string& output_dir,
        const std::vector<Eigen::VectorXd>& param_samples,
        const std::vector<std::string>& param_names,
        int burn_in,
        int thinning
    ) = 0;
    
    /**
     * @brief Save posterior predictive check data (async)
     * @param output_dir Output directory
     * @param ppd_data Posterior predictive data structure
     */
    virtual void savePosteriorPredictiveData(
        const std::string& output_dir,
        const PosteriorPredictiveData& ppd_data
    ) = 0;
    
    /**
     * @brief Save batch metrics to CSV (async)
     * @param filepath Output file path
     * @param batch_metrics Vector of metrics for this batch
     * @param num_age_classes Number of age classes
     */
    virtual void saveBatchMetrics(
        const std::string& filepath,
        const std::vector<EssentialMetrics>& batch_metrics,
        int num_age_classes
    ) = 0;
    
    /**
     * @brief Save aggregated summary statistics (async)
     * @param filepath Output file path
     * @param summary Map of metric names to aggregated stats
     */
    virtual void saveAggregatedSummary(
        const std::string& filepath,
        const std::map<std::string, AggregatedStats>& summary
    ) = 0;
    
    /**
     * @brief Save scenario comparison results (async)
     * @param filepath Output file path
     * @param scenarios Vector of (scenario_name, metrics) pairs
     */
    virtual void saveScenarioComparison(
        const std::string& filepath,
        const std::vector<std::pair<std::string, EssentialMetrics>>& scenarios
    ) = 0;
    
    /**
     * @brief Save ENE-COVID validation data (async)
     * @param filepath Output file path
     * @param ene_covid_data Validation data map
     */
    virtual void saveEneCovidValidation(
        const std::string& filepath,
        const std::map<std::string, double>& ene_covid_data
    ) = 0;
    
    /**
     * @brief Save aggregated trajectory with uncertainty (async)
     * @param filepath Output file path
     * @param time_points Time points vector
     * @param aggregated_data Map of time -> aggregated stats
     */
    virtual void saveAggregatedTrajectory(
        const std::string& filepath,
        const std::vector<double>& time_points,
        const std::map<double, AggregatedStats>& aggregated_data
    ) = 0;
    
    /**
     * @brief Block until all pending I/O operations complete
     */
    virtual void waitForCompletion() = 0;
    
    /**
     * @brief Get number of pending write operations
     * @return Number of tasks in queue
     */
    virtual size_t getPendingTaskCount() const = 0;
};

} // namespace epidemic

#endif // I_ANALYSIS_WRITER_HPP
