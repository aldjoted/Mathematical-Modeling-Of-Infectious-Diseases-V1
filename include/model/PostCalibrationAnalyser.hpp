#ifndef POST_CALIBRATION_ANALYSER_H
#define POST_CALIBRATION_ANALYSER_H

#include <memory>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <Eigen/Dense>
#include "model/AgeSEPAIHRDModel.hpp"
#include "model/parameters/SEPAIHRDParameters.hpp"
#include "sir_age_structured/interfaces/IParameterManager.hpp"
#include "sir_age_structured/interfaces/IOdeSolverStrategy.hpp"
#include "sir_age_structured/SimulationResult.hpp"
#include "utils/GetCalibrationData.hpp"

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/median.hpp> // For tag::median if preferred over quantile(0.5)
#include <boost/accumulators/statistics/extended_p_square_quantile.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>


namespace epidemic {

/**
 * @brief Lightweight structure for essential metrics to reduce memory footprint
 */
struct EssentialMetrics {
    // Key scalar metrics
    double R0 = 0.0;
    double overall_IFR = 0.0;
    double overall_attack_rate = 0.0;
    double peak_hospital_occupancy = 0.0;
    double peak_ICU_occupancy = 0.0;
    double time_to_peak_hospital = 0.0;
    double time_to_peak_ICU = 0.0;
    double total_cumulative_deaths = 0.0;
    
    // Summary statistics for trajectories (instead of full trajectories)
    double max_Rt = 0.0;
    double min_Rt = 1e6;
    double final_Rt = 0.0;
    double seroprevalence_at_target_day = 0.0; // For ENE-COVID validation
    
    // Age-specific metrics (compact storage)
    std::vector<double> age_specific_IFR;
    std::vector<double> age_specific_IHR;
    std::vector<double> age_specific_IICUR;
    std::vector<double> age_specific_attack_rate;
    
    // NPI parameters
    std::map<std::string, double> kappa_values;
};

/**
 * @brief Structure for posterior predictive data with reduced memory footprint
 */
struct PosteriorPredictiveData {
    std::vector<double> time_points;
    
    struct IncidenceData {
        Eigen::MatrixXd median;
        Eigen::MatrixXd lower_90; // Corresponds to 0.05 quantile
        Eigen::MatrixXd upper_90; // Corresponds to 0.95 quantile
        Eigen::MatrixXd lower_95; // Corresponds to 0.025 quantile
        Eigen::MatrixXd upper_95; // Corresponds to 0.975 quantile
        Eigen::MatrixXd observed;
    };
    
    IncidenceData daily_hospitalizations;
    IncidenceData daily_icu_admissions;
    IncidenceData daily_deaths;
    IncidenceData cumulative_hospitalizations;
    IncidenceData cumulative_icu_admissions;
    IncidenceData cumulative_deaths;
};

/**
 * @brief Memory-optimized post-calibration analysis tool
 */
class PostCalibrationAnalyser {
public:
    PostCalibrationAnalyser(
        std::shared_ptr<AgeSEPAIHRDModel> model_template,
        std::shared_ptr<IOdeSolverStrategy> solver,
        const std::vector<double>& time_points,
        const Eigen::VectorXd& initial_state,
        const std::string& output_directory,
        const CalibrationData& observed_data
    );

    /**
     * @brief Generate full report with memory-efficient processing
     */
    void generateFullReport(
        const std::vector<Eigen::VectorXd>& param_samples,
        IParameterManager& param_manager,
        int num_samples_for_posterior_pred,
        int burn_in_for_summary = 0,
        int thinning_for_summary = 1,
        int batch_size = 50  // Process samples in batches
    );

    /**
     * @brief Analyze single run and extract essential metrics only
     */
    EssentialMetrics analyzeSingleRunLightweight(
        const SEPAIHRDParameters& params,
        const std::string& run_id = "single_run"
    );

    /**
     * @brief Memory-efficient MCMC analysis with batch processing
     */
    void analyzeMCMCRunsInBatches(
        const std::vector<Eigen::VectorXd>& param_samples,
        IParameterManager& param_manager,
        int burn_in = 0,
        int thinning = 1,
        int batch_size = 50
    );

    /**
     * @brief Generate posterior predictive checks with streaming approach
     */
    PosteriorPredictiveData generatePosteriorPredictiveChecksOptimized(
        const std::vector<Eigen::VectorXd>& param_samples,
        IParameterManager& param_manager,
        int num_samples_for_ppc
    );

    /**
     * @brief Perform scenario analysis with memory-efficient approach
     */
    void performScenarioAnalysisOptimized(
        const SEPAIHRDParameters& baseline_params,
        const std::vector<std::pair<std::string, SEPAIHRDParameters>>& scenarios
    );

    /**
     * @brief Validate against ENE-COVID with lightweight metrics
     */
    void validateAgainstENECOVID(
        const std::vector<EssentialMetrics>& all_metrics,
        double ene_covid_target_day = 64.0,
        double ene_covid_mean = 0.048,
        double ene_covid_lower_ci = 0.043,
        double ene_covid_upper_ci = 0.054
    );

    /**
     * @brief Save parameter posteriors with streaming approach
     */
    void saveParameterPosteriorsStreaming(
        const std::vector<Eigen::VectorXd>& param_samples,
        IParameterManager& param_manager,
        int burn_in = 0,
        int thinning = 1
    );

    /**
     * @brief Aggregate trajectory results from individual run files
     */
    void aggregateTrajectoryResults(
        const std::string& source_subdir,
        const std::string& output_filename
    );

private:
    std::shared_ptr<AgeSEPAIHRDModel> model_template_;
    std::shared_ptr<IOdeSolverStrategy> solver_strategy_;
    std::vector<double> time_points_;
    Eigen::VectorXd initial_state_;
    std::string output_dir_base_;
    CalibrationData observed_data_;
    int num_age_classes_;

    // Memory-efficient helpers
    void ensureOutputSubdirectoryExists(const std::string& subdir_name);
    
    void saveEssentialMetricsCSV(
        const std::string& filepath,
        const EssentialMetrics& metrics
    );
    
    void saveVectorToCSV(
        const std::string& filepath,
        const std::vector<double>& data
    );
    
    void saveMetricsSummaryFromBatches(
        const std::string& output_subdir
    );
    
    Eigen::MatrixXd calculateDailyIncidenceFlow(
        const SimulationResult& sim_result,
        const SEPAIHRDParameters& params,
        const std::string& type
    );
    
    void processAndSaveBatch(
        const std::vector<EssentialMetrics>& batch_metrics,
        int batch_index,
        const std::string& batch_subdir
    );
    
    void aggregateBatchResults(
        const std::string& batch_subdir,
        int num_batches
    );
    
    // Streaming file operations
    void appendToCSV(
        const std::string& filepath,
        const std::vector<std::string>& row_data,
        bool write_header = false
    );
    
    std::vector<double> computeQuantilesFromFile(
        const std::string& filepath,
        int column_index,
        const std::vector<double>& quantile_probs
    );

    // Added declarations for missing methods
    void savePosteriorPredictiveCheckData(
        const PosteriorPredictiveData& ppd_data,
        const std::string& sub_directory
    );

    void performENECOVIDValidationFromBatches(
        const std::string& batch_subdir,
        int num_batches
    );
    
    void calculateAndSaveTrajectories(
        const SimulationResult& sim_result,
        const SEPAIHRDParameters& params,
        int sample_idx
    );
};

} // namespace epidemic

#endif // POST_CALIBRATION_ANALYSER_H