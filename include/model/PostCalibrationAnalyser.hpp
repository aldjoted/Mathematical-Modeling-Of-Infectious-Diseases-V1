#ifndef POST_CALIBRATION_ANALYSER_H
#define POST_CALIBRATION_ANALYSER_H

#include <memory>
#include <vector>
#include <string>
#include <map>
#include <Eigen/Dense>
#include "model/AgeSEPAIHRDModel.hpp"
#include "model/parameters/SEPAIHRDParameters.hpp"
#include "sir_age_structured/interfaces/IParameterManager.hpp"
#include "sir_age_structured/interfaces/IOdeSolverStrategy.hpp"
#include "sir_age_structured/SimulationResult.hpp"
#include "utils/GetCalibrationData.hpp" // For observed data access

namespace epidemic {

/**
 * @brief Structure to hold comprehensive metrics from epidemiological model simulations.
 * 
 * Contains both single-run metrics and aggregated results from multiple MCMC runs,
 * including reproduction numbers, age-specific severity metrics, and trajectory data.
 */
struct PostCalibrationMetrics {
    // Basic reproduction number
    double R0 = 0.0;

    // Time-varying reproduction number (median and CIs if aggregated)
    std::vector<double> Rt_time;
    std::vector<double> Rt_median;
    std::vector<double> Rt_lower_95; //  for aggregated results
    std::vector<double> Rt_upper_95; //  for aggregated results

    // Age-specific severity metrics (key: "age_0", "age_1", etc.)
    std::map<std::string, double> age_specific_IFR;   // Infection Fatality Rate
    std::map<std::string, double> age_specific_IHR;   // Infection Hospitalization Rate
    std::map<std::string, double> age_specific_IICUR; // Infection ICU Admission Rate
    std::map<std::string, double> age_specific_attack_rate;

    // Overall metrics
    double overall_IFR = 0.0;
    double overall_attack_rate = 0.0; // Cumulative infections / total population
    double peak_hospital_occupancy = 0.0;
    double peak_ICU_occupancy = 0.0;
    double time_to_peak_hospital = 0.0;
    double time_to_peak_ICU = 0.0;
    double total_cumulative_deaths = 0.0;

    // NPI impact parameters (Kappa values)
    std::map<std::string, double> kappa_values;

    // Seroprevalence proxy (cumulative infections / population)
    std::vector<double> seroprevalence_time;
    std::vector<double> seroprevalence_median;
    std::vector<double> seroprevalence_lower_95; //  for aggregated results
    std::vector<double> seroprevalence_upper_95; //  for aggregated results

    // Hidden dynamics (trajectories for E, P, A, I, R compartments)
    // Key: "E", "P", "A", "I", "R". Value: Matrix (time x age_groups)
    std::map<std::string, Eigen::MatrixXd> hidden_compartments_median;
    std::map<std::string, Eigen::MatrixXd> hidden_compartments_lower_95; //     std::map<std::string, Eigen::MatrixXd> hidden_compartments_upper_95; // 

        // Prevalence of active infection (P+A+I+H+ICU) / N
        // Key: "age_specific", "overall". Value: Matrix (time x age_groups or time x 1)
        std::map<std::string, Eigen::MatrixXd> prevalence_trajectories_median;
        std::map<std::string, Eigen::MatrixXd> prevalence_trajectories_lower_95;
        std::map<std::string, Eigen::MatrixXd> prevalence_trajectories_upper_95;
    };

/**
 * @brief Structure to hold posterior predictive check data for model validation.
 * 
 * Contains predicted incidence data with confidence intervals for comparison
 * against observed data to assess model fit quality.
 */
struct PosteriorPredictiveData {
    /** @brief Time points corresponding to the prediction data. */
    std::vector<double> time_points;

    /**
     * @brief Structure containing incidence data with confidence intervals.
     * 
     * Holds median predictions, confidence intervals, and observed data
     * for posterior predictive checks.
     */
    struct IncidenceData {
        Eigen::MatrixXd median;   // rows: time, cols: age groups
        Eigen::MatrixXd lower_90;
        Eigen::MatrixXd upper_90;
        Eigen::MatrixXd lower_95;
        Eigen::MatrixXd upper_95;
        Eigen::MatrixXd observed; // Actual data for comparison
    };

    IncidenceData daily_hospitalizations;
    IncidenceData daily_icu_admissions;
    IncidenceData daily_deaths;

    IncidenceData cumulative_hospitalizations;
    IncidenceData cumulative_icu_admissions;
    IncidenceData cumulative_deaths;
};

/**
 * @brief Comprehensive post-calibration analysis tool for epidemiological models.
 * 
 * Provides functionality to analyze MCMC parameter samples, generate posterior
 * predictive checks, perform scenario analysis, and validate model predictions
 * against external data sources.
 */
class PostCalibrationAnalyser {
public:
    /**
     * @brief Constructs a PostCalibrationAnalyser instance.
     * 
     * @param model_template Shared pointer to the epidemiological model template
     * @param solver Shared pointer to the ODE solver strategy
     * @param time_points Vector of time points for simulation
     * @param initial_state Initial state vector for the model
     * @param output_directory Base directory for saving analysis outputs
     * @param observed_data Calibration data containing observed values
     */
    PostCalibrationAnalyser(
        std::shared_ptr<AgeSEPAIHRDModel> model_template,
        std::shared_ptr<IOdeSolverStrategy> solver,
        const std::vector<double>& time_points,
        const Eigen::VectorXd& initial_state,
        const std::string& output_directory,
        const CalibrationData& observed_data
    );

    /**
     * @brief Generates a comprehensive analysis report from MCMC parameter samples.
     * 
     * @param param_samples Vector of parameter samples from MCMC
     * @param param_manager Parameter manager for converting samples to model parameters
     * @param num_samples_for_posterior_pred Number of samples to use for posterior predictive checks
     * @param burn_in_for_summary Number of initial samples to discard
     * @param thinning_for_summary Thinning interval for sample processing
     */
    void generateFullReport(
        const std::vector<Eigen::VectorXd>& param_samples,
        IParameterManager& param_manager,
        int num_samples_for_posterior_pred = 200, // Number of samples for PPC and some aggregations
        int burn_in_for_summary = 0,
        int thinning_for_summary = 1
    );

    /**
     * @brief Analyzes a single parameter set to compute epidemiological metrics.
     * 
     * @param params Model parameters for the simulation
     * @param run_id Identifier for this particular run
     * @return PostCalibrationMetrics structure containing computed metrics
     */
    PostCalibrationMetrics analyzeSingleRun(
        const SEPAIHRDParameters& params,
        const std::string& run_id = "single_run"
    );

    /**
     * @brief Analyzes multiple MCMC parameter samples to generate posterior distributions.
     * 
     * @param param_samples Vector of parameter samples from MCMC
     * @param param_manager Parameter manager for converting samples
     * @param num_samples_to_process Number of samples to process (-1 for all)
     * @param burn_in Number of initial samples to discard
     * @param thinning Thinning interval for sample processing
     * @param save_individual_mcmc_run_details Whether to save detailed files for each sample
     * @return Vector of metrics for each processed sample
     */
    std::vector<PostCalibrationMetrics> analyzeMCMCRuns(
        const std::vector<Eigen::VectorXd>& param_samples,
        IParameterManager& param_manager,
        int num_samples_to_process = -1, // -1 for all after burn-in/thinning
        int burn_in = 0,
        int thinning = 1,
        bool save_individual_mcmc_run_details = false // To control saving detailed files for each MCMC sample
    );

    /**
     * @brief Generates posterior predictive checks for model validation.
     * 
     * @param param_samples Vector of parameter samples from MCMC
     * @param param_manager Parameter manager for converting samples
     * @param num_samples_for_ppc Number of posterior samples to use
     * @return PosteriorPredictiveData structure with prediction intervals
     */
    PosteriorPredictiveData generatePosteriorPredictiveChecks(
        const std::vector<Eigen::VectorXd>& param_samples,
        IParameterManager& param_manager,
        int num_samples_for_ppc = 100 // Number of posterior samples to use for PPC
    );

    /**
     * @brief Performs scenario analysis comparing baseline to alternative scenarios.
     * 
     * @param baseline_params Baseline parameter set
     * @param scenarios Vector of scenario names and their corresponding parameters
     */
    void performScenarioAnalysis(
        const SEPAIHRDParameters& baseline_params,
        const std::vector<std::pair<std::string, SEPAIHRDParameters>>& scenarios
    );

    /**
     * @brief Validates model predictions against ENE-COVID seroprevalence study data.
     * 
     * @param all_run_metrics Vector of metrics from MCMC runs
     * @param ene_covid_target_day Target day for validation (default: day 64, May 4th 2020)
     * @param ene_covid_mean Mean seroprevalence from ENE-COVID study
     * @param ene_covid_lower_ci Lower confidence interval from ENE-COVID
     * @param ene_covid_upper_ci Upper confidence interval from ENE-COVID
     */
    void validateAgainstENECOVID(
        const std::vector<PostCalibrationMetrics>& all_run_metrics, // From analyzeMCMCRuns
        double ene_covid_target_day = 64.0, // Approx. May 4th, 2020
        double ene_covid_mean = 0.048,      // ENE-COVID overall prevalence for first round
        double ene_covid_lower_ci = 0.043,
        double ene_covid_upper_ci = 0.054
    );

    /**
     * @brief Saves parameter posterior distributions from MCMC samples to files.
     * 
     * @param param_samples Vector of parameter samples from MCMC
     * @param param_manager Parameter manager for parameter names and conversion
     * @param burn_in Number of initial samples to discard
     * @param thinning Thinning interval for sample processing
     */
    void saveParameterPosteriors(
        const std::vector<Eigen::VectorXd>& param_samples,
        IParameterManager& param_manager,
        int burn_in = 0,
        int thinning = 1
    );


private:
    std::shared_ptr<AgeSEPAIHRDModel> model_template_; // Used for structure and updated by param_manager
    std::shared_ptr<IOdeSolverStrategy> solver_strategy_;
    std::vector<double> time_points_;
    Eigen::VectorXd initial_state_;
    std::string output_dir_base_; // Base output directory
    CalibrationData observed_data_;
    int num_age_classes_;

    // Helper: Process simulation results into metrics
    PostCalibrationMetrics processSimulationResults(
        const SimulationResult& result,
        const SEPAIHRDParameters& params, // Parameters used for this specific run
        const std::string& run_id
    );
    
    // Helper: Save metrics and trajectories for aggregated MCMC results
    void saveAggregatedMCMCMetrics(
        const std::vector<PostCalibrationMetrics>& all_metrics,
        const std::string& sub_directory = "mcmc_aggregated"
    );

    // Helpers for saving various outputs
    void ensureOutputSubdirectoryExists(const std::string& subdir_name);
    void saveTimeSeriesCSV(
        const std::string& filepath,
        const std::vector<std::string>& headers,
        const std::vector<std::vector<double>>& data_columns // Each inner vector is a column
    );
    void saveMatrixTimeSeriesCSV(
        const std::string& filepath,
        const std::vector<double>& time_vector,
        const Eigen::MatrixXd& data_matrix, // Rows are time, cols are age groups/variables
        const std::vector<std::string>& data_column_base_names // e.g., "S", will become "S_age0", "S_age1"
    );
    void saveScalarMetricsCSV(
        const std::string& filepath,
        const PostCalibrationMetrics& metrics
    );
     void savePosteriorPredictiveCheckData(
        const PosteriorPredictiveData& ppd_data,
        const std::string& sub_directory = "posterior_predictive"
    );
     void saveScenarioComparisonCSV(
        const PostCalibrationMetrics& baseline_metrics,
        const std::vector<std::pair<std::string, PostCalibrationMetrics>>& scenario_results,
        const std::string& sub_directory = "scenarios"
    );


    // Calculation helpers
    Eigen::MatrixXd calculateDailyIncidenceFlow(
        const SimulationResult& sim_result,
        const SEPAIHRDParameters& params,
        const std::string& type // "hospitalizations", "icu", "deaths"
    );
    Eigen::MatrixXd calculateCumulativeFromDaily(const Eigen::MatrixXd& daily_incidence);
    std::vector<Eigen::MatrixXd> extractMatchingTrajectories(
        const std::vector<std::map<std::string, Eigen::MatrixXd>>& all_trajectories_maps,
        const std::string& key
    );
    
    template<typename T>
    std::vector<T> getQuantiles(const std::vector<T>& sorted_values, const std::vector<double>& quantiles_probs);

    Eigen::MatrixXd getQuantileMatrix(const std::vector<Eigen::MatrixXd>& matrix_samples, double quantile_prob);
    
    // For scenario analysis
    double getTotalCumulativeDeathsFromSimulation(const SEPAIHRDParameters& params);
    Eigen::VectorXd getOverallTrajectory(const SimulationResult& sim_result, int compartment_offset_multiplier);
};

} // namespace epidemic

#endif // POST_CALIBRATION_ANALYSER_H