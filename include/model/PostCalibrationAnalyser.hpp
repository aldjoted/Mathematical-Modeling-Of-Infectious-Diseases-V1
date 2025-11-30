#ifndef POST_CALIBRATION_ANALYSER_H
#define POST_CALIBRATION_ANALYSER_H

#include <memory>
#include <vector>
#include <string>
#include <map>
#include <Eigen/Dense>
#include "model/AnalysisTypes.hpp"
#include "model/AgeSEPAIHRDModel.hpp"
#include "model/parameters/SEPAIHRDParameters.hpp"
#include "model/interfaces/ISimulationRunner.hpp"
#include "model/interfaces/IMetricsCalculator.hpp"
#include "model/interfaces/IAnalysisWriter.hpp"
#include "model/interfaces/IResultAggregator.hpp"
#include "sir_age_structured/interfaces/IParameterManager.hpp"
#include "sir_age_structured/interfaces/IOdeSolverStrategy.hpp"
#include "utils/GetCalibrationData.hpp"

namespace epidemic {

/**
 * @brief High-level orchestrator for post-calibration analysis
 * 
 * This class coordinates modular components to execute the analysis workflow.
 * All heavy lifting is delegated to injected dependencies:
 * - ISimulationRunner: Runs simulations with caching
 * - IMetricsCalculator: Performs pure metric calculations
 * - IAnalysisWriter: Handles asynchronous file I/O
 * - IResultAggregator: Performs statistical aggregation
 */
class PostCalibrationAnalyser {
public:
    /**
     * @brief Constructor with dependency injection
     * @param model_template Shared pointer to model template
     * @param solver Shared pointer to ODE solver strategy
     * @param time_points Time points for simulation
     * @param initial_state Initial state vector
     * @param output_directory Output directory path
     * @param observed_data Observed calibration data
     * @param runner Unique pointer to simulation runner (with caching)
     * @param metrics_calculator Unique pointer to metrics calculator
     * @param writer Unique pointer to async writer
     * @param aggregator Unique pointer to result aggregator
     */
    PostCalibrationAnalyser(
        std::shared_ptr<AgeSEPAIHRDModel> model_template,
        std::shared_ptr<IOdeSolverStrategy> solver,
        const std::vector<double>& time_points,
        const Eigen::VectorXd& initial_state,
        const std::string& output_directory,
        const CalibrationData& observed_data,
        std::unique_ptr<ISimulationRunner> runner,
        std::unique_ptr<IMetricsCalculator> metrics_calculator,
        std::unique_ptr<IAnalysisWriter> writer,
        std::unique_ptr<IResultAggregator> aggregator
    );

    /**
     * @brief Generate full report with orchestrated workflow
     */
    void generateFullReport(
        const std::vector<Eigen::VectorXd>& param_samples,
        IParameterManager& param_manager,
        int num_samples_for_posterior_pred,
        int burn_in_for_summary = 0,
        int thinning_for_summary = 1,
        int batch_size = 50
    );

    /**
     * @brief Analyze single run and extract essential metrics
     */
    EssentialMetrics analyzeSingleRunLightweight(
        const SEPAIHRDParameters& params,
        const std::string& run_id = "single_run"
    );

    /**
     * @brief MCMC analysis with batch processing and in-memory aggregation
     */
    void analyzeMCMCRunsInBatches(
        const std::vector<Eigen::VectorXd>& param_samples,
        IParameterManager& param_manager,
        int burn_in = 0,
        int thinning = 1,
        int batch_size = 50
    );

    /**
     * @brief Generate posterior predictive checks
     */
    PosteriorPredictiveData generatePosteriorPredictiveChecksOptimized(
        const std::vector<Eigen::VectorXd>& param_samples,
        IParameterManager& param_manager,
        int num_samples_for_ppc
    );

    /**
     * @brief Perform scenario analysis
     */
    void performScenarioAnalysisOptimized(
        const SEPAIHRDParameters& baseline_params,
        const std::vector<std::pair<std::string, SEPAIHRDParameters>>& scenarios
    );


private:
    // Core dependencies (shared ownership)
    std::shared_ptr<AgeSEPAIHRDModel> model_template_;
    std::shared_ptr<IOdeSolverStrategy> solver_strategy_;
    
    // Injected components (unique ownership)
    std::unique_ptr<ISimulationRunner> runner_;
    std::unique_ptr<IMetricsCalculator> metrics_calculator_;
    std::unique_ptr<IAnalysisWriter> writer_;
    std::unique_ptr<IResultAggregator> aggregator_;
    
    // Configuration
    std::vector<double> time_points_;
    Eigen::VectorXd initial_state_;
    std::string output_dir_base_;
    CalibrationData observed_data_;
    int num_age_classes_;

    // Helper methods
    void ensureOutputSubdirectoryExists(const std::string& subdir_name);
};

} // namespace epidemic

#endif // POST_CALIBRATION_ANALYSER_H

