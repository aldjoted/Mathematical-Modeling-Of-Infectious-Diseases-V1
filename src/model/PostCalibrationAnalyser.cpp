#include "model/PostCalibrationAnalyser.hpp"
#include "model/AgeSEPAIHRDsimulator.hpp"
#include "model/ReproductionNumberCalculator.hpp"
#include "model/PieceWiseConstantNPIStrategy.hpp"
#include "model/parameters/SEPAIHRDParameterManager.hpp"
#include "utils/FileUtils.hpp"
#include "utils/Logger.hpp"
#include <algorithm>
#include <numeric>

namespace epidemic {

PostCalibrationAnalyser::PostCalibrationAnalyser(
    std::shared_ptr<AgeSEPAIHRDModel> model_template,
    std::shared_ptr<IOdeSolverStrategy> solver,
    const std::vector<double>& time_points,
    const Eigen::VectorXd& initial_state,
    const std::string& output_directory,
    const CalibrationData& observed_data,
    std::unique_ptr<ISimulationRunner> runner,
    std::unique_ptr<IMetricsCalculator> metrics_calculator,
    std::unique_ptr<IAnalysisWriter> writer,
    std::unique_ptr<IResultAggregator> aggregator)
    : model_template_(model_template),
      solver_strategy_(solver),
      runner_(std::move(runner)),
      metrics_calculator_(std::move(metrics_calculator)),
      writer_(std::move(writer)),
      aggregator_(std::move(aggregator)),
      time_points_(time_points),
      initial_state_(initial_state),
      output_dir_base_(output_directory),
      observed_data_(observed_data) {

    if (!model_template_) throw std::invalid_argument("PostCalibrationAnalyser: Model template cannot be null");
    if (!solver_strategy_) throw std::invalid_argument("PostCalibrationAnalyser: Solver strategy cannot be null");
    if (!runner_) throw std::invalid_argument("PostCalibrationAnalyser: SimulationRunner cannot be null");
    if (!metrics_calculator_) throw std::invalid_argument("PostCalibrationAnalyser: MetricsCalculator cannot be null");
    if (!writer_) throw std::invalid_argument("PostCalibrationAnalyser: AnalysisWriter cannot be null");
    if (!aggregator_) throw std::invalid_argument("PostCalibrationAnalyser: ResultAggregator cannot be null");
    if (time_points_.empty()) throw std::invalid_argument("PostCalibrationAnalyser: Time points cannot be empty");
    if (initial_state_.size() == 0) throw std::invalid_argument("PostCalibrationAnalyser: Initial state cannot be empty");
    if (initial_state_.size() != model_template_->getStateSize()) {
        throw std::invalid_argument("PostCalibrationAnalyser: Initial state size does not match model state size");
    }

    num_age_classes_ = model_template_->getNumAgeClasses();
    FileUtils::ensureDirectoryExists(output_dir_base_);
    Logger::getInstance().info("PostCalibrationAnalyser", "Orchestrator initialized with output directory: " + output_dir_base_);
}


void PostCalibrationAnalyser::generateFullReport(
    const std::vector<Eigen::VectorXd>& param_samples,
    IParameterManager& param_manager,
    int num_samples_for_posterior_pred,
    int burn_in_for_summary,
    int thinning_for_summary,
    int batch_size) {

    Logger::getInstance().info("PostCalibrationAnalyser", "Starting orchestrated full report generation...");

    // 1. Posterior Predictive Checks
    ensureOutputSubdirectoryExists("posterior_predictive");
    Logger::getInstance().info("PostCalibrationAnalyser", "Generating Posterior Predictive Checks...");
    PosteriorPredictiveData ppd_data = generatePosteriorPredictiveChecksOptimized(
        param_samples, param_manager, num_samples_for_posterior_pred
    );
    
    // Save PPC data asynchronously
    writer_->savePosteriorPredictiveData(
        FileUtils::joinPaths(output_dir_base_, "posterior_predictive"),
        ppd_data
    );
    
    // Clear PPC data from memory
    ppd_data = PosteriorPredictiveData();
    
    // 2. MCMC Analysis in batches (includes trajectory aggregation)
    Logger::getInstance().info("PostCalibrationAnalyser", "Analyzing MCMC runs in batches...");
    analyzeMCMCRunsInBatches(param_samples, param_manager, burn_in_for_summary, thinning_for_summary, batch_size);
    
    // 3. Parameter Posteriors
    ensureOutputSubdirectoryExists("parameter_posteriors");
    Logger::getInstance().info("PostCalibrationAnalyser", "Saving parameter posteriors...");
    writer_->saveParameterPosteriors(
        FileUtils::joinPaths(output_dir_base_, "parameter_posteriors"),
        param_samples,
        param_manager.getParameterNames(),
        burn_in_for_summary,
        thinning_for_summary
    );
    
    // 4. Scenario Analysis
    ensureOutputSubdirectoryExists("scenarios");
    if (!param_samples.empty()) {
        Logger::getInstance().info("PostCalibrationAnalyser", "Performing scenario analysis...");
        
        // Compute mean parameters
        Eigen::VectorXd mean_params = Eigen::VectorXd::Zero(param_manager.getParameterCount());
        int count = 0;
        for (size_t i = burn_in_for_summary; i < param_samples.size(); i += thinning_for_summary) {
            mean_params += param_samples[i];
            count++;
        }
        if (count > 0) mean_params /= count;
        
        param_manager.updateModelParameters(mean_params);
        SEPAIHRDParameters baseline_params = model_template_->getModelParameters();
        
        // Define scenarios modifying the first calibratable NPI parameter
        std::vector<std::pair<std::string, SEPAIHRDParameters>> scenarios;
        
        auto npi_strategy = model_template_->getNpiStrategy();
        size_t first_modifiable_idx = 1;
        
        auto piecewise_npi = std::dynamic_pointer_cast<PiecewiseConstantNpiStrategy>(npi_strategy);
        if (piecewise_npi && !piecewise_npi->isBaselineFixed()) {
            first_modifiable_idx = 0;
        }
        
        if (baseline_params.kappa_values.size() > first_modifiable_idx) {
            SEPAIHRDParameters stricter_params = baseline_params;
            stricter_params.kappa_values[first_modifiable_idx] *= 0.9;
            scenarios.push_back({"stricter_lockdown", stricter_params});
            
            SEPAIHRDParameters weaker_params = baseline_params;
            weaker_params.kappa_values[first_modifiable_idx] *= 1.1;
            scenarios.push_back({"weaker_lockdown", weaker_params});
            
            Logger::getInstance().info("PostCalibrationAnalyser", 
                "Scenario analysis modifying kappa index " + std::to_string(first_modifiable_idx) + 
                " (baseline value: " + std::to_string(baseline_params.kappa_values[first_modifiable_idx]) + ")");
        } else {
            Logger::getInstance().warning("PostCalibrationAnalyser", 
                "Not enough kappa values for scenario analysis. kappa_values size: " + 
                std::to_string(baseline_params.kappa_values.size()));
        }
        
        performScenarioAnalysisOptimized(baseline_params, scenarios);
    }
    
    // 5. Wait for all async I/O to complete
    Logger::getInstance().info("PostCalibrationAnalyser", "Waiting for all async I/O to complete...");
    writer_->waitForCompletion();
    
    Logger::getInstance().info("PostCalibrationAnalyser", "Full report generation completed.");
}

EssentialMetrics PostCalibrationAnalyser::analyzeSingleRunLightweight(
    const SEPAIHRDParameters& params,
    const std::string& run_id) {
    
    Logger::getInstance().debug("PostCalibrationAnalyser", "Analyzing run: " + run_id);
    
    SimulationResult sim_result = runner_->runSimulation(params, initial_state_, time_points_);
    
    if (!sim_result.isValid()) {
        Logger::getInstance().warning("PostCalibrationAnalyser", 
            "Invalid simulation result for run: " + run_id);
    }
    
    auto run_npi_strategy = model_template_->getNpiStrategy()->clone();
    auto run_model = std::make_shared<AgeSEPAIHRDModel>(params, run_npi_strategy);
    
    EssentialMetrics metrics = metrics_calculator_->calculateEssentialMetrics(
        sim_result, run_model, params, initial_state_, time_points_
    );
    
    return metrics;
}

void PostCalibrationAnalyser::analyzeMCMCRunsInBatches(
    const std::vector<Eigen::VectorXd>& param_samples,
    IParameterManager& param_manager,
    int burn_in,
    int thinning,
    int batch_size) {
    
    if (param_samples.empty()) {
        Logger::getInstance().warning("PostCalibrationAnalyser", "No MCMC samples provided.");
        return;
    }
    
    int total_samples = param_samples.size();
    int effective_start = burn_in;
    if (effective_start >= total_samples) {
        Logger::getInstance().warning("PostCalibrationAnalyser", "Burn-in too large for sample size.");
        return;
    }
    
    Logger::getInstance().info("PostCalibrationAnalyser", "Starting batch processing with in-memory aggregation...");
    
    ensureOutputSubdirectoryExists("mcmc_batches");
    ensureOutputSubdirectoryExists("mcmc_aggregated");
    
    std::vector<EssentialMetrics> batch_metrics_buffer;
    batch_metrics_buffer.reserve(batch_size);
    std::vector<std::map<std::string, AggregatedStats>> all_batch_stats;
    
    // In-memory trajectory accumulators (trading memory for I/O efficiency)
    std::vector<std::vector<double>> all_rt_trajectories;
    std::vector<std::vector<double>> all_sero_trajectories;
    all_rt_trajectories.reserve(total_samples - effective_start);
    all_sero_trajectories.reserve(total_samples - effective_start);

    int batch_count = 0;
    int processed_samples = 0;
    
    for (int i = effective_start; i < total_samples; i += thinning) {
        // Update model with current sample parameters
        auto* sepaihrd_pm = dynamic_cast<SEPAIHRDParameterManager*>(&param_manager);
        if (sepaihrd_pm) {
            sepaihrd_pm->updateModelParameters(param_samples[i], model_template_);
        } else {
            param_manager.updateModelParameters(param_samples[i]);
        }
        
        SEPAIHRDParameters params = model_template_->getModelParameters();
        
        SimulationResult sim_result = runner_->runSimulation(params, initial_state_, time_points_);
        
        if (!sim_result.isValid()) {
            Logger::getInstance().warning("PostCalibrationAnalyser", 
                "Invalid simulation at sample " + std::to_string(i));
            continue;
        }
        
        auto run_npi_strategy = model_template_->getNpiStrategy()->clone();
        auto run_model = std::make_shared<AgeSEPAIHRDModel>(params, run_npi_strategy);
        
        EssentialMetrics metrics = metrics_calculator_->calculateEssentialMetrics(
            sim_result, run_model, params, initial_state_, time_points_
        );
        batch_metrics_buffer.push_back(metrics);
        
        auto rt_trajectory = metrics_calculator_->calculateRtTrajectory(
            sim_result, run_model, time_points_
        );
        all_rt_trajectories.push_back(std::move(rt_trajectory));
        
        auto sero_trajectory = metrics_calculator_->calculateSeroprevalenceTrajectory(
            sim_result, params, time_points_, initial_state_
        );
        all_sero_trajectories.push_back(std::move(sero_trajectory));
        
        processed_samples++;
        
        if (batch_metrics_buffer.size() >= static_cast<size_t>(batch_size) || 
            i + thinning >= total_samples) {
            
            writer_->saveBatchMetrics(
                FileUtils::joinPaths(output_dir_base_, 
                    "mcmc_batches/batch_" + std::to_string(batch_count) + ".csv"),
                batch_metrics_buffer,
                num_age_classes_
            );
            
            auto batch_stats = aggregator_->aggregateBatchMetrics(
                batch_metrics_buffer, num_age_classes_
            );
            all_batch_stats.push_back(batch_stats);
            
            Logger::getInstance().info("PostCalibrationAnalyser", 
                "Completed batch " + std::to_string(batch_count) + 
                " (" + std::to_string(processed_samples) + " samples processed)");
            
            batch_metrics_buffer.clear();
            batch_count++;
        }
        
        if (processed_samples % 10 == 0) {
            Logger::getInstance().info("PostCalibrationAnalyser", 
                "Processed " + std::to_string(processed_samples) + " samples...");
        }
    }
    
    // Final aggregation across all batches
    Logger::getInstance().info("PostCalibrationAnalyser", 
        "Aggregating results from " + std::to_string(batch_count) + " batches...");
    auto final_summary = aggregator_->aggregateAllBatches(all_batch_stats);
    
    writer_->saveAggregatedSummary(
        FileUtils::joinPaths(output_dir_base_, "mcmc_aggregated/metrics_summary.csv"),
        final_summary
    );
    
    ensureOutputSubdirectoryExists("seroprevalence");
    auto ene_covid_data = aggregator_->performENECOVIDValidation(
        final_summary,
        64.0,
        0.048,
        0.043,
        0.054
    );
    writer_->saveEneCovidValidation(
        FileUtils::joinPaths(output_dir_base_, "seroprevalence/ene_covid_validation.csv"),
        ene_covid_data
    );
    
    ensureOutputSubdirectoryExists("rt_trajectories");
    
    auto aggregateAndSaveTrajectory = [&](const std::vector<std::vector<double>>& trajectories, const std::string& filepath) {
        if (trajectories.empty()) return;
        
        size_t num_timesteps = trajectories[0].size();
        std::map<double, AggregatedStats> aggregated_data;
        std::vector<double> probs = {0.025, 0.05, 0.5, 0.95, 0.975};
        
        for (size_t t = 0; t < num_timesteps && t < time_points_.size(); ++t) {
            std::vector<double> values_at_t;
            values_at_t.reserve(trajectories.size());
            for (const auto& traj : trajectories) {
                if (t < traj.size()) values_at_t.push_back(traj[t]);
            }
            
            if (!values_at_t.empty()) {
                std::sort(values_at_t.begin(), values_at_t.end());
                AggregatedStats stats;
                auto get_quantile = [&](double q) {
                    double pos = q * (values_at_t.size() - 1);
                    size_t idx = static_cast<size_t>(pos);
                    double frac = pos - idx;
                    if (idx + 1 < values_at_t.size()) {
                        return values_at_t[idx] * (1.0 - frac) + values_at_t[idx + 1] * frac;
                    }
                    return values_at_t[idx];
                };
                
                stats["median"] = get_quantile(0.5);
                stats["q025"] = get_quantile(0.025);
                stats["q975"] = get_quantile(0.975);
                stats["q05"] = get_quantile(0.05);
                stats["q95"] = get_quantile(0.95);
                
                aggregated_data[time_points_[t]] = stats;
            }
        }
        writer_->saveAggregatedTrajectory(filepath, time_points_, aggregated_data);
    };

    aggregateAndSaveTrajectory(all_rt_trajectories, FileUtils::joinPaths(output_dir_base_, "rt_trajectories/Rt_aggregated_with_uncertainty.csv"));
    aggregateAndSaveTrajectory(all_sero_trajectories, FileUtils::joinPaths(output_dir_base_, "seroprevalence/seroprevalence_trajectory.csv"));
    
    Logger::getInstance().info("PostCalibrationAnalyser", "Waiting for async I/O to complete...");
    writer_->waitForCompletion();
    
    auto [cache_hits, total_calls] = runner_->getCacheStats();
    Logger::getInstance().info("PostCalibrationAnalyser", 
        "Simulation cache stats: " + std::to_string(cache_hits) + "/" + 
        std::to_string(total_calls) + " hits (" + 
        std::to_string(total_calls > 0 ? (100.0 * cache_hits / total_calls) : 0.0) + "%)");
    
    Logger::getInstance().info("PostCalibrationAnalyser", "MCMC batch analysis completed");
}

PosteriorPredictiveData PostCalibrationAnalyser::generatePosteriorPredictiveChecksOptimized(
    const std::vector<Eigen::VectorXd>& param_samples,
    IParameterManager& param_manager,
    int num_samples_for_ppc) {
    
    Logger::getInstance().info("PostCalibrationAnalyser", "Delegating PPC generation to aggregator...");
    
    // Delegate entire PPC workflow to the aggregator
    return aggregator_->aggregatePosteriorPredictives(
        param_samples,
        param_manager,
        *runner_,
        *metrics_calculator_,
        num_samples_for_ppc,
        time_points_,
        initial_state_,
        observed_data_,
        model_template_
    );
}

void PostCalibrationAnalyser::performScenarioAnalysisOptimized(
    const SEPAIHRDParameters& baseline_params,
    const std::vector<std::pair<std::string, SEPAIHRDParameters>>& scenarios) {
    
    Logger::getInstance().info("PostCalibrationAnalyser", "Starting scenario analysis...");
    
    EssentialMetrics baseline_metrics = analyzeSingleRunLightweight(baseline_params, "baseline");
    
    std::vector<std::pair<std::string, EssentialMetrics>> scenario_results;
    scenario_results.push_back({"baseline", baseline_metrics});
    
    for (const auto& [scenario_name, scenario_params] : scenarios) {
        Logger::getInstance().info("PostCalibrationAnalyser", "Analyzing scenario: " + scenario_name);
        EssentialMetrics scenario_metrics = analyzeSingleRunLightweight(scenario_params, scenario_name);
        scenario_results.push_back({scenario_name, scenario_metrics});
    }
    
    writer_->saveScenarioComparison(
        FileUtils::joinPaths(output_dir_base_, "scenarios/scenario_comparison.csv"),
        scenario_results
    );
        
    Logger::getInstance().info("PostCalibrationAnalyser", "Scenario analysis completed");
}

void PostCalibrationAnalyser::ensureOutputSubdirectoryExists(const std::string& subdir_name) {
    FileUtils::ensureDirectoryExists(FileUtils::joinPaths(output_dir_base_, subdir_name));
}

} // namespace epidemic

