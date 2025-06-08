#include "model/PostCalibrationAnalyser.hpp"
#include "model/AgeSEPAIHRDsimulator.hpp"
#include "model/ReproductionNumberCalculator.hpp"
#include "model/ModelConstants.hpp"
#include "sir_age_structured/SimulationResultProcessor.hpp"
#include "utils/FileUtils.hpp"
#include "utils/Logger.hpp"
#include <fstream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>

namespace epidemic {

PostCalibrationAnalyser::PostCalibrationAnalyser(
    std::shared_ptr<AgeSEPAIHRDModel> model_template,
    std::shared_ptr<IOdeSolverStrategy> solver,
    const std::vector<double>& time_points,
    const Eigen::VectorXd& initial_state,
    const std::string& output_directory,
    const CalibrationData& observed_data)
    : model_template_(model_template),
      solver_strategy_(solver),
      time_points_(time_points),
      initial_state_(initial_state),
      output_dir_base_(output_directory),
      observed_data_(observed_data) {

    if (!model_template_) throw std::invalid_argument("PostCalibrationAnalyser: Model template cannot be null");
    if (!solver_strategy_) throw std::invalid_argument("PostCalibrationAnalyser: Solver strategy cannot be null");
    if (time_points_.empty()) throw std::invalid_argument("PostCalibrationAnalyser: Time points cannot be empty");
    if (initial_state_.size() == 0) throw std::invalid_argument("PostCalibrationAnalyser: Initial state cannot be empty");
    if (initial_state_.size() != model_template_->getStateSize()) {
        throw std::invalid_argument("PostCalibrationAnalyser: Initial state size does not match model state size");
    }

    num_age_classes_ = model_template_->getNumAgeClasses();
    FileUtils::ensureDirectoryExists(output_dir_base_);
    Logger::getInstance().info("PostCalibrationAnalyser", "Output directory: " + output_dir_base_);
}

void PostCalibrationAnalyser::ensureOutputSubdirectoryExists(const std::string& subdir_name) {
    FileUtils::ensureDirectoryExists(FileUtils::joinPaths(output_dir_base_, subdir_name));
}

void PostCalibrationAnalyser::generateFullReport(
    const std::vector<Eigen::VectorXd>& param_samples,
    IParameterManager& param_manager,
    int num_samples_for_posterior_pred,
    int burn_in_for_summary,
    int thinning_for_summary,
    int batch_size) {

    Logger::getInstance().info("PostCalibrationAnalyser", "Starting memory-optimized full report generation...");

    // 1. Posterior Predictive Checks
    ensureOutputSubdirectoryExists("posterior_predictive");
    Logger::getInstance().info("PostCalibrationAnalyser", "Generating Posterior Predictive Checks...");
    PosteriorPredictiveData ppd_data = generatePosteriorPredictiveChecksOptimized(
        param_samples, param_manager, num_samples_for_posterior_pred
    );
    savePosteriorPredictiveCheckData(ppd_data, "posterior_predictive");
    
    // Clear PPC data from memory
    ppd_data = PosteriorPredictiveData();
    
    // 2. MCMC Analysis in batches
    ensureOutputSubdirectoryExists("mcmc_batches");
    Logger::getInstance().info("PostCalibrationAnalyser", "Analyzing MCMC runs in batches...");
    analyzeMCMCRunsInBatches(param_samples, param_manager, burn_in_for_summary, thinning_for_summary, batch_size);
    
    // 3. Parameter Posteriors with streaming
    ensureOutputSubdirectoryExists("parameter_posteriors");
    Logger::getInstance().info("PostCalibrationAnalyser", "Saving parameter posteriors...");
    saveParameterPosteriorsStreaming(param_samples, param_manager, burn_in_for_summary, thinning_for_summary);
    
    // 4. Scenario Analysis
    ensureOutputSubdirectoryExists("scenarios");
    if (!param_samples.empty()) {
        Logger::getInstance().info("PostCalibrationAnalyser", "Performing scenario analysis...");
        
        // Compute mean parameters efficiently
        Eigen::VectorXd mean_params = Eigen::VectorXd::Zero(param_manager.getParameterCount());
        int count = 0;
        for (size_t i = burn_in_for_summary; i < param_samples.size(); i += thinning_for_summary) {
            mean_params += param_samples[i];
            count++;
        }
        if (count > 0) mean_params /= count;
        
        param_manager.updateModelParameters(mean_params);
        SEPAIHRDParameters baseline_params = model_template_->getModelParameters();
        
        // Define scenarios
        std::vector<std::pair<std::string, SEPAIHRDParameters>> scenarios;
        
        SEPAIHRDParameters stricter_params = baseline_params;
        if (stricter_params.kappa_values.size() > 1) {
            stricter_params.kappa_values[1] *= 0.9;
        }
        scenarios.push_back({"stricter_lockdown", stricter_params});
        
        SEPAIHRDParameters weaker_params = baseline_params;
        if (weaker_params.kappa_values.size() > 1) {
            weaker_params.kappa_values[1] *= 1.1;
        }
        scenarios.push_back({"weaker_lockdown", weaker_params});
        
        performScenarioAnalysisOptimized(baseline_params, scenarios);
    }
    
    Logger::getInstance().info("PostCalibrationAnalyser", "Full report generation completed.");
}

EssentialMetrics PostCalibrationAnalyser::analyzeSingleRunLightweight(
    const SEPAIHRDParameters& params,
    const std::string& run_id) {
    
    EssentialMetrics metrics;
    metrics.age_specific_IFR.resize(num_age_classes_);
    metrics.age_specific_IHR.resize(num_age_classes_);
    metrics.age_specific_IICUR.resize(num_age_classes_);
    metrics.age_specific_attack_rate.resize(num_age_classes_);
    
    // Create model and run simulation
    auto run_npi_strategy = model_template_->getNpiStrategy()->clone();
    auto run_model = std::make_shared<AgeSEPAIHRDModel>(params, run_npi_strategy);
    
    // Calculate R0
    ReproductionNumberCalculator rn_calc(run_model);
    metrics.R0 = rn_calc.calculateR0();
    
    // Run simulation
    AgeSEPAIHRDSimulator simulator(run_model, solver_strategy_,
                                  time_points_.front(), time_points_.back(), 1.0, 1e-6, 1e-6);
    SimulationResult sim_result = simulator.run(initial_state_, time_points_);
    
    if (!sim_result.isValid()) {
        Logger::getInstance().warning("PostCalibrationAnalyser", "Invalid simulation for " + run_id);
        return metrics;
    }
    
    // Process results efficiently without storing full trajectories
    Eigen::VectorXd cumulative_infections = Eigen::VectorXd::Zero(num_age_classes_);
    Eigen::VectorXd cumulative_hosp = Eigen::VectorXd::Zero(num_age_classes_);
    Eigen::VectorXd cumulative_icu = Eigen::VectorXd::Zero(num_age_classes_);
    double total_population = params.N.sum();
    
    // Variables for tracking peaks and Rt
    metrics.peak_hospital_occupancy = 0.0;
    metrics.peak_ICU_occupancy = 0.0;
    
    // Target day for seroprevalence (day 64 = May 4th)
    const double target_day = 64.0;
    size_t target_idx = 0;
    for (size_t i = 0; i < time_points_.size(); ++i) {
        if (std::abs(time_points_[i] - target_day) < 0.5) {
            target_idx = i;
            break;
        }
    }
    
    // Process simulation timestep by timestep
    for (size_t t = 0; t < time_points_.size(); ++t) {
        double time_t = time_points_[t];
        double dt = (t > 0) ? (time_t - time_points_[t-1]) : 1.0;
        
        // Extract current state
        Eigen::Map<const Eigen::VectorXd> S_t(&sim_result.solution[t][0 * num_age_classes_], num_age_classes_);
        Eigen::Map<const Eigen::VectorXd> P_t(&sim_result.solution[t][2 * num_age_classes_], num_age_classes_);
        Eigen::Map<const Eigen::VectorXd> A_t(&sim_result.solution[t][3 * num_age_classes_], num_age_classes_);
        Eigen::Map<const Eigen::VectorXd> I_t(&sim_result.solution[t][4 * num_age_classes_], num_age_classes_);
        Eigen::Map<const Eigen::VectorXd> H_t(&sim_result.solution[t][5 * num_age_classes_], num_age_classes_);
        Eigen::Map<const Eigen::VectorXd> ICU_t(&sim_result.solution[t][6 * num_age_classes_], num_age_classes_);
        
        // Calculate Rt
        double Rt = rn_calc.calculateRt(S_t, time_t);
        metrics.max_Rt = std::max(metrics.max_Rt, Rt);
        metrics.min_Rt = std::min(metrics.min_Rt, Rt);
        if (t == time_points_.size() - 1) metrics.final_Rt = Rt;
        
        // Track peaks
        double total_H = H_t.sum();
        double total_ICU = ICU_t.sum();
        if (total_H > metrics.peak_hospital_occupancy) {
            metrics.peak_hospital_occupancy = total_H;
            metrics.time_to_peak_hospital = time_t;
        }
        if (total_ICU > metrics.peak_ICU_occupancy) {
            metrics.peak_ICU_occupancy = total_ICU;
            metrics.time_to_peak_ICU = time_t;
        }
        
        // Accumulate flows
        double kappa_t = run_model->getNpiStrategy()->getReductionFactor(time_t);
        Eigen::VectorXd infectious_load = Eigen::VectorXd::Zero(num_age_classes_);
        for (int j = 0; j < num_age_classes_; ++j) {
            if (params.N(j) > 1e-9) {
                infectious_load(j) = (P_t(j) + A_t(j) + params.theta * I_t(j)) / params.N(j);
            }
        }
        Eigen::VectorXd lambda_t = params.beta * kappa_t * params.M_baseline * infectious_load;
        Eigen::VectorXd new_infections = lambda_t.array() * S_t.array() * dt;
        
        cumulative_infections += new_infections;
        cumulative_hosp += (params.h.array() * I_t.array() * dt).matrix();
        cumulative_icu += (params.icu.array() * H_t.array() * dt).matrix();        
        // Seroprevalence at target day
        if (t == target_idx) {
            metrics.seroprevalence_at_target_day = cumulative_infections.sum() / total_population;
        }
    }
    
    // Final metrics
    Eigen::Map<const Eigen::VectorXd> D_final(&sim_result.solution.back()[8 * num_age_classes_], num_age_classes_);
    Eigen::Map<const Eigen::VectorXd> D_initial(&initial_state_[8 * num_age_classes_], num_age_classes_);
    Eigen::VectorXd cumulative_deaths = D_final - D_initial;
    
    metrics.total_cumulative_deaths = cumulative_deaths.sum();
    metrics.overall_attack_rate = cumulative_infections.sum() / total_population;
    metrics.overall_IFR = (cumulative_infections.sum() > 1e-9) ? 
        cumulative_deaths.sum() / cumulative_infections.sum() : 0.0;
    
    // Age-specific metrics
    for (int age = 0; age < num_age_classes_; ++age) {
        metrics.age_specific_attack_rate[age] = (params.N(age) > 0) ? 
            cumulative_infections(age) / params.N(age) : 0.0;
        
        if (cumulative_infections(age) > 1e-9) {
            metrics.age_specific_IFR[age] = cumulative_deaths(age) / cumulative_infections(age);
            metrics.age_specific_IHR[age] = cumulative_hosp(age) / cumulative_infections(age);
            metrics.age_specific_IICUR[age] = cumulative_icu(age) / cumulative_infections(age);
        } else {
            metrics.age_specific_IFR[age] = 0.0;
            metrics.age_specific_IHR[age] = 0.0;
            metrics.age_specific_IICUR[age] = 0.0;
        }
    }
    
    // Kappa values
    for (size_t i = 0; i < params.kappa_values.size(); ++i) {
        metrics.kappa_values["kappa_" + std::to_string(i + 1)] = params.kappa_values[i];
    }
    
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
    
    // Create batch directory
    ensureOutputSubdirectoryExists("mcmc_batches");
    
    // Process samples in batches
    int batch_count = 0;
    int processed_samples = 0;
    
    for (int start_idx = effective_start; start_idx < total_samples; start_idx += batch_size * thinning) {
        std::vector<EssentialMetrics> batch_metrics;
        batch_metrics.reserve(batch_size);
        
        // Process current batch
        for (int i = 0; i < batch_size && (start_idx + i * thinning) < total_samples; ++i) {
            int sample_idx = start_idx + i * thinning;
            
            const Eigen::VectorXd& param_vec = param_samples[sample_idx];
            param_manager.updateModelParameters(param_vec);
            SEPAIHRDParameters params = model_template_->getModelParameters();
            
            std::string run_id = "mcmc_sample_" + std::to_string(sample_idx);
            EssentialMetrics metrics = analyzeSingleRunLightweight(params, run_id);
            batch_metrics.push_back(metrics);
            processed_samples++;
            
            if (processed_samples % 10 == 0) {
                Logger::getInstance().info("PostCalibrationAnalyser", 
                    "Processed " + std::to_string(processed_samples) + " samples...");
            }
        }
        
        // Save batch results
        processAndSaveBatch(batch_metrics, batch_count, "mcmc_batches");
        batch_count++;
        
        // Clear batch memory
        batch_metrics.clear();
        batch_metrics.shrink_to_fit();
    }
    
    // Aggregate results from all batches
    Logger::getInstance().info("PostCalibrationAnalyser", "Aggregating results from " + 
                              std::to_string(batch_count) + " batches...");
    aggregateBatchResults("mcmc_batches", batch_count);
    
    // ENE-COVID validation using aggregated results
    ensureOutputSubdirectoryExists("seroprevalence");
    performENECOVIDValidationFromBatches("mcmc_batches", batch_count);
}

PosteriorPredictiveData PostCalibrationAnalyser::generatePosteriorPredictiveChecksOptimized(
    const std::vector<Eigen::VectorXd>& param_samples,
    IParameterManager& param_manager,
    int num_samples_for_ppc) {
    
    Logger::getInstance().info("PostCalibrationAnalyser", "Starting optimized PPC generation...");
    
    PosteriorPredictiveData ppd_data;
    ppd_data.time_points = time_points_;
    
    // Fill observed data
    ppd_data.daily_hospitalizations.observed = observed_data_.getNewHospitalizations();
    ppd_data.daily_icu_admissions.observed = observed_data_.getNewICU();
    ppd_data.daily_deaths.observed = observed_data_.getNewDeaths();
    ppd_data.cumulative_hospitalizations.observed = observed_data_.getCumulativeHospitalizations();
    ppd_data.cumulative_icu_admissions.observed = observed_data_.getCumulativeICU();
    ppd_data.cumulative_deaths.observed = observed_data_.getCumulativeDeaths();
    
    if (param_samples.empty()) {
        Logger::getInstance().warning("PostCalibrationAnalyser", "No samples for PPC.");
        return ppd_data;
    }
    
    int T = time_points_.size();
    int A = num_age_classes_;
    
    // Initialize quantile accumulators
    std::vector<std::vector<QuantileAccumulator>> hosp_acc(T, std::vector<QuantileAccumulator>(A));
    std::vector<std::vector<QuantileAccumulator>> icu_acc(T, std::vector<QuantileAccumulator>(A));
    std::vector<std::vector<QuantileAccumulator>> death_acc(T, std::vector<QuantileAccumulator>(A));
    std::vector<std::vector<QuantileAccumulator>> c_hosp_acc(T, std::vector<QuantileAccumulator>(A));
    std::vector<std::vector<QuantileAccumulator>> c_icu_acc(T, std::vector<QuantileAccumulator>(A));
    std::vector<std::vector<QuantileAccumulator>> c_death_acc(T, std::vector<QuantileAccumulator>(A));
    
    // Select samples
    std::vector<int> selected_indices;
    if (num_samples_for_ppc > 0 && static_cast<size_t>(num_samples_for_ppc) < param_samples.size()) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(0, param_samples.size() - 1);
        selected_indices.reserve(num_samples_for_ppc);
        for (int i = 0; i < num_samples_for_ppc; ++i) {
            selected_indices.push_back(distrib(gen));
        }
    } else {
        selected_indices.resize(param_samples.size());
        std::iota(selected_indices.begin(), selected_indices.end(), 0);
    }
    
    // Reserve space for accumulators
    int expected_samples = selected_indices.size();
    for (int t = 0; t < T; ++t) {
        for (int a = 0; a < A; ++a) {
            hosp_acc[t][a].reserve(expected_samples);
            icu_acc[t][a].reserve(expected_samples);
            death_acc[t][a].reserve(expected_samples);
            c_hosp_acc[t][a].reserve(expected_samples);
            c_icu_acc[t][a].reserve(expected_samples);
            c_death_acc[t][a].reserve(expected_samples);
        }
    }
    
    // Process samples one at a time
    int processed = 0;
    for (int sample_idx : selected_indices) {
        const Eigen::VectorXd& p_vec = param_samples[sample_idx];
        param_manager.updateModelParameters(p_vec);
        SEPAIHRDParameters params = model_template_->getModelParameters();
        
        auto run_npi_strategy = model_template_->getNpiStrategy()->clone();
        auto run_model = std::make_shared<AgeSEPAIHRDModel>(params, run_npi_strategy);
        AgeSEPAIHRDSimulator simulator(run_model, solver_strategy_, 
                                      time_points_.front(), time_points_.back(), 1.0, 1e-6, 1e-6);
        
        SimulationResult sim_result = simulator.run(initial_state_, time_points_);
        if (!sim_result.isValid()) continue;
        
        // Calculate daily incidence flows
        Eigen::MatrixXd daily_hosp = Eigen::MatrixXd::Zero(T, A);
        Eigen::MatrixXd daily_icu = Eigen::MatrixXd::Zero(T, A);
        Eigen::MatrixXd daily_deaths = Eigen::MatrixXd::Zero(T, A);
        
        for (int t = 0; t < T; ++t) {
            Eigen::Map<const Eigen::VectorXd> I_t(&sim_result.solution[t][4 * num_age_classes_], num_age_classes_);
            Eigen::Map<const Eigen::VectorXd> H_t(&sim_result.solution[t][5 * num_age_classes_], num_age_classes_);
            Eigen::Map<const Eigen::VectorXd> ICU_t(&sim_result.solution[t][6 * num_age_classes_], num_age_classes_);
            
            for (int age = 0; age < A; ++age) {
                daily_hosp(t, age) = params.h(age) * I_t(age);
                daily_icu(t, age) = params.icu(age) * H_t(age);
                daily_deaths(t, age) = params.d_H(age) * H_t(age) + params.d_ICU(age) * ICU_t(age);
            }
        }
        
        // Accumulate values for quantile calculation
        for (int t = 0; t < T; ++t) {
            for (int a = 0; a < A; ++a) {
                hosp_acc[t][a].add(daily_hosp(t, a));
                icu_acc[t][a].add(daily_icu(t, a));
                death_acc[t][a].add(daily_deaths(t, a));
                
                // Cumulative values
                double c_h = (t > 0) ? c_hosp_acc[t-1][a].quantile(1.0) + daily_hosp(t, a) : daily_hosp(t, a);
                double c_i = (t > 0) ? c_icu_acc[t-1][a].quantile(1.0) + daily_icu(t, a) : daily_icu(t, a);
                double c_d = (t > 0) ? c_death_acc[t-1][a].quantile(1.0) + daily_deaths(t, a) : daily_deaths(t, a);
                
                c_hosp_acc[t][a].add(c_h);
                c_icu_acc[t][a].add(c_i);
                c_death_acc[t][a].add(c_d);
            }
        }
        
        processed++;
        if (processed % 10 == 0) {
            Logger::getInstance().info("PostCalibrationAnalyser", 
                "PPC: Processed " + std::to_string(processed) + "/" + std::to_string(expected_samples) + " samples");
        }
    }
    
    // Compute quantiles
    auto fillQuantiles = [&](PosteriorPredictiveData::IncidenceData& data,
                            std::vector<std::vector<QuantileAccumulator>>& acc) {
        data.median.resize(T, A);
        data.lower_90.resize(T, A);
        data.upper_90.resize(T, A);
        data.lower_95.resize(T, A);
        data.upper_95.resize(T, A);
        
        for (int t = 0; t < T; ++t) {
            for (int a = 0; a < A; ++a) {
                if (acc[t][a].size() > 0) {
                    data.median(t, a) = acc[t][a].quantile(0.5);
                    data.lower_90(t, a) = acc[t][a].quantile(0.05);
                    data.upper_90(t, a) = acc[t][a].quantile(0.95);
                    data.lower_95(t, a) = acc[t][a].quantile(0.025);
                    data.upper_95(t, a) = acc[t][a].quantile(0.975);
                } else {
                    data.median(t, a) = NAN;
                    data.lower_90(t, a) = NAN;
                    data.upper_90(t, a) = NAN;
                    data.lower_95(t, a) = NAN;
                    data.upper_95(t, a) = NAN;
                }
                // Clear accumulator to free memory
                acc[t][a].clear();
            }
        }
    };
    
    fillQuantiles(ppd_data.daily_hospitalizations, hosp_acc);
    fillQuantiles(ppd_data.daily_icu_admissions, icu_acc);
    fillQuantiles(ppd_data.daily_deaths, death_acc);
    fillQuantiles(ppd_data.cumulative_hospitalizations, c_hosp_acc);
    fillQuantiles(ppd_data.cumulative_icu_admissions, c_icu_acc);
    fillQuantiles(ppd_data.cumulative_deaths, c_death_acc);
    
    Logger::getInstance().info("PostCalibrationAnalyser", 
        "PPC completed using " + std::to_string(processed) + " samples");
    
    return ppd_data;
}

void PostCalibrationAnalyser::performScenarioAnalysisOptimized(
    const SEPAIHRDParameters& baseline_params,
    const std::vector<std::pair<std::string, SEPAIHRDParameters>>& scenarios) {
    
    Logger::getInstance().info("PostCalibrationAnalyser", "Starting optimized scenario analysis...");
    
    // Analyze baseline
    EssentialMetrics baseline_metrics = analyzeSingleRunLightweight(baseline_params, "baseline");
    
    // Save comparison summary
    std::string summary_path = FileUtils::joinPaths(
        FileUtils::joinPaths(output_dir_base_, "scenarios"), "scenario_comparison.csv");
    std::ofstream file(summary_path);
    
    file << "scenario,R0,overall_IFR,overall_attack_rate,peak_hospital,peak_ICU,"
         << "time_to_peak_hospital,time_to_peak_ICU,total_deaths,seroprevalence_day64";
    
    // Add kappa headers
    for (const auto& kappa_pair : baseline_metrics.kappa_values) {
        file << "," << kappa_pair.first;
    }
    file << "\n";
    
    // Write baseline
    file << "baseline," << baseline_metrics.R0 << "," << baseline_metrics.overall_IFR << ","
         << baseline_metrics.overall_attack_rate << "," << baseline_metrics.peak_hospital_occupancy << ","
         << baseline_metrics.peak_ICU_occupancy << "," << baseline_metrics.time_to_peak_hospital << ","
         << baseline_metrics.time_to_peak_ICU << "," << baseline_metrics.total_cumulative_deaths << ","
         << baseline_metrics.seroprevalence_at_target_day;
    
    for (const auto& kappa_pair : baseline_metrics.kappa_values) {
        file << "," << kappa_pair.second;
    }
    file << "\n";
    
    // Process scenarios
    for (const auto& scenario : scenarios) {
        Logger::getInstance().info("PostCalibrationAnalyser", "Analyzing scenario: " + scenario.first);
        EssentialMetrics scenario_metrics = analyzeSingleRunLightweight(scenario.second, scenario.first);
        
        file << scenario.first << "," << scenario_metrics.R0 << "," << scenario_metrics.overall_IFR << ","
             << scenario_metrics.overall_attack_rate << "," << scenario_metrics.peak_hospital_occupancy << ","
             << scenario_metrics.peak_ICU_occupancy << "," << scenario_metrics.time_to_peak_hospital << ","
             << scenario_metrics.time_to_peak_ICU << "," << scenario_metrics.total_cumulative_deaths << ","
             << scenario_metrics.seroprevalence_at_target_day;
        
        for (const auto& kappa_pair : baseline_metrics.kappa_values) {
            if (scenario_metrics.kappa_values.count(kappa_pair.first)) {
                file << "," << scenario_metrics.kappa_values.at(kappa_pair.first);
            } else {
                file << ",";
            }
        }
        file << "\n";
    }
    
    file.close();
    Logger::getInstance().info("PostCalibrationAnalyser", "Scenario analysis saved to: " + summary_path);
}

void PostCalibrationAnalyser::validateAgainstENECOVID(
    const std::vector<EssentialMetrics>& all_metrics,
    double ene_covid_target_day,
    double ene_covid_mean,
    double ene_covid_lower_ci,
    double ene_covid_upper_ci) {
    
    if (all_metrics.empty()) {
        Logger::getInstance().warning("PostCalibrationAnalyser", "No metrics for ENE-COVID validation");
        return;
    }
    
    // Collect seroprevalence values
    std::vector<double> sero_values;
    sero_values.reserve(all_metrics.size());
    
    for (const auto& m : all_metrics) {
        sero_values.push_back(m.seroprevalence_at_target_day);
    }
    
    std::sort(sero_values.begin(), sero_values.end());
    
    double model_median = sero_values[sero_values.size() / 2];
    double model_q025 = sero_values[static_cast<size_t>(0.025 * sero_values.size())];
    double model_q975 = sero_values[static_cast<size_t>(0.975 * sero_values.size())];
    
    std::string filepath = FileUtils::joinPaths(
        FileUtils::joinPaths(output_dir_base_, "seroprevalence"), "ene_covid_validation.csv");
    
    std::ofstream file(filepath);
    file << "source,median_seroprevalence,lower_95ci,upper_95ci,target_day\n";
    file << std::fixed << std::setprecision(5);
    file << "Model," << model_median << "," << model_q025 << "," << model_q975 << "," << ene_covid_target_day << "\n";
    file << "ENE_COVID," << ene_covid_mean << "," << ene_covid_lower_ci << "," << ene_covid_upper_ci << "," 
         << ene_covid_target_day << "\n";
    file.close();
    
    Logger::getInstance().info("PostCalibrationAnalyser", "ENE-COVID validation saved to: " + filepath);
}

void PostCalibrationAnalyser::saveParameterPosteriorsStreaming(
    const std::vector<Eigen::VectorXd>& param_samples,
    IParameterManager& param_manager,
    int burn_in,
    int thinning) {
    
    const auto& param_names = param_manager.getParameterNames(); 
    // Save posterior samples
    std::string samples_filepath = FileUtils::joinPaths(
        FileUtils::joinPaths(output_dir_base_, "parameter_posteriors"), "posterior_samples.csv");
    std::ofstream sfile(samples_filepath);
    
    // Write header
    sfile << "sample_index";
    for (const auto& name : param_names) {
        sfile << "," << name;
    }
    sfile << "\n";
    
    // Write samples
    int saved_count = 0;
    for (size_t i = burn_in; i < param_samples.size(); i += thinning) {
        sfile << saved_count++;
        for (int j = 0; j < param_samples[i].size(); ++j) {
            sfile << "," << std::scientific << std::setprecision(8) << param_samples[i][j];
        }
        sfile << "\n";
    }
    sfile.close();
    
    // Compute and save summary statistics
    std::string summary_filepath = FileUtils::joinPaths(
        FileUtils::joinPaths(output_dir_base_, "parameter_posteriors"), "posterior_summary.csv");
    std::ofstream sumfile(summary_filepath);
    sumfile << "parameter,mean,median,std_dev,q025,q975\n";
    sumfile << std::fixed << std::setprecision(8);
    
    // Process one parameter at a time to minimize memory usage
    for (size_t p_idx = 0; p_idx < param_names.size(); ++p_idx) {
        std::vector<double> values;
        values.reserve((param_samples.size() - burn_in) / thinning);
        
        for (size_t i = burn_in; i < param_samples.size(); i += thinning) {
            if (p_idx < static_cast<size_t>(param_samples[i].size())) {
                values.push_back(param_samples[i][p_idx]);
            }
        }
        
        if (!values.empty()) {
            std::sort(values.begin(), values.end());
            double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
            double median = values[values.size() / 2];
            double q025 = values[static_cast<size_t>(0.025 * values.size())];
            double q975 = values[static_cast<size_t>(0.975 * values.size())];
            
            double sum_sq_diff = 0.0;
            for (double val : values) {
                sum_sq_diff += (val - mean) * (val - mean);
            }
            double std_dev = std::sqrt(sum_sq_diff / values.size());
            
            sumfile << param_names[p_idx] << "," << mean << "," << median << "," 
                   << std_dev << "," << q025 << "," << q975 << "\n";
        }
        
        // Clear values to free memory
        values.clear();
        values.shrink_to_fit();
    }
    
    sumfile.close();
    Logger::getInstance().info("PostCalibrationAnalyser", "Parameter posteriors saved");
}

// Helper methods

void PostCalibrationAnalyser::saveEssentialMetricsCSV(
    const std::string& filepath,
    const EssentialMetrics& metrics) {
    
    std::ofstream file(filepath);
    file << "metric_name,value\n";
    file << std::fixed << std::setprecision(8);
    file << "R0," << metrics.R0 << "\n";
    file << "overall_IFR," << metrics.overall_IFR << "\n";
    file << "overall_attack_rate," << metrics.overall_attack_rate << "\n";
    file << "peak_hospital_occupancy," << metrics.peak_hospital_occupancy << "\n";
    file << "peak_ICU_occupancy," << metrics.peak_ICU_occupancy << "\n";
    file << "time_to_peak_hospital," << metrics.time_to_peak_hospital << "\n";
    file << "time_to_peak_ICU," << metrics.time_to_peak_ICU << "\n";
    file << "total_cumulative_deaths," << metrics.total_cumulative_deaths << "\n";
    file << "max_Rt," << metrics.max_Rt << "\n";
    file << "min_Rt," << metrics.min_Rt << "\n";
    file << "final_Rt," << metrics.final_Rt << "\n";
    file << "seroprevalence_at_target_day," << metrics.seroprevalence_at_target_day << "\n";
    
    // Age-specific metrics
    for (int age = 0; age < num_age_classes_; ++age) {
        file << "IFR_age_" << age << "," << metrics.age_specific_IFR[age] << "\n";
        file << "IHR_age_" << age << "," << metrics.age_specific_IHR[age] << "\n";
        file << "IICUR_age_" << age << "," << metrics.age_specific_IICUR[age] << "\n";
        file << "AttackRate_age_" << age << "," << metrics.age_specific_attack_rate[age] << "\n";
    }
    
    // Kappa values
    for (const auto& kappa_pair : metrics.kappa_values) {
        file << kappa_pair.first << "," << kappa_pair.second << "\n";
    }
    
    file.close();
}

void PostCalibrationAnalyser::processAndSaveBatch(
    const std::vector<EssentialMetrics>& batch_metrics,
    int batch_index,
    const std::string& batch_subdir) {
    
    std::string batch_file = FileUtils::joinPaths(
        FileUtils::joinPaths(output_dir_base_, batch_subdir),
        "batch_" + std::to_string(batch_index) + ".csv");
    
    std::ofstream file(batch_file);
    
    // Write header
    file << "sample_idx,R0,overall_IFR,overall_attack_rate,peak_hospital,peak_ICU,"
         << "time_to_peak_hospital,time_to_peak_ICU,total_deaths,"
         << "max_Rt,min_Rt,final_Rt,seroprevalence_day64";
    
    for (int age = 0; age < num_age_classes_; ++age) {
        file << ",IFR_age_" << age << ",IHR_age_" << age 
             << ",IICUR_age_" << age << ",AttackRate_age_" << age;
    }
    
    if (!batch_metrics.empty() && !batch_metrics[0].kappa_values.empty()) {
        for (const auto& kappa_pair : batch_metrics[0].kappa_values) {
            file << "," << kappa_pair.first;
        }
    }
    file << "\n";
    
    // Write data
    for (size_t i = 0; i < batch_metrics.size(); ++i) {
        const auto& m = batch_metrics[i];
        file << i << "," << m.R0 << "," << m.overall_IFR << "," << m.overall_attack_rate
             << "," << m.peak_hospital_occupancy << "," << m.peak_ICU_occupancy
             << "," << m.time_to_peak_hospital << "," << m.time_to_peak_ICU
             << "," << m.total_cumulative_deaths << "," << m.max_Rt
             << "," << m.min_Rt << "," << m.final_Rt << "," << m.seroprevalence_at_target_day;
        
        for (int age = 0; age < num_age_classes_; ++age) {
            file << "," << m.age_specific_IFR[age] << "," << m.age_specific_IHR[age]
                 << "," << m.age_specific_IICUR[age] << "," << m.age_specific_attack_rate[age];
        }
        
        for (const auto& kappa_pair : m.kappa_values) {
            file << "," << kappa_pair.second;
        }
        file << "\n";
    }
    
    file.close();
}

void PostCalibrationAnalyser::aggregateBatchResults(
    const std::string& batch_subdir,
    int num_batches) {
    
    // Define columns to aggregate
    std::vector<std::string> scalar_columns = {
        "R0", "overall_IFR", "overall_attack_rate", "peak_hospital", "peak_ICU",
        "time_to_peak_hospital", "time_to_peak_ICU", "total_deaths",
        "max_Rt", "min_Rt", "final_Rt", "seroprevalence_day64"
    };
    
    // Add age-specific columns
    for (int age = 0; age < num_age_classes_; ++age) {
        scalar_columns.push_back("IFR_age_" + std::to_string(age));
        scalar_columns.push_back("IHR_age_" + std::to_string(age));
        scalar_columns.push_back("IICUR_age_" + std::to_string(age));
        scalar_columns.push_back("AttackRate_age_" + std::to_string(age));
    }
    
    // Create aggregated summary
    std::string summary_path = FileUtils::joinPaths(
        FileUtils::joinPaths(output_dir_base_, "mcmc_aggregated"), "metrics_summary.csv");
    
    std::ofstream summary_file(summary_path);
    summary_file << "metric,mean,median,std_dev,q025,q975\n";
    summary_file << std::fixed << std::setprecision(8);
    
    // Process each metric
    for (const auto& col_name : scalar_columns) {
        std::vector<double> all_values;
        
        // Read values from all batch files
        for (int batch = 0; batch < num_batches; ++batch) {
            std::string batch_file = FileUtils::joinPaths(
                FileUtils::joinPaths(output_dir_base_, batch_subdir),
                "batch_" + std::to_string(batch) + ".csv");
            
            std::ifstream file(batch_file);
            std::string line;
            std::getline(file, line); // Skip header
            
            // Find column index
            std::istringstream header_stream(line);
            std::string col;
            int col_idx = -1;
            int current_idx = 0;
            while (std::getline(header_stream, col, ',')) {
                if (col == col_name) {
                    col_idx = current_idx;
                    break;
                }
                current_idx++;
            }
            
            if (col_idx >= 0) {
                while (std::getline(file, line)) {
                    std::istringstream data_stream(line);
                    std::string value;
                    for (int i = 0; i <= col_idx; ++i) {
                        std::getline(data_stream, value, ',');
                    }
                    if (!value.empty()) {
                        all_values.push_back(std::stod(value));
                    }
                }
            }
            file.close();
        }
        
        // Compute statistics
        if (!all_values.empty()) {
            std::sort(all_values.begin(), all_values.end());
            double mean = std::accumulate(all_values.begin(), all_values.end(), 0.0) / all_values.size();
            double median = all_values[all_values.size() / 2];
            double q025 = all_values[static_cast<size_t>(0.025 * all_values.size())];
            double q975 = all_values[static_cast<size_t>(0.975 * all_values.size())];
            
            double sum_sq_diff = 0.0;
            for (double val : all_values) {
                sum_sq_diff += (val - mean) * (val - mean);
            }
            double std_dev = std::sqrt(sum_sq_diff / all_values.size());
            
            summary_file << col_name << "," << mean << "," << median << ","
                        << std_dev << "," << q025 << "," << q975 << "\n";
        }
    }
    
    summary_file.close();
    Logger::getInstance().info("PostCalibrationAnalyser", "Aggregated summary saved to: " + summary_path);
}

void PostCalibrationAnalyser::performENECOVIDValidationFromBatches(
    const std::string& batch_subdir,
    int num_batches) {
    
    std::vector<double> sero_values;
    
    // Collect seroprevalence values from all batches
    for (int batch = 0; batch < num_batches; ++batch) {
        std::string batch_file = FileUtils::joinPaths(
            FileUtils::joinPaths(output_dir_base_, batch_subdir),
            "batch_" + std::to_string(batch) + ".csv");
        
        std::ifstream file(batch_file);
        std::string line;
        std::getline(file, line); // Header
        
        // Find seroprevalence column
        std::istringstream header_stream(line);
        std::string col;
        int sero_col_idx = -1;
        int current_idx = 0;
        while (std::getline(header_stream, col, ',')) {
            if (col == "seroprevalence_day64") {
                sero_col_idx = current_idx;
                break;
            }
            current_idx++;
        }
        
        if (sero_col_idx >= 0) {
            while (std::getline(file, line)) {
                std::istringstream data_stream(line);
                std::string value;
                for (int i = 0; i <= sero_col_idx; ++i) {
                    std::getline(data_stream, value, ',');
                }
                if (!value.empty()) {
                    sero_values.push_back(std::stod(value));
                }
            }
        }
        file.close();
    }
    
    // Create lightweight metrics for validation
    std::vector<EssentialMetrics> lightweight_metrics;
    for (double sero : sero_values) {
        EssentialMetrics m;
        m.seroprevalence_at_target_day = sero;
        lightweight_metrics.push_back(m);
    }
    
    validateAgainstENECOVID(lightweight_metrics);
}

void PostCalibrationAnalyser::savePosteriorPredictiveCheckData(
    const PosteriorPredictiveData& ppd_data,
    const std::string& sub_directory) {
    
    ensureOutputSubdirectoryExists(sub_directory);
    
    auto saveIncidenceData = [&](const PosteriorPredictiveData::IncidenceData& data,
                                const std::string& base_name) {
        auto saveMatrix = [&](const Eigen::MatrixXd& matrix, const std::string& suffix) {
            std::string filepath = FileUtils::joinPaths(
                FileUtils::joinPaths(output_dir_base_, sub_directory),
                base_name + "_" + suffix + ".csv");
            
            std::ofstream file(filepath);
            file << "time";
            for (int age = 0; age < epidemic::constants::DEFAULT_NUM_AGE_CLASSES; ++age) {
                file << ",age_" << age;
            }
            file << "\n";
            
            for (size_t t = 0; t < ppd_data.time_points.size(); ++t) {
                file << ppd_data.time_points[t];
                for (int age = 0; age < epidemic::constants::DEFAULT_NUM_AGE_CLASSES; ++age) {
                    file << "," << std::fixed << std::setprecision(6) << matrix(t, age);
                }
                file << "\n";
            }
            file.close();
        };
        
        saveMatrix(data.median, "median");
        saveMatrix(data.lower_90, "lower90");
        saveMatrix(data.upper_90, "upper90");
        saveMatrix(data.lower_95, "lower95");
        saveMatrix(data.upper_95, "upper95");
        saveMatrix(data.observed, "observed");
    };
    
    saveIncidenceData(ppd_data.daily_hospitalizations, "daily_hospitalizations");
    saveIncidenceData(ppd_data.daily_icu_admissions, "daily_icu_admissions");
    saveIncidenceData(ppd_data.daily_deaths, "daily_deaths");
    saveIncidenceData(ppd_data.cumulative_hospitalizations, "cumulative_hospitalizations");
    saveIncidenceData(ppd_data.cumulative_icu_admissions, "cumulative_icu_admissions");
    saveIncidenceData(ppd_data.cumulative_deaths, "cumulative_deaths");
    
    Logger::getInstance().info("PostCalibrationAnalyser", 
        "Posterior predictive check data saved to: " + sub_directory);
}

Eigen::MatrixXd PostCalibrationAnalyser::calculateDailyIncidenceFlow(
    const SimulationResult& sim_result,
    const SEPAIHRDParameters& params,
    const std::string& type) {
    
    int T = sim_result.time_points.size();
    Eigen::MatrixXd daily_flow = Eigen::MatrixXd::Zero(T, num_age_classes_);
    
    for (int t = 0; t < T; ++t) {
        Eigen::Map<const Eigen::VectorXd> I_t(&sim_result.solution[t][4 * num_age_classes_], num_age_classes_);
        Eigen::Map<const Eigen::VectorXd> H_t(&sim_result.solution[t][5 * num_age_classes_], num_age_classes_);
        Eigen::Map<const Eigen::VectorXd> ICU_t(&sim_result.solution[t][6 * num_age_classes_], num_age_classes_);
        
        for (int age = 0; age < num_age_classes_; ++age) {
            if (type == "hospitalizations") {
                daily_flow(t, age) = params.h(age) * I_t(age);
            } else if (type == "icu") {
                daily_flow(t, age) = params.icu(age) * H_t(age);
            } else if (type == "deaths") {
                daily_flow(t, age) = params.d_H(age) * H_t(age) + params.d_ICU(age) * ICU_t(age);
            }
        }
    }
    
    return daily_flow.cwiseMax(0.0);
}

} // namespace epidemic