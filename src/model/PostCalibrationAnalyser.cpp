#include "model/PostCalibrationAnalyser.hpp"
#include "model/AgeSEPAIHRDsimulator.hpp"
#include "model/ReproductionNumberCalculator.hpp"
#include "sir_age_structured/SimulationResultProcessor.hpp"
#include "utils/FileUtils.hpp"
#include "utils/Logger.hpp"
#include <fstream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <random>

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
    if (initial_state_.size() == 0) throw std::invalid_argument("PostCalibrationAnalyser: Initial state cannot be empty.");
    if (initial_state_.size() != model_template_->getStateSize()) {
        throw std::invalid_argument("PostCalibrationAnalyser: Initial state size does not match model state size.");
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
    int thinning_for_summary) {

    Logger::getInstance().info("PostCalibrationAnalyser", "Starting full report generation...");

    ensureOutputSubdirectoryExists("posterior_predictive");
    Logger::getInstance().info("PostCalibrationAnalyser", "Generating Posterior Predictive Checks...");
    PosteriorPredictiveData ppd_data = generatePosteriorPredictiveChecks(
        param_samples, param_manager, num_samples_for_posterior_pred
    );
    savePosteriorPredictiveCheckData(ppd_data, "posterior_predictive");
    Logger::getInstance().info("PostCalibrationAnalyser", "Posterior Predictive Checks saved.");

    ensureOutputSubdirectoryExists("mcmc_aggregated");
    Logger::getInstance().info("PostCalibrationAnalyser", "Analyzing MCMC runs for aggregated metrics...");
    std::vector<PostCalibrationMetrics> all_mcmc_metrics = analyzeMCMCRuns(
        param_samples, param_manager, -1, burn_in_for_summary, thinning_for_summary, false
    );
    if (!all_mcmc_metrics.empty()) {
        saveAggregatedMCMCMetrics(all_mcmc_metrics, "mcmc_aggregated");
        Logger::getInstance().info("PostCalibrationAnalyser", "Aggregated MCMC metrics saved.");

        ensureOutputSubdirectoryExists("seroprevalence");
        Logger::getInstance().info("PostCalibrationAnalyser", "Performing ENE-COVID validation...");
        validateAgainstENECOVID(all_mcmc_metrics);
        Logger::getInstance().info("PostCalibrationAnalyser", "ENE-COVID validation saved.");
    } else {
        Logger::getInstance().warning("PostCalibrationAnalyser", "No MCMC metrics generated, skipping ENE-COVID validation and aggregated saves.");
    }

    ensureOutputSubdirectoryExists("parameter_posteriors");
    Logger::getInstance().info("PostCalibrationAnalyser", "Saving parameter posteriors...");
    saveParameterPosteriors(param_samples, param_manager, burn_in_for_summary, thinning_for_summary);
    Logger::getInstance().info("PostCalibrationAnalyser", "Parameter posteriors saved.");

    ensureOutputSubdirectoryExists("scenarios");
    if (!param_samples.empty()) {
        Logger::getInstance().info("PostCalibrationAnalyser", "Preparing for Scenario Analysis...");
        Eigen::VectorXd mean_params = Eigen::VectorXd::Zero(param_manager.getParameterCount());
        int count = 0;
        for (size_t i = burn_in_for_summary; i < param_samples.size(); i += thinning_for_summary) {
            mean_params += param_samples[i];
            count++;
        }
        if (count > 0) mean_params /= count;
        else if (!param_samples.empty()) mean_params = param_samples.back();
        else {
             Logger::getInstance().error("PostCalibrationAnalyser", "No samples to derive mean parameters for scenarios.");
             return;
        }

        param_manager.updateModelParameters(mean_params);
        SEPAIHRDParameters baseline_scenario_params = model_template_->getModelParameters();

        std::vector<std::pair<std::string, SEPAIHRDParameters>> scenarios;
        SEPAIHRDParameters stricter_params = baseline_scenario_params;
        if (stricter_params.kappa_values.size() > 1) {
            stricter_params.kappa_values[1] = std::max(0.0, stricter_params.kappa_values[1] * 0.9);
        }
        scenarios.push_back({"stricter_lockdown_k2_10pct", stricter_params});

        SEPAIHRDParameters weaker_params = baseline_scenario_params;
        if (weaker_params.kappa_values.size() > 1) {
            weaker_params.kappa_values[1] *= 1.1;
        }
        scenarios.push_back({"weaker_lockdown_k2_p10pct", weaker_params});
        
        SEPAIHRDParameters earlier_lockdown_params = baseline_scenario_params;
        if (earlier_lockdown_params.kappa_end_times.size() > 1) {
            earlier_lockdown_params.kappa_end_times[0] = std::max(0.0, earlier_lockdown_params.kappa_end_times[0] - 7.0);
            if (earlier_lockdown_params.kappa_end_times.size() > 0 && earlier_lockdown_params.kappa_end_times[1] <= earlier_lockdown_params.kappa_end_times[0]) {
                 earlier_lockdown_params.kappa_end_times[1] = earlier_lockdown_params.kappa_end_times[0] + 7.0;
            }
        }
        scenarios.push_back({"earlier_lockdown_p2_1wk", earlier_lockdown_params});

        performScenarioAnalysis(baseline_scenario_params, scenarios);
        Logger::getInstance().info("PostCalibrationAnalyser", "Scenario Analysis completed and saved.");
    } else {
        Logger::getInstance().warning("PostCalibrationAnalyser", "No MCMC samples provided, skipping scenario analysis.");
    }

    Logger::getInstance().info("PostCalibrationAnalyser", "Full report generation finished.");
}


PostCalibrationMetrics PostCalibrationAnalyser::analyzeSingleRun(
    const SEPAIHRDParameters& params, const std::string& run_id) {
    
    auto run_npi_strategy = model_template_->getNpiStrategy()->clone(); 

    auto run_model = std::make_shared<AgeSEPAIHRDModel>(params, run_npi_strategy);

    AgeSEPAIHRDSimulator simulator(run_model, solver_strategy_,
                                  time_points_.front(), time_points_.back(), 1.0, // dt_hint = 1.0
                                  1e-6, 1e-6); // Default abs/rel error

    SimulationResult sim_result = simulator.run(initial_state_, time_points_);

    PostCalibrationMetrics metrics = processSimulationResults(sim_result, params, run_id);
    return metrics;
}

PostCalibrationMetrics PostCalibrationAnalyser::processSimulationResults(
    const SimulationResult& sim_result,
    const SEPAIHRDParameters& params,
    const std::string& run_id) {
    (void)run_id;

    PostCalibrationMetrics metrics;
    metrics.kappa_values.clear(); 

    auto current_npi_strategy = model_template_->getNpiStrategy()->clone();
    auto current_model_for_rn = std::make_shared<AgeSEPAIHRDModel>(params, current_npi_strategy);
    ReproductionNumberCalculator rn_calc(current_model_for_rn);

    metrics.R0 = rn_calc.calculateR0();

    metrics.Rt_time = sim_result.time_points;
    metrics.Rt_median.resize(sim_result.time_points.size());
    for (size_t t = 0; t < sim_result.time_points.size(); ++t) {
        Eigen::Map<const Eigen::VectorXd> S_t(&sim_result.solution[t][0 * num_age_classes_], num_age_classes_);
        metrics.Rt_median[t] = rn_calc.calculateRt(S_t, sim_result.time_points[t]);
    }
    
    int num_model_compartments = AgeSEPAIHRDSimulator::NUM_COMPARTMENTS; // S,E,P,A,I,H,ICU,R,D = 9
    metrics.hidden_compartments_median["E"] = SimulationResultProcessor::getCompartmentData(sim_result, *current_model_for_rn, "E", num_model_compartments);
    metrics.hidden_compartments_median["P"] = SimulationResultProcessor::getCompartmentData(sim_result, *current_model_for_rn, "P", num_model_compartments);
    metrics.hidden_compartments_median["A"] = SimulationResultProcessor::getCompartmentData(sim_result, *current_model_for_rn, "A", num_model_compartments);
    metrics.hidden_compartments_median["I"] = SimulationResultProcessor::getCompartmentData(sim_result, *current_model_for_rn, "I", num_model_compartments);
    metrics.hidden_compartments_median["R"] = SimulationResultProcessor::getCompartmentData(sim_result, *current_model_for_rn, "R", num_model_compartments);

    Eigen::MatrixXd H_comp = SimulationResultProcessor::getCompartmentData(sim_result, *current_model_for_rn, "H", num_model_compartments);
    Eigen::MatrixXd ICU_comp = SimulationResultProcessor::getCompartmentData(sim_result, *current_model_for_rn, "ICU", num_model_compartments);
    
    Eigen::MatrixXd age_specific_prevalence(sim_result.time_points.size(), num_age_classes_);
    Eigen::VectorXd overall_prevalence_vec = Eigen::VectorXd::Zero(sim_result.time_points.size());
    double total_population_N = params.N.sum();

    for (size_t t = 0; t < sim_result.time_points.size(); ++t) {
        double total_active_this_timestep = 0;
        for (int age = 0; age < num_age_classes_; ++age) {
            double active_in_age = metrics.hidden_compartments_median["P"](t, age) +
                                   metrics.hidden_compartments_median["A"](t, age) +
                                   metrics.hidden_compartments_median["I"](t, age) +
                                   H_comp(t, age) +
                                   ICU_comp(t, age);
            age_specific_prevalence(t, age) = (params.N(age) > 0) ? active_in_age / params.N(age) : 0.0;
            total_active_this_timestep += active_in_age;
        }
        overall_prevalence_vec(t) = (total_population_N > 0) ? total_active_this_timestep / total_population_N : 0.0;
    }
    metrics.prevalence_trajectories_median["age_specific"] = age_specific_prevalence;
    metrics.prevalence_trajectories_median["overall"] = overall_prevalence_vec;


    Eigen::VectorXd cumulative_infections_age = Eigen::VectorXd::Zero(num_age_classes_);
    Eigen::VectorXd cumulative_hosp_age = Eigen::VectorXd::Zero(num_age_classes_);
    Eigen::VectorXd cumulative_icu_age = Eigen::VectorXd::Zero(num_age_classes_);
    metrics.seroprevalence_time = sim_result.time_points;
    metrics.seroprevalence_median.resize(sim_result.time_points.size(), 0.0);

    Eigen::VectorXd current_S = initial_state_.head(num_age_classes_);
    Eigen::VectorXd current_E = initial_state_.segment(1 * num_age_classes_, num_age_classes_);
    Eigen::VectorXd current_P = initial_state_.segment(2 * num_age_classes_, num_age_classes_);
    Eigen::VectorXd current_A = initial_state_.segment(3 * num_age_classes_, num_age_classes_);
    Eigen::VectorXd current_I = initial_state_.segment(4 * num_age_classes_, num_age_classes_);
    Eigen::VectorXd current_H = initial_state_.segment(5 * num_age_classes_, num_age_classes_);

    for (size_t t_idx = 0; t_idx < sim_result.time_points.size(); ++t_idx) {
        double time_t = sim_result.time_points[t_idx];
        double dt = (t_idx > 0) ? (time_t - sim_result.time_points[t_idx - 1]) : 0.0;
        if (t_idx == 0 && sim_result.time_points.size() > 1) { 
             dt = sim_result.time_points[1] - sim_result.time_points[0];
        }


        for(int age=0; age < num_age_classes_; ++age) {
            current_S(age) = sim_result.solution[t_idx][0 * num_age_classes_ + age];
            current_P(age) = sim_result.solution[t_idx][2 * num_age_classes_ + age];
            current_A(age) = sim_result.solution[t_idx][3 * num_age_classes_ + age];
            current_I(age) = sim_result.solution[t_idx][4 * num_age_classes_ + age];
            current_H(age) = sim_result.solution[t_idx][5 * num_age_classes_ + age];
        }
        
        double kappa_t = current_model_for_rn->getNpiStrategy()->getReductionFactor(time_t);
        Eigen::VectorXd infectious_load_per_capita = Eigen::VectorXd::Zero(num_age_classes_);
        for (int j = 0; j < num_age_classes_; ++j) {
            if (params.N(j) > 1e-9) {
                infectious_load_per_capita(j) = (current_P(j) + current_A(j) + params.theta * current_I(j)) / params.N(j);
            }
        }
        Eigen::VectorXd lambda_t = params.beta * kappa_t * params.M_baseline * infectious_load_per_capita;
        lambda_t = lambda_t.cwiseMax(0.0);
        Eigen::VectorXd new_infections_rate_t = lambda_t.array() * current_S.array();

        if (dt > 0) {
            cumulative_infections_age += new_infections_rate_t * dt;
            cumulative_hosp_age += (params.h.array() * current_I.array() * dt).matrix();
            cumulative_icu_age += (params.icu.array() * current_H.array() * dt).matrix();
        }
        
        if (total_population_N > 0) {
            metrics.seroprevalence_median[t_idx] = cumulative_infections_age.sum() / total_population_N;
        }
    }

    Eigen::Map<const Eigen::VectorXd> D_final(&sim_result.solution.back()[8 * num_age_classes_], num_age_classes_);
    Eigen::Map<const Eigen::VectorXd> D_initial(&initial_state_[8 * num_age_classes_], num_age_classes_);
    Eigen::VectorXd cumulative_deaths_age = D_final - D_initial;
    metrics.total_cumulative_deaths = cumulative_deaths_age.sum();


    for (int age = 0; age < num_age_classes_; ++age) {
        std::string age_key = "age_" + std::to_string(age);
        metrics.age_specific_attack_rate[age_key] = (params.N(age) > 0) ? cumulative_infections_age(age) / params.N(age) : 0.0;
        if (cumulative_infections_age(age) > 1e-9) {
            metrics.age_specific_IFR[age_key] = cumulative_deaths_age(age) / cumulative_infections_age(age);
            metrics.age_specific_IHR[age_key] = cumulative_hosp_age(age) / cumulative_infections_age(age);
            metrics.age_specific_IICUR[age_key] = cumulative_icu_age(age) / cumulative_infections_age(age);
        } else {
            metrics.age_specific_IFR[age_key] = 0.0;
            metrics.age_specific_IHR[age_key] = 0.0;
            metrics.age_specific_IICUR[age_key] = 0.0;
        }
    }
    
    metrics.overall_attack_rate = (total_population_N > 0) ? cumulative_infections_age.sum() / total_population_N : 0.0;
    metrics.overall_IFR = (cumulative_infections_age.sum() > 1e-9) ? cumulative_deaths_age.sum() / cumulative_infections_age.sum() : 0.0;

    metrics.peak_hospital_occupancy = 0.0;
    metrics.peak_ICU_occupancy = 0.0;
    for(size_t t_idx = 0; t_idx < sim_result.time_points.size(); ++t_idx) {
        double current_total_H = H_comp.row(t_idx).sum();
        double current_total_ICU = ICU_comp.row(t_idx).sum();
        if (current_total_H > metrics.peak_hospital_occupancy) {
            metrics.peak_hospital_occupancy = current_total_H;
            metrics.time_to_peak_hospital = sim_result.time_points[t_idx];
        }
        if (current_total_ICU > metrics.peak_ICU_occupancy) {
            metrics.peak_ICU_occupancy = current_total_ICU;
            metrics.time_to_peak_ICU = sim_result.time_points[t_idx];
        }
    }

    if (params.kappa_values.size() == params.kappa_end_times.size()) {
        for (size_t i = 0; i < params.kappa_values.size(); ++i) {
             metrics.kappa_values["kappa_" + std::to_string(i + 1)] = params.kappa_values[i];
        }
    }
    return metrics;
}


std::vector<PostCalibrationMetrics> PostCalibrationAnalyser::analyzeMCMCRuns(
    const std::vector<Eigen::VectorXd>& param_samples,
    IParameterManager& param_manager,
    int num_samples_to_process,
    int burn_in,
    int thinning,
    bool save_individual_mcmc_run_details) {

    std::vector<PostCalibrationMetrics> all_metrics;
    if (param_samples.empty()) {
        Logger::getInstance().warning("PostCalibrationAnalyser::analyzeMCMCRuns", "No MCMC samples provided.");
        return all_metrics;
    }

    int n_params = param_manager.getParameterCount();
    int total_available_samples = param_samples.size();
    
    int effective_start_index = burn_in;
    if (effective_start_index >= total_available_samples) {
        Logger::getInstance().warning("PostCalibrationAnalyser::analyzeMCMCRuns", "Burn-in period is too large for the number of samples.");
        return all_metrics;
    }

    int samples_after_burn_in = total_available_samples - effective_start_index;
    int max_possible_thinned_samples = (samples_after_burn_in + thinning -1) / thinning;


    int actual_samples_to_process = (num_samples_to_process == -1) ? max_possible_thinned_samples : std::min(num_samples_to_process, max_possible_thinned_samples);
    
    all_metrics.reserve(actual_samples_to_process);
    
    Logger::getInstance().info("PostCalibrationAnalyser::analyzeMCMCRuns", "Processing " + std::to_string(actual_samples_to_process) +
                               " MCMC samples (Burn-in: " + std::to_string(burn_in) + ", Thinning: " + std::to_string(thinning) + ").");

    for (int i = 0; i < actual_samples_to_process; ++i) {
        int sample_idx = effective_start_index + i * thinning;
        if (sample_idx >= total_available_samples) break; 

        const Eigen::VectorXd& current_param_vec = param_samples[sample_idx];
        if (current_param_vec.size() != n_params) {
            Logger::getInstance().warning("PostCalibrationAnalyser::analyzeMCMCRuns", "Parameter vector size mismatch for sample " + std::to_string(sample_idx) + ". Skipping.");
            continue;
        }

        param_manager.updateModelParameters(current_param_vec);
        SEPAIHRDParameters current_params_struct = model_template_->getModelParameters(); // Get full struct after update

        std::string run_id = "mcmc_sample_" + std::to_string(sample_idx);
        PostCalibrationMetrics metrics = analyzeSingleRun(current_params_struct, run_id);
        all_metrics.push_back(metrics);

        if (save_individual_mcmc_run_details) {
            ensureOutputSubdirectoryExists("mcmc_individual_runs/" + run_id);
            saveScalarMetricsCSV(FileUtils::joinPaths(FileUtils::joinPaths(output_dir_base_, "mcmc_individual_runs/" + run_id), "scalar_metrics.csv"), metrics);
        }
        
        if ((i + 1) % (std::max(1, actual_samples_to_process / 10)) == 0) {
            Logger::getInstance().info("PostCalibrationAnalyser::analyzeMCMCRuns", "Processed " + std::to_string(i + 1) + "/" + std::to_string(actual_samples_to_process) + " samples.");
        }
    }
    return all_metrics;
}

PosteriorPredictiveData PostCalibrationAnalyser::generatePosteriorPredictiveChecks(
    const std::vector<Eigen::VectorXd>& param_samples,
    IParameterManager& param_manager,
    int num_samples_for_ppc) {

    PosteriorPredictiveData ppd_data;
    ppd_data.time_points = time_points_;

    ppd_data.daily_hospitalizations.observed = observed_data_.getNewHospitalizations();
    ppd_data.daily_icu_admissions.observed = observed_data_.getNewICU();
    ppd_data.daily_deaths.observed = observed_data_.getNewDeaths();
    ppd_data.cumulative_hospitalizations.observed = observed_data_.getCumulativeHospitalizations();
    ppd_data.cumulative_icu_admissions.observed = observed_data_.getCumulativeICU();
    ppd_data.cumulative_deaths.observed = observed_data_.getCumulativeDeaths();
    
    if (param_samples.empty()) {
        Logger::getInstance().warning("PostCalibrationAnalyser::generatePosteriorPredictiveChecks", "No MCMC samples provided.");
        int T = time_points_.size();
        int A = num_age_classes_;
        auto nan_matrix = Eigen::MatrixXd::Constant(T, A, std::nan(""));
        ppd_data.daily_hospitalizations.median = nan_matrix; 
        ppd_data.daily_hospitalizations.lower_90 = nan_matrix;
        ppd_data.daily_hospitalizations.upper_90 = nan_matrix;
        ppd_data.daily_hospitalizations.lower_95 = nan_matrix;
        ppd_data.daily_hospitalizations.upper_95 = nan_matrix;
        ppd_data.daily_icu_admissions.median = nan_matrix;
        ppd_data.daily_icu_admissions.lower_90 = nan_matrix;
        ppd_data.daily_icu_admissions.upper_90 = nan_matrix;
        ppd_data.daily_icu_admissions.lower_95 = nan_matrix;
        ppd_data.daily_icu_admissions.upper_95 = nan_matrix;
        ppd_data.daily_deaths.median = nan_matrix;
        ppd_data.daily_deaths.lower_90 = nan_matrix;
        ppd_data.daily_deaths.upper_90 = nan_matrix;
        ppd_data.daily_deaths.lower_95 = nan_matrix;
        ppd_data.daily_deaths.upper_95 = nan_matrix;
        ppd_data.cumulative_hospitalizations.median = nan_matrix;
        ppd_data.cumulative_hospitalizations.lower_90 = nan_matrix;
        ppd_data.cumulative_hospitalizations.upper_90 = nan_matrix;
        ppd_data.cumulative_hospitalizations.lower_95 = nan_matrix;
        ppd_data.cumulative_hospitalizations.upper_95 = nan_matrix;
        ppd_data.cumulative_icu_admissions.median = nan_matrix;
        ppd_data.cumulative_icu_admissions.lower_90 = nan_matrix;
        ppd_data.cumulative_icu_admissions.upper_90 = nan_matrix;
        ppd_data.cumulative_icu_admissions.lower_95 = nan_matrix;
        ppd_data.cumulative_icu_admissions.upper_95 = nan_matrix;
        ppd_data.cumulative_deaths.median = nan_matrix;
        ppd_data.cumulative_deaths.lower_90 = nan_matrix;
        ppd_data.cumulative_deaths.upper_90 = nan_matrix;
        ppd_data.cumulative_deaths.lower_95 = nan_matrix;
        ppd_data.cumulative_deaths.upper_95 = nan_matrix;
        return ppd_data;
    }

    std::vector<int> selected_indices;
    if (num_samples_for_ppc > 0 && static_cast<size_t>(num_samples_for_ppc) < param_samples.size()) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(0, param_samples.size() - 1);
        for (int i = 0; i < num_samples_for_ppc; ++i) selected_indices.push_back(distrib(gen));
    } else {
        selected_indices.resize(param_samples.size());
        std::iota(selected_indices.begin(), selected_indices.end(), 0);
    }

    std::vector<Eigen::MatrixXd> daily_hosp_preds, daily_icu_preds, daily_death_preds;
    std::vector<Eigen::MatrixXd> cumul_hosp_preds, cumul_icu_preds, cumul_death_preds;

    for (int sample_master_idx : selected_indices) {
        const Eigen::VectorXd& p_vec = param_samples[sample_master_idx];
        param_manager.updateModelParameters(p_vec);
        SEPAIHRDParameters current_params_struct = model_template_->getModelParameters();

        auto run_npi_strategy = model_template_->getNpiStrategy()->clone();
        auto run_model = std::make_shared<AgeSEPAIHRDModel>(current_params_struct, run_npi_strategy);
        AgeSEPAIHRDSimulator simulator(run_model, solver_strategy_, time_points_.front(), time_points_.back(), 1.0, 1e-6, 1e-6);
        SimulationResult sim_result = simulator.run(initial_state_, time_points_);

        Eigen::MatrixXd dh = calculateDailyIncidenceFlow(sim_result, current_params_struct, "hospitalizations");
        Eigen::MatrixXd di = calculateDailyIncidenceFlow(sim_result, current_params_struct, "icu");
        Eigen::MatrixXd dd = calculateDailyIncidenceFlow(sim_result, current_params_struct, "deaths");
        daily_hosp_preds.push_back(dh);
        daily_icu_preds.push_back(di);
        daily_death_preds.push_back(dd);

        cumul_hosp_preds.push_back(calculateCumulativeFromDaily(dh));
        cumul_icu_preds.push_back(calculateCumulativeFromDaily(di));
        cumul_death_preds.push_back(calculateCumulativeFromDaily(dd));
    }

    if (!daily_hosp_preds.empty()) {
        ppd_data.daily_hospitalizations.median = getQuantileMatrix(daily_hosp_preds, 0.5);
        ppd_data.daily_hospitalizations.lower_90 = getQuantileMatrix(daily_hosp_preds, 0.05);
        ppd_data.daily_hospitalizations.upper_90 = getQuantileMatrix(daily_hosp_preds, 0.95);
        ppd_data.daily_hospitalizations.lower_95 = getQuantileMatrix(daily_hosp_preds, 0.025);
        ppd_data.daily_hospitalizations.upper_95 = getQuantileMatrix(daily_hosp_preds, 0.975);
        
        ppd_data.daily_icu_admissions.median = getQuantileMatrix(daily_icu_preds, 0.5);
        ppd_data.daily_icu_admissions.lower_90 = getQuantileMatrix(daily_icu_preds, 0.05);
        ppd_data.daily_icu_admissions.upper_90 = getQuantileMatrix(daily_icu_preds, 0.95);
        ppd_data.daily_icu_admissions.lower_95 = getQuantileMatrix(daily_icu_preds, 0.025);
        ppd_data.daily_icu_admissions.upper_95 = getQuantileMatrix(daily_icu_preds, 0.975);
        
        ppd_data.daily_deaths.median = getQuantileMatrix(daily_death_preds, 0.5);
        ppd_data.daily_deaths.lower_90 = getQuantileMatrix(daily_death_preds, 0.05);
        ppd_data.daily_deaths.upper_90 = getQuantileMatrix(daily_death_preds, 0.95);
        ppd_data.daily_deaths.lower_95 = getQuantileMatrix(daily_death_preds, 0.025);
        ppd_data.daily_deaths.upper_95 = getQuantileMatrix(daily_death_preds, 0.975);
        
        ppd_data.cumulative_hospitalizations.median = getQuantileMatrix(cumul_hosp_preds, 0.5);
        ppd_data.cumulative_hospitalizations.lower_90 = getQuantileMatrix(cumul_hosp_preds, 0.05);
        ppd_data.cumulative_hospitalizations.upper_90 = getQuantileMatrix(cumul_hosp_preds, 0.95);
        ppd_data.cumulative_hospitalizations.lower_95 = getQuantileMatrix(cumul_hosp_preds, 0.025);
        ppd_data.cumulative_hospitalizations.upper_95 = getQuantileMatrix(cumul_hosp_preds, 0.975);
        
        ppd_data.cumulative_icu_admissions.median = getQuantileMatrix(cumul_icu_preds, 0.5);
        ppd_data.cumulative_icu_admissions.lower_90 = getQuantileMatrix(cumul_icu_preds, 0.05);
        ppd_data.cumulative_icu_admissions.upper_90 = getQuantileMatrix(cumul_icu_preds, 0.95);
        ppd_data.cumulative_icu_admissions.lower_95 = getQuantileMatrix(cumul_icu_preds, 0.025);
        ppd_data.cumulative_icu_admissions.upper_95 = getQuantileMatrix(cumul_icu_preds, 0.975);
        
        ppd_data.cumulative_deaths.median = getQuantileMatrix(cumul_death_preds, 0.5);
        ppd_data.cumulative_deaths.lower_90 = getQuantileMatrix(cumul_death_preds, 0.05);
        ppd_data.cumulative_deaths.upper_90 = getQuantileMatrix(cumul_death_preds, 0.95);
        ppd_data.cumulative_deaths.lower_95 = getQuantileMatrix(cumul_death_preds, 0.025);
        ppd_data.cumulative_deaths.upper_95 = getQuantileMatrix(cumul_death_preds, 0.975);
        
        Logger::getInstance().info("PostCalibrationAnalyser", "Posterior predictive checks completed using " + 
                                  std::to_string(selected_indices.size()) + " samples.");
    } else {
        Logger::getInstance().warning("PostCalibrationAnalyser", "No predictions generated for posterior predictive checks.");
    }

    return ppd_data;
}


void PostCalibrationAnalyser::performScenarioAnalysis(
    const SEPAIHRDParameters& baseline_params,
    const std::vector<std::pair<std::string, SEPAIHRDParameters>>& scenarios) {

    Logger::getInstance().info("PostCalibrationAnalyser", "Performing scenario analysis...");
    ensureOutputSubdirectoryExists("scenarios");

    PostCalibrationMetrics baseline_metrics = analyzeSingleRun(baseline_params, "scenario_baseline");

    std::vector<std::pair<std::string, PostCalibrationMetrics>> scenario_metrics_list;
    for (const auto& scenario_pair : scenarios) {
        const std::string& scenario_name = scenario_pair.first;
        const SEPAIHRDParameters& scenario_params = scenario_pair.second;
        Logger::getInstance().info("PostCalibrationAnalyser", "Analyzing scenario: " + scenario_name);
        PostCalibrationMetrics metrics = analyzeSingleRun(scenario_params, "scenario_" + scenario_name);
        scenario_metrics_list.push_back({scenario_name, metrics});
    }
    
    saveScenarioComparisonCSV(baseline_metrics, scenario_metrics_list, "scenarios");

    std::string death_traj_path = FileUtils::joinPaths(FileUtils::joinPaths(output_dir_base_, "scenarios"), "cumulative_deaths_trajectories.csv");
    std::ofstream d_file(death_traj_path);
    d_file << "time,baseline_deaths";
    for(const auto& sm_pair : scenario_metrics_list) d_file << "," << sm_pair.first << "_deaths";
    d_file << "\n";

    auto get_D_trajectory = [&](const SEPAIHRDParameters& p_struct) {
        auto npi_s = model_template_->getNpiStrategy()->clone();
        auto m = std::make_shared<AgeSEPAIHRDModel>(p_struct, npi_s);
        AgeSEPAIHRDSimulator sim(m, solver_strategy_, time_points_.front(), time_points_.back(), 1.0, 1e-6, 1e-6);
        SimulationResult sr = sim.run(initial_state_, time_points_);
        Eigen::MatrixXd D_comp = SimulationResultProcessor::getCompartmentData(sr, *m, "D", AgeSEPAIHRDSimulator::NUM_COMPARTMENTS);
        return D_comp.rowwise().sum();
    };
    
    Eigen::VectorXd baseline_D_traj = get_D_trajectory(baseline_params);
    std::vector<Eigen::VectorXd> scenario_D_trajs;
    for(const auto& scen_pair : scenarios) scenario_D_trajs.push_back(get_D_trajectory(scen_pair.second));

    for(size_t t=0; t < time_points_.size(); ++t) {
        d_file << time_points_[t] << "," << baseline_D_traj(t);
        for(const auto& scen_D_traj : scenario_D_trajs) d_file << "," << scen_D_traj(t);
        d_file << "\n";
    }
    d_file.close();
    Logger::getInstance().info("PostCalibrationAnalyser", "Scenario death trajectories saved to: " + death_traj_path);


    std::string rt_traj_path = FileUtils::joinPaths(FileUtils::joinPaths(output_dir_base_, "scenarios"), "rt_trajectories.csv");
    std::ofstream rt_file(rt_traj_path);
    rt_file << "time,baseline_rt";
     for(const auto& sm_pair : scenario_metrics_list) rt_file << "," << sm_pair.first << "_rt";
    rt_file << "\n";
    for(size_t t=0; t<baseline_metrics.Rt_time.size(); ++t) {
        rt_file << baseline_metrics.Rt_time[t] << "," << baseline_metrics.Rt_median[t];
        for(const auto& sm_pair : scenario_metrics_list) {
            if (t < sm_pair.second.Rt_median.size()) rt_file << "," << sm_pair.second.Rt_median[t];
            else rt_file << ","; // Missing data
        }
        rt_file << "\n";
    }
    rt_file.close();
    Logger::getInstance().info("PostCalibrationAnalyser", "Scenario Rt trajectories saved to: " + rt_traj_path);


    Logger::getInstance().info("PostCalibrationAnalyser", "Scenario analysis complete.");
}

void PostCalibrationAnalyser::validateAgainstENECOVID(
    const std::vector<PostCalibrationMetrics>& all_run_metrics,
    double ene_covid_target_day,
    double ene_covid_mean,
    double ene_covid_lower_ci,
    double ene_covid_upper_ci) {

    ensureOutputSubdirectoryExists("seroprevalence");
    std::string filepath = FileUtils::joinPaths(FileUtils::joinPaths(output_dir_base_, "seroprevalence"), "ene_covid_validation.csv");
    
    if (all_run_metrics.empty()) {
        Logger::getInstance().warning("PostCalibrationAnalyser::validateAgainstENECOVID", "No metrics provided for ENE-COVID validation.");
        return;
    }

    std::vector<double> model_seroprevalences_at_target_day;
    for (const auto& metrics : all_run_metrics) {
        size_t target_idx = static_cast<size_t>(-1);
        for (size_t t = 0; t < metrics.seroprevalence_time.size(); ++t) {
            if (std::abs(metrics.seroprevalence_time[t] - ene_covid_target_day) < 0.5) { // Find closest time point
                target_idx = t;
                break;
            }
        }
        if (target_idx != static_cast<size_t>(-1) && target_idx < metrics.seroprevalence_median.size()) {
            model_seroprevalences_at_target_day.push_back(metrics.seroprevalence_median[target_idx]);
        }
    }

    if (model_seroprevalences_at_target_day.empty()) {
        Logger::getInstance().warning("PostCalibrationAnalyser::validateAgainstENECOVID", "Could not find model seroprevalence at target day " + std::to_string(ene_covid_target_day));
        return;
    }

    std::sort(model_seroprevalences_at_target_day.begin(), model_seroprevalences_at_target_day.end());
    
    double model_median = getQuantiles(model_seroprevalences_at_target_day, {0.5})[0];
    double model_q025 = getQuantiles(model_seroprevalences_at_target_day, {0.025})[0];
    double model_q975 = getQuantiles(model_seroprevalences_at_target_day, {0.975})[0];
    
    std::ofstream file(filepath);
    file << "source,median_seroprevalence,lower_95ci,upper_95ci,target_day\n";
    file << std::fixed << std::setprecision(5);
    file << "Model," << model_median << "," << model_q025 << "," << model_q975 << "," << ene_covid_target_day << "\n";
    file << "ENE_COVID," << ene_covid_mean << "," << ene_covid_lower_ci << "," << ene_covid_upper_ci << "," << ene_covid_target_day << "\n";
    file.close();
    Logger::getInstance().info("PostCalibrationAnalyser", "ENE-COVID validation results saved to: " + filepath);
}


void PostCalibrationAnalyser::saveParameterPosteriors(
    const std::vector<Eigen::VectorXd>& param_samples,
    IParameterManager& param_manager,
    int burn_in,
    int thinning) {

    ensureOutputSubdirectoryExists("parameter_posteriors");
    std::string samples_filepath = FileUtils::joinPaths(FileUtils::joinPaths(output_dir_base_, "parameter_posteriors"), "posterior_samples.csv");
    std::string summary_filepath = FileUtils::joinPaths(FileUtils::joinPaths(output_dir_base_, "parameter_posteriors"), "posterior_summary.csv");

    const auto& param_names = param_manager.getParameterNames();

    std::ofstream sfile(samples_filepath);
    sfile << "sample_index";
    for(const auto& name : param_names) sfile << "," << name;
    sfile << "\n";

    int saved_sample_count = 0;
    for (size_t i = burn_in; i < param_samples.size(); i += thinning) {
        sfile << saved_sample_count++;
        for (int j = 0; j < param_samples[i].size(); ++j) {
            sfile << "," << std::scientific << std::setprecision(8) << param_samples[i][j];
        }
        sfile << "\n";
    }
    sfile.close();
    Logger::getInstance().info("PostCalibrationAnalyser", "Posterior samples saved to: " + samples_filepath);

    std::ofstream sumfile(summary_filepath);
    sumfile << "parameter,mean,median,std_dev,q025,q975\n";
    sumfile << std::fixed << std::setprecision(8);

    for (size_t p_idx = 0; p_idx < param_names.size(); ++p_idx) {
        std::vector<double> p_values;
        for (size_t i = burn_in; i < param_samples.size(); i += thinning) {
             if (p_idx < static_cast<size_t>(param_samples[i].size())) {
                p_values.push_back(param_samples[i][p_idx]);
            }
        }
        if (p_values.empty()) continue;

        std::sort(p_values.begin(), p_values.end());
        double mean = std::accumulate(p_values.begin(), p_values.end(), 0.0) / p_values.size();
        double median = getQuantiles(p_values, {0.5})[0];
        double q025 = getQuantiles(p_values, {0.025})[0];
        double q975 = getQuantiles(p_values, {0.975})[0];
        
        double sum_sq_diff = 0.0;
        for(double val : p_values) sum_sq_diff += (val - mean) * (val - mean);
        double std_dev = std::sqrt(sum_sq_diff / p_values.size());

        sumfile << param_names[p_idx] << "," << mean << "," << median << "," << std_dev << "," << q025 << "," << q975 << "\n";
    }
    sumfile.close();
    Logger::getInstance().info("PostCalibrationAnalyser", "Posterior summary saved to: " + summary_filepath);
}


// --- Private Helper Implementations ---

Eigen::MatrixXd PostCalibrationAnalyser::calculateDailyIncidenceFlow(
    const SimulationResult& sim_result,
    const SEPAIHRDParameters& params,
    const std::string& type) {
    
    int T = sim_result.time_points.size();
    Eigen::MatrixXd daily_flow = Eigen::MatrixXd::Zero(T, num_age_classes_);
    int num_model_compartments = AgeSEPAIHRDSimulator::NUM_COMPARTMENTS;

    Eigen::MatrixXd I_data = SimulationResultProcessor::getCompartmentData(sim_result, *model_template_, "I", num_model_compartments);
    Eigen::MatrixXd H_data = SimulationResultProcessor::getCompartmentData(sim_result, *model_template_, "H", num_model_compartments);
    Eigen::MatrixXd ICU_data = SimulationResultProcessor::getCompartmentData(sim_result, *model_template_, "ICU", num_model_compartments);


    for (int t = 0; t < T; ++t) {
        for (int age = 0; age < num_age_classes_; ++age) {
            if (type == "hospitalizations") {
                daily_flow(t, age) = params.h(age) * I_data(t, age);
            } else if (type == "icu") {
                daily_flow(t, age) = params.icu(age) * H_data(t, age);
            } else if (type == "deaths") {
                 daily_flow(t, age) = params.d_H(age) * H_data(t,age) + params.d_ICU(age) * ICU_data(t,age);
            }
        }
    }
     // For deaths, if we want *new* deaths, it's D(t) - D(t-1).
    // The D compartment in SEPAIHRD is cumulative.
    if (type == "deaths_from_D_compartment") { // Alternative death calculation
        Eigen::MatrixXd D_data_cumulative = SimulationResultProcessor::getCompartmentData(sim_result, *model_template_, "D", num_model_compartments);
        daily_flow.row(0) = D_data_cumulative.row(0) - initial_state_.segment(8 * num_age_classes_, num_age_classes_).transpose();
        for (int t = 1; t < T; ++t) {
            daily_flow.row(t) = D_data_cumulative.row(t) - D_data_cumulative.row(t-1);
        }
    }

    return daily_flow.cwiseMax(0.0); // Ensure non-negative counts
}

Eigen::MatrixXd PostCalibrationAnalyser::calculateCumulativeFromDaily(const Eigen::MatrixXd& daily_incidence) {
    Eigen::MatrixXd cumulative = daily_incidence;
    for (int t = 1; t < daily_incidence.rows(); ++t) {
        cumulative.row(t) += cumulative.row(t - 1);
    }
    return cumulative;
}

template<typename T>
std::vector<T> PostCalibrationAnalyser::getQuantiles(const std::vector<T>& sorted_values, const std::vector<double>& quantiles_probs) {
    std::vector<T> results;
    if (sorted_values.empty()) return results;
    for (double q_prob : quantiles_probs) {
        int idx = static_cast<int>(q_prob * (sorted_values.size() -1) ); // -1 for 0-based index from count
        idx = std::max(0, std::min(idx, static_cast<int>(sorted_values.size()) - 1));
        results.push_back(sorted_values[idx]);
    }
    return results;
}

Eigen::MatrixXd PostCalibrationAnalyser::getQuantileMatrix(const std::vector<Eigen::MatrixXd>& matrix_samples, double quantile_prob) {
    if (matrix_samples.empty()) return Eigen::MatrixXd();
    int rows = matrix_samples[0].rows();
    int cols = matrix_samples[0].cols();
    Eigen::MatrixXd quantile_matrix(rows, cols);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::vector<double> values_at_rc;
            values_at_rc.reserve(matrix_samples.size());
            for (const auto& mat : matrix_samples) {
                values_at_rc.push_back(mat(r, c));
            }
            std::sort(values_at_rc.begin(), values_at_rc.end());
            quantile_matrix(r, c) = getQuantiles(values_at_rc, {quantile_prob})[0];
        }
    }
    return quantile_matrix;
}


void PostCalibrationAnalyser::saveTimeSeriesCSV(
    const std::string& filepath,
    const std::vector<std::string>& headers,
    const std::vector<std::vector<double>>& data_columns) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        Logger::getInstance().error("PostCalibrationAnalyser", "Failed to open file for writing: " + filepath);
        return;
    }

    for (size_t i = 0; i < headers.size(); ++i) {
        file << headers[i] << (i == headers.size() - 1 ? "" : ",");
    }
    file << "\n";

    if (data_columns.empty() || data_columns[0].empty()) return;
    size_t num_rows = data_columns[0].size();

    for (size_t r = 0; r < num_rows; ++r) {
        for (size_t c = 0; c < data_columns.size(); ++c) {
            if (r < data_columns[c].size()) {
                 file << std::fixed << std::setprecision(6) << data_columns[c][r];
            } else {
                 file << ""; // Missing data
            }
            file << (c == data_columns.size() - 1 ? "" : ",");
        }
        file << "\n";
    }
    file.close();
}

void PostCalibrationAnalyser::saveMatrixTimeSeriesCSV(
    const std::string& filepath,
    const std::vector<double>& time_vector,
    const Eigen::MatrixXd& data_matrix,
    const std::vector<std::string>& data_column_base_names) {
    
    std::ofstream file(filepath);
     if (!file.is_open()) {
        Logger::getInstance().error("PostCalibrationAnalyser", "Failed to open file for writing: " + filepath);
        return;
    }
    file << "time";
    for(const auto& base_name : data_column_base_names) {
        for(int age=0; age < num_age_classes_; ++age) {
            file << "," << base_name << "_age" << age;
        }
    }
    file << "\n";

    for(int t=0; t < data_matrix.rows(); ++t) {
        file << time_vector[t];
        for(int age=0; age < data_matrix.cols(); ++age) { 
            file << "," << std::fixed << std::setprecision(6) << data_matrix(t, age);
        }
        file << "\n";
    }
    file.close();
}

void PostCalibrationAnalyser::saveScalarMetricsCSV(
    const std::string& filepath,
    const PostCalibrationMetrics& metrics) {
    std::ofstream file(filepath);
     if (!file.is_open()) {
        Logger::getInstance().error("PostCalibrationAnalyser", "Failed to open file for writing: " + filepath);
        return;
    }
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

    for(const auto& pair : metrics.age_specific_IFR) file << "IFR_" << pair.first << "," << pair.second << "\n";
    for(const auto& pair : metrics.age_specific_IHR) file << "IHR_" << pair.first << "," << pair.second << "\n";
    for(const auto& pair : metrics.age_specific_IICUR) file << "IICUR_" << pair.first << "," << pair.second << "\n";
    for(const auto& pair : metrics.age_specific_attack_rate) file << "AttackRate_" << pair.first << "," << pair.second << "\n";
    for(const auto& pair : metrics.kappa_values) file << pair.first << "," << pair.second << "\n";
    file.close();
}

void PostCalibrationAnalyser::savePosteriorPredictiveCheckData(
    const PosteriorPredictiveData& ppd_data,
    const std::string& sub_directory) {
    
    ensureOutputSubdirectoryExists(sub_directory);
    auto save_ppc_incidence = [&](const PosteriorPredictiveData::IncidenceData& data, const std::string& type_name) {
        std::string fp_base = FileUtils::joinPaths(FileUtils::joinPaths(output_dir_base_, sub_directory), type_name);
        saveMatrixTimeSeriesCSV(fp_base + "_median.csv", ppd_data.time_points, data.median, {"median"});
        saveMatrixTimeSeriesCSV(fp_base + "_lower90.csv", ppd_data.time_points, data.lower_90, {"lower90"});
        saveMatrixTimeSeriesCSV(fp_base + "_upper90.csv", ppd_data.time_points, data.upper_90, {"upper90"});
        saveMatrixTimeSeriesCSV(fp_base + "_lower95.csv", ppd_data.time_points, data.lower_95, {"lower95"});
        saveMatrixTimeSeriesCSV(fp_base + "_upper95.csv", ppd_data.time_points, data.upper_95, {"upper95"});
        saveMatrixTimeSeriesCSV(fp_base + "_observed.csv", ppd_data.time_points, data.observed, {"observed"});
    };

    save_ppc_incidence(ppd_data.daily_hospitalizations, "daily_hospitalizations");
    save_ppc_incidence(ppd_data.daily_icu_admissions, "daily_icu_admissions");
    save_ppc_incidence(ppd_data.daily_deaths, "daily_deaths");
    save_ppc_incidence(ppd_data.cumulative_hospitalizations, "cumulative_hospitalizations");
    save_ppc_incidence(ppd_data.cumulative_icu_admissions, "cumulative_icu_admissions");
    save_ppc_incidence(ppd_data.cumulative_deaths, "cumulative_deaths");
    Logger::getInstance().info("PostCalibrationAnalyser", "Posterior predictive check data saved to subdirectory: " + sub_directory);
}

void PostCalibrationAnalyser::saveAggregatedMCMCMetrics(
    const std::vector<PostCalibrationMetrics>& all_metrics,
    const std::string& sub_directory) {

    if (all_metrics.empty()) return;
    ensureOutputSubdirectoryExists(sub_directory);
    
    // --- Scalar Metrics Summary ---
    std::string scalar_summary_path = FileUtils::joinPaths(FileUtils::joinPaths(output_dir_base_, sub_directory), "scalar_metrics_summary.csv");
    std::ofstream ss_file(scalar_summary_path);
    ss_file << "metric,mean,median,std_dev,q025,q975\n";
    ss_file << std::fixed << std::setprecision(8);

    auto write_scalar_summary = [&](const std::string& name, const std::vector<double>& values) {
        if (values.empty()) return;
        std::vector<double> sorted_v = values; std::sort(sorted_v.begin(), sorted_v.end());
        double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        double median = getQuantiles(sorted_v, {0.5})[0];
        double q025 = getQuantiles(sorted_v, {0.025})[0];
        double q975 = getQuantiles(sorted_v, {0.975})[0];
        double sum_sq_diff = 0.0; for(double val : values) sum_sq_diff += (val - mean) * (val - mean);
        double std_dev = std::sqrt(sum_sq_diff / values.size());
        ss_file << name << "," << mean << "," << median << "," << std_dev << "," << q025 << "," << q975 << "\n";
    };
    
    std::vector<double> R0s, oIFRs, oARs, peakHs, peakICUs, timePHs, timePICUs, totDeaths;
    for(const auto& m : all_metrics) { 
        R0s.push_back(m.R0);
        oIFRs.push_back(m.overall_IFR);
        oARs.push_back(m.overall_attack_rate);
        peakHs.push_back(m.peak_hospital_occupancy);
        peakICUs.push_back(m.peak_ICU_occupancy);
        timePHs.push_back(m.time_to_peak_hospital);
        timePICUs.push_back(m.time_to_peak_ICU);
        totDeaths.push_back(m.total_cumulative_deaths);
    }
    
    write_scalar_summary("R0", R0s); 
    write_scalar_summary("overall_IFR", oIFRs);
    write_scalar_summary("overall_attack_rate", oARs);
    write_scalar_summary("peak_hospital_occupancy", peakHs);
    write_scalar_summary("peak_ICU_occupancy", peakICUs);
    write_scalar_summary("time_to_peak_hospital", timePHs);
    write_scalar_summary("time_to_peak_ICU", timePICUs);
    write_scalar_summary("TotalCumulativeDeaths", totDeaths);

    if (!all_metrics.empty() && !all_metrics[0].age_specific_IFR.empty()) {
        for (const auto& age_pair : all_metrics[0].age_specific_IFR) {
            std::vector<double> age_ifr_vals, age_ihr_vals, age_iicur_vals, age_ar_vals;
            for (const auto& m : all_metrics) {
                if (m.age_specific_IFR.count(age_pair.first)) {
                    age_ifr_vals.push_back(m.age_specific_IFR.at(age_pair.first));
                }
                if (m.age_specific_IHR.count(age_pair.first)) {
                    age_ihr_vals.push_back(m.age_specific_IHR.at(age_pair.first));
                }
                if (m.age_specific_IICUR.count(age_pair.first)) {
                    age_iicur_vals.push_back(m.age_specific_IICUR.at(age_pair.first));
                }
                if (m.age_specific_attack_rate.count(age_pair.first)) {
                    age_ar_vals.push_back(m.age_specific_attack_rate.at(age_pair.first));
                }
            }
            write_scalar_summary("IFR_" + age_pair.first, age_ifr_vals);
            write_scalar_summary("IHR_" + age_pair.first, age_ihr_vals);
            write_scalar_summary("IICUR_" + age_pair.first, age_iicur_vals);
            write_scalar_summary("AttackRate_" + age_pair.first, age_ar_vals);
        }
    }

    // Kappa values
    if (!all_metrics.empty() && !all_metrics[0].kappa_values.empty()) {
        for (const auto& kappa_pair : all_metrics[0].kappa_values) {
            std::vector<double> k_vals;
            for (const auto& m : all_metrics) {
                if (m.kappa_values.count(kappa_pair.first)) {
                    k_vals.push_back(m.kappa_values.at(kappa_pair.first));
                }
            }
            write_scalar_summary(kappa_pair.first, k_vals);
        }
    }
    ss_file.close();

    // --- Time Series (Rt, Seroprevalence) ---
    auto save_timeseries_summary = [&](const std::string& name, 
                                       const std::function<const std::vector<double>*(const PostCalibrationMetrics&)>& getter,
                                       const std::function<const std::vector<double>*(const PostCalibrationMetrics&)>& time_getter) {
        std::vector<std::vector<double>> all_ts_values; // Each inner vector is one MCMC sample's trajectory
        std::vector<double> time_points;
        
        for(const auto& m : all_metrics) {
            const std::vector<double>* ts_ptr = getter(m);
            if(ts_ptr && !ts_ptr->empty()) {
                all_ts_values.push_back(*ts_ptr);
                if (time_points.empty()) {
                    const std::vector<double>* time_ptr = time_getter(m);
                    if (time_ptr) time_points = *time_ptr;
                }
            }
        }
        if (all_ts_values.empty() || all_ts_values[0].empty()) return;

        int T_len = all_ts_values[0].size();
        std::vector<double> med_ts(T_len), q025_ts(T_len), q975_ts(T_len);
        for(int t=0; t < T_len; ++t) {
            std::vector<double> values_at_t;
            for(const auto& ts : all_ts_values) if (t < (int)ts.size()) values_at_t.push_back(ts[t]);
            std::sort(values_at_t.begin(), values_at_t.end());
            if (!values_at_t.empty()) {
                med_ts[t] = getQuantiles(values_at_t, {0.5})[0];
                q025_ts[t] = getQuantiles(values_at_t, {0.025})[0];
                q975_ts[t] = getQuantiles(values_at_t, {0.975})[0];
            }
        }
        std::string ts_path = FileUtils::joinPaths(FileUtils::joinPaths(output_dir_base_, sub_directory), name + "_trajectory.csv");
        saveTimeSeriesCSV(ts_path, {"time", "median", "q025", "q975"}, {time_points, med_ts, q025_ts, q975_ts});
    };
    
    save_timeseries_summary("Rt", 
                           [](const PostCalibrationMetrics& m){ return &m.Rt_median; },
                           [](const PostCalibrationMetrics& m){ return &m.Rt_time; });
    save_timeseries_summary("Seroprevalence", 
                           [](const PostCalibrationMetrics& m){ return &m.seroprevalence_median; },
                           [](const PostCalibrationMetrics& m){ return &m.seroprevalence_time; });
    
    // --- Hidden Dynamics & Prevalence (median and CIs) ---
    std::vector<std::string> hidden_comp_names = {"E", "P", "A", "I", "R"};
    for (const auto& comp_name : hidden_comp_names) {
        std::vector<Eigen::MatrixXd> all_comp_trajs;
        for(const auto& m : all_metrics) {
            if(m.hidden_compartments_median.count(comp_name)) {
                all_comp_trajs.push_back(m.hidden_compartments_median.at(comp_name));
            }
        }
        if (all_comp_trajs.empty()) continue;
        Eigen::MatrixXd med_traj = getQuantileMatrix(all_comp_trajs, 0.5);
        Eigen::MatrixXd q025_traj = getQuantileMatrix(all_comp_trajs, 0.025);
        Eigen::MatrixXd q975_traj = getQuantileMatrix(all_comp_trajs, 0.975);

        std::string comp_path_base = FileUtils::joinPaths(FileUtils::joinPaths(output_dir_base_, sub_directory), comp_name + "_hidden_dynamics");
        std::vector<double> time_points = all_metrics[0].Rt_time;
        saveMatrixTimeSeriesCSV(comp_path_base + "_median.csv", time_points, med_traj, {comp_name});
        saveMatrixTimeSeriesCSV(comp_path_base + "_q025.csv", time_points, q025_traj, {comp_name});
        saveMatrixTimeSeriesCSV(comp_path_base + "_q975.csv", time_points, q975_traj, {comp_name});
    }
    
    // --- Prevalence Trajectories ---
    std::vector<std::string> prevalence_types = {"age_specific", "overall"};
    for (const auto& prev_type : prevalence_types) {
        std::vector<Eigen::MatrixXd> all_prev_trajs;
        for(const auto& m : all_metrics) {
            if(m.prevalence_trajectories_median.count(prev_type)) {
                all_prev_trajs.push_back(m.prevalence_trajectories_median.at(prev_type));
            }
        }
        if (all_prev_trajs.empty()) continue;
        
        Eigen::MatrixXd med_prev_traj = getQuantileMatrix(all_prev_trajs, 0.5);
        Eigen::MatrixXd q025_prev_traj = getQuantileMatrix(all_prev_trajs, 0.025);
        Eigen::MatrixXd q975_prev_traj = getQuantileMatrix(all_prev_trajs, 0.975);

        std::string prev_path_base = FileUtils::joinPaths(FileUtils::joinPaths(output_dir_base_, sub_directory), "prevalence_" + prev_type);
        std::vector<double> time_points = all_metrics[0].Rt_time;
        saveMatrixTimeSeriesCSV(prev_path_base + "_median.csv", time_points, med_prev_traj, {"prevalence"});
        saveMatrixTimeSeriesCSV(prev_path_base + "_q025.csv", time_points, q025_prev_traj, {"prevalence"});
        saveMatrixTimeSeriesCSV(prev_path_base + "_q975.csv", time_points, q975_prev_traj, {"prevalence"});
    }

    Logger::getInstance().info("PostCalibrationAnalyser", "Aggregated MCMC metrics saved to subdirectory: " + sub_directory);
}

void PostCalibrationAnalyser::saveScenarioComparisonCSV(
    const PostCalibrationMetrics& baseline_metrics,
    const std::vector<std::pair<std::string, PostCalibrationMetrics>>& scenario_results,
    const std::string& sub_directory) {

    ensureOutputSubdirectoryExists(sub_directory);
    std::string filepath = FileUtils::joinPaths(FileUtils::joinPaths(output_dir_base_, sub_directory), "scenario_comparison_summary.csv");
    std::ofstream file(filepath);
    file << "scenario_name,R0,overall_IFR,overall_attack_rate,peak_hospital_occupancy,peak_ICU_occupancy,time_to_peak_hospital,time_to_peak_ICU,total_cumulative_deaths";
    // Add Kappa headers
    if (!baseline_metrics.kappa_values.empty()) {
        for (const auto& k_pair : baseline_metrics.kappa_values) file << "," << k_pair.first;
    }
    file << "\n";
    file << std::fixed << std::setprecision(8);

    auto write_metrics_row = [&](const std::string& name, const PostCalibrationMetrics& m) {
        file << name << "," << m.R0 << "," << m.overall_IFR << "," << m.overall_attack_rate << ","
             << m.peak_hospital_occupancy << "," << m.peak_ICU_occupancy << ","
             << m.time_to_peak_hospital << "," << m.time_to_peak_ICU << "," << m.total_cumulative_deaths;
        if (!m.kappa_values.empty()) {
             for (const auto& k_pair_baseline : baseline_metrics.kappa_values) { // Use baseline for order
                if (m.kappa_values.count(k_pair_baseline.first)) {
                    file << "," << m.kappa_values.at(k_pair_baseline.first);
                } else {
                    file << ","; // Missing kappa
                }
            }
        }
        file << "\n";
    };
    
    write_metrics_row("baseline", baseline_metrics);
    for(const auto& scen_pair : scenario_results) {
        write_metrics_row(scen_pair.first, scen_pair.second);
    }
    file.close();
    Logger::getInstance().info("PostCalibrationAnalyser", "Scenario comparison summary saved to: " + filepath);
}


double PostCalibrationAnalyser::getTotalCumulativeDeathsFromSimulation(const SEPAIHRDParameters& params) {
    auto run_npi_strategy = model_template_->getNpiStrategy()->clone();
    auto run_model = std::make_shared<AgeSEPAIHRDModel>(params, run_npi_strategy);
    AgeSEPAIHRDSimulator simulator(run_model, solver_strategy_, time_points_.front(), time_points_.back(), 1.0, 1e-6, 1e-6);
    SimulationResult sim_result = simulator.run(initial_state_, time_points_);
    
    Eigen::Map<const Eigen::VectorXd> D_final(&sim_result.solution.back()[8 * num_age_classes_], num_age_classes_);
    Eigen::Map<const Eigen::VectorXd> D_initial(&initial_state_[8 * num_age_classes_], num_age_classes_);
    return (D_final - D_initial).sum();
}

// For scenario trajectory comparison
Eigen::VectorXd PostCalibrationAnalyser::getOverallTrajectory(const SimulationResult& sim_result, int compartment_offset_multiplier) {
    Eigen::VectorXd trajectory = Eigen::VectorXd::Zero(sim_result.time_points.size());
    for(size_t t=0; t < sim_result.time_points.size(); ++t) {
        double sum_val = 0;
        for(int age=0; age < num_age_classes_; ++age) {
            sum_val += sim_result.solution[t][compartment_offset_multiplier * num_age_classes_ + age];
        }
        trajectory(t) = sum_val;
    }
    return trajectory;
}


// Implement the missing extractMatchingTrajectories helper
std::vector<Eigen::MatrixXd> PostCalibrationAnalyser::extractMatchingTrajectories(
    const std::vector<std::map<std::string, Eigen::MatrixXd>>& all_trajectories_maps,
    const std::string& key) {
    
    std::vector<Eigen::MatrixXd> matching_trajectories;
    matching_trajectories.reserve(all_trajectories_maps.size());
    
    for (const auto& trajectory_map : all_trajectories_maps) {
        auto it = trajectory_map.find(key);
        if (it != trajectory_map.end()) {
            matching_trajectories.push_back(it->second);
        }
    }
    
    return matching_trajectories;
}

// Explicit template instantiation to fix linking issues
template std::vector<double> PostCalibrationAnalyser::getQuantiles<double>(
    const std::vector<double>& sorted_values, 
    const std::vector<double>& quantiles_probs);

} // namespace epidemic