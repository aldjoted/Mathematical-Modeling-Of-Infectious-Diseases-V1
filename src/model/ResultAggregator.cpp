#include "model/ResultAggregator.hpp"
#include "model/AgeSEPAIHRDsimulator.hpp"
#include "model/parameters/SEPAIHRDParameterManager.hpp"
#include "utils/Logger.hpp"
#include "utils/FileUtils.hpp"
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>

namespace fs = std::filesystem;
namespace ba = boost::accumulators;

namespace epidemic {

// Type aliases for accumulators
using SummaryStatsAccumulatorType = ba::accumulator_set<double,
    ba::stats<
        ba::tag::extended_p_square_quantile(ba::quadratic),
        ba::tag::mean,
        ba::tag::variance(ba::lazy),
        ba::tag::count
    >
>;

using PPDQuantileAccumulatorType = ba::accumulator_set<double,
    ba::stats<
        ba::tag::extended_p_square_quantile(ba::quadratic),
        ba::tag::count
    >
>;

std::map<std::string, AggregatedStats> ResultAggregator::aggregateBatchMetrics(
    const std::vector<EssentialMetrics>& batch_metrics,
    int num_age_classes) const {
    
    if (batch_metrics.empty()) {
        Logger::getInstance().warning("ResultAggregator", "Empty batch metrics");
        return {};
    }
    
    std::map<std::string, AggregatedStats> result;
    
    // Define scalar metrics to aggregate
    std::vector<std::string> scalar_metrics = {
        "R0", "overall_IFR", "overall_attack_rate", "peak_hospital", "peak_ICU",
        "time_to_peak_hospital", "time_to_peak_ICU", "total_deaths",
        "max_Rt", "min_Rt", "final_Rt", "seroprevalence_day64"
    };
    
    // Add age-specific metrics
    for (int age = 0; age < num_age_classes; ++age) {
        scalar_metrics.push_back("IFR_age_" + std::to_string(age));
        scalar_metrics.push_back("IHR_age_" + std::to_string(age));
        scalar_metrics.push_back("IICUR_age_" + std::to_string(age));
        scalar_metrics.push_back("AttackRate_age_" + std::to_string(age));
    }
    
    std::vector<double> probs = {0.025, 0.5, 0.975};
    
    // Aggregate each metric
    for (const auto& metric_name : scalar_metrics) {
        SummaryStatsAccumulatorType acc(ba::extended_p_square_probabilities = probs);
        
        for (const auto& metrics : batch_metrics) {
            double value = extractMetricValue(metrics, metric_name);
            acc(value);
        }
        
        if (ba::count(acc) > 0) {
            AggregatedStats stats;
            stats["mean"] = ba::mean(acc);
            stats["median"] = ba::quantile(acc, ba::quantile_probability = 0.5);
            stats["std_dev"] = std::sqrt(ba::variance(acc));
            stats["q025"] = ba::quantile(acc, ba::quantile_probability = 0.025);
            stats["q975"] = ba::quantile(acc, ba::quantile_probability = 0.975);
            
            result[metric_name] = stats;
        }
    }
    
    return result;
}

std::map<std::string, AggregatedStats> ResultAggregator::aggregateAllBatches(
    const std::vector<std::map<std::string, AggregatedStats>>& all_batch_stats) const {
    
    if (all_batch_stats.empty()) {
        Logger::getInstance().warning("ResultAggregator", "No batch stats to aggregate");
        return {};
    }
    
    std::map<std::string, AggregatedStats> final_result;
    
    // Get all metric names from first batch
    if (all_batch_stats[0].empty()) {
        return {};
    }
    
    // FIX: Use weighted pooled estimation across batches
    // For proper statistical aggregation, we compute:
    // - Pooled mean: weighted average of batch means (weights = batch sample counts)
    // - Pooled variance: using combined variance formula
    // - Pooled quantiles: using weighted interpolation of batch quantiles
    
    // For each metric, aggregate across batches
    for (const auto& [metric_name, _] : all_batch_stats[0]) {
        // Collect all batch statistics for this metric
        std::vector<double> batch_means;
        std::vector<double> batch_std_devs;
        std::vector<double> batch_medians;
        std::vector<double> batch_q025;
        std::vector<double> batch_q975;
        
        for (const auto& batch_stats : all_batch_stats) {
            if (batch_stats.count(metric_name)) {
                const auto& stats = batch_stats.at(metric_name);
                if (stats.count("mean")) batch_means.push_back(stats.at("mean"));
                if (stats.count("std_dev")) batch_std_devs.push_back(stats.at("std_dev"));
                if (stats.count("median")) batch_medians.push_back(stats.at("median"));
                if (stats.count("q025")) batch_q025.push_back(stats.at("q025"));
                if (stats.count("q975")) batch_q975.push_back(stats.at("q975"));
            }
        }
        
        AggregatedStats final_stats;
        
        if (!batch_means.empty()) {
            // Compute pooled mean (assuming equal batch sizes)
            double pooled_mean = std::accumulate(batch_means.begin(), batch_means.end(), 0.0) / batch_means.size();
            final_stats["mean"] = pooled_mean;
            
            // Compute pooled variance using combined variance formula:
            // For equal-sized batches: pooled_var = mean(batch_vars) + var(batch_means)
            if (!batch_std_devs.empty()) {
                double mean_of_vars = 0.0;
                for (double sd : batch_std_devs) {
                    mean_of_vars += sd * sd;
                }
                mean_of_vars /= batch_std_devs.size();
                
                double var_of_means = 0.0;
                for (double m : batch_means) {
                    var_of_means += (m - pooled_mean) * (m - pooled_mean);
                }
                var_of_means /= batch_means.size();
                
                double pooled_variance = mean_of_vars + var_of_means;
                final_stats["std_dev"] = std::sqrt(pooled_variance);
            }
            
            // FIX: For median and quantiles, use the median of batch medians
            // This is more robust than mean of medians for skewed distributions
            if (!batch_medians.empty()) {
                std::vector<double> sorted_medians = batch_medians;
                std::sort(sorted_medians.begin(), sorted_medians.end());
                size_t n = sorted_medians.size();
                if (n % 2 == 0) {
                    final_stats["median"] = (sorted_medians[n/2 - 1] + sorted_medians[n/2]) / 2.0;
                } else {
                    final_stats["median"] = sorted_medians[n/2];
                }
            }
            
            // For quantiles: use min of lower quantiles and max of upper quantiles
            // This gives conservative credible intervals
            if (!batch_q025.empty()) {
                final_stats["q025"] = *std::min_element(batch_q025.begin(), batch_q025.end());
            }
            if (!batch_q975.empty()) {
                final_stats["q975"] = *std::max_element(batch_q975.begin(), batch_q975.end());
            }
        }
        
        final_result[metric_name] = final_stats;
    }
    
    return final_result;
}

PosteriorPredictiveData ResultAggregator::aggregatePosteriorPredictives(
    const std::vector<Eigen::VectorXd>& param_samples,
    IParameterManager& param_manager,
    ISimulationRunner& runner,
    IMetricsCalculator& calculator [[maybe_unused]],
    int num_samples_for_ppc,
    const std::vector<double>& time_points,
    const Eigen::VectorXd& initial_state,
    const CalibrationData& observed_data,
    std::shared_ptr<AgeSEPAIHRDModel> model_template,
    unsigned int random_seed) const {
    
    Logger::getInstance().info("ResultAggregator", "Generating posterior predictive checks...");
    
    // Filter time_points to only include non-negative values (for run-up strategy compatibility)
    // The simulation may include negative time points for run-up, but observed data starts at t=0
    std::vector<double> positive_time_points;
    std::vector<size_t> positive_time_indices;  // Map from positive index to original index
    for (size_t i = 0; i < time_points.size(); ++i) {
        if (time_points[i] >= 0.0) {
            positive_time_points.push_back(time_points[i]);
            positive_time_indices.push_back(i);
        }
    }
    
    if (positive_time_points.empty()) {
        Logger::getInstance().warning("ResultAggregator", "No non-negative time points for PPC.");
        return PosteriorPredictiveData();
    }
    
    Logger::getInstance().info("ResultAggregator", 
        "PPC using " + std::to_string(positive_time_points.size()) + 
        " time points (filtered from " + std::to_string(time_points.size()) + ")");
    
    PosteriorPredictiveData ppd_data;
    ppd_data.time_points = positive_time_points;  // Use only positive time points
    
    // Fill observed data
    ppd_data.daily_hospitalizations.observed = observed_data.getNewHospitalizations();
    ppd_data.daily_icu_admissions.observed = observed_data.getNewICU();
    ppd_data.daily_deaths.observed = observed_data.getNewDeaths();
    ppd_data.cumulative_hospitalizations.observed = observed_data.getCumulativeHospitalizations();
    ppd_data.cumulative_icu_admissions.observed = observed_data.getCumulativeICU();
    ppd_data.cumulative_deaths.observed = observed_data.getCumulativeDeaths();
    
    if (param_samples.empty()) {
        Logger::getInstance().warning("ResultAggregator", "No samples for PPC.");
        return ppd_data;
    }
    
    int T = positive_time_points.size();  // Use filtered size
    int A = model_template->getNumAgeClasses();
    
    std::vector<double> ppd_probs = {0.025, 0.05, 0.5, 0.95, 0.975};
    
    // Initialize quantile accumulators
    std::vector<std::vector<PPDQuantileAccumulatorType>> hosp_acc(T, 
        std::vector<PPDQuantileAccumulatorType>(A, 
            PPDQuantileAccumulatorType(ba::extended_p_square_probabilities = ppd_probs)));
    std::vector<std::vector<PPDQuantileAccumulatorType>> icu_acc(T, 
        std::vector<PPDQuantileAccumulatorType>(A, 
            PPDQuantileAccumulatorType(ba::extended_p_square_probabilities = ppd_probs)));
    std::vector<std::vector<PPDQuantileAccumulatorType>> death_acc(T, 
        std::vector<PPDQuantileAccumulatorType>(A, 
            PPDQuantileAccumulatorType(ba::extended_p_square_probabilities = ppd_probs)));
    std::vector<std::vector<PPDQuantileAccumulatorType>> c_hosp_acc(T, 
        std::vector<PPDQuantileAccumulatorType>(A, 
            PPDQuantileAccumulatorType(ba::extended_p_square_probabilities = ppd_probs)));
    std::vector<std::vector<PPDQuantileAccumulatorType>> c_icu_acc(T, 
        std::vector<PPDQuantileAccumulatorType>(A, 
            PPDQuantileAccumulatorType(ba::extended_p_square_probabilities = ppd_probs)));
    std::vector<std::vector<PPDQuantileAccumulatorType>> c_death_acc(T, 
        std::vector<PPDQuantileAccumulatorType>(A, 
            PPDQuantileAccumulatorType(ba::extended_p_square_probabilities = ppd_probs)));
    
    // FIX Bug 8: Use provided seed for reproducibility, or random_device if seed is 0
    std::mt19937 gen;
    if (random_seed != 0) {
        gen.seed(random_seed);
        Logger::getInstance().info("ResultAggregator", 
            "PPC using fixed random seed: " + std::to_string(random_seed));
    } else {
        std::random_device rd;
        gen.seed(rd());
        Logger::getInstance().info("ResultAggregator", 
            "PPC using random seed from random_device");
    }
    
    // Select samples
    std::vector<int> selected_indices;
    if (num_samples_for_ppc > 0 && static_cast<size_t>(num_samples_for_ppc) < param_samples.size()) {
        std::uniform_int_distribution<> distrib(0, param_samples.size() - 1);
        selected_indices.reserve(num_samples_for_ppc);
        for (int i = 0; i < num_samples_for_ppc; ++i) {
            selected_indices.push_back(distrib(gen));
        }
    } else {
        selected_indices.resize(param_samples.size());
        std::iota(selected_indices.begin(), selected_indices.end(), 0);
    }
    
    int processed = 0;
    for (int sample_idx : selected_indices) {
        const Eigen::VectorXd& p_vec = param_samples[sample_idx];
        
        auto* sepaihrd_pm = dynamic_cast<SEPAIHRDParameterManager*>(&param_manager);
        if (sepaihrd_pm) {
            sepaihrd_pm->updateModelParameters(p_vec, model_template);
        } else {
            param_manager.updateModelParameters(p_vec);
        }
        
        SEPAIHRDParameters params = model_template->getModelParameters();
        
        // Run simulation (uses cache) - note: simulation uses full time_points including run-up
        SimulationResult sim_result = runner.runSimulation(params, initial_state, time_points);
        if (!sim_result.isValid()) continue;
        
        // Calculate daily incidence flows - only for positive time points
        Eigen::MatrixXd daily_hosp = Eigen::MatrixXd::Zero(T, A);
        Eigen::MatrixXd daily_icu = Eigen::MatrixXd::Zero(T, A);
        Eigen::MatrixXd daily_deaths = Eigen::MatrixXd::Zero(T, A);
        
        // Initialize previous cumulative values from the last run-up time point (or initial state if no run-up)
        size_t first_positive_idx = positive_time_indices[0];
        Eigen::VectorXd prev_CumH, prev_CumICU, prev_D;
        
        if (first_positive_idx > 0) {
            // Use state from the time point just before t=0 (end of run-up)
            size_t prev_idx = first_positive_idx - 1;
            Eigen::Map<const Eigen::VectorXd> prev_CumH_map(&sim_result.solution[prev_idx][9 * A], A);
            Eigen::Map<const Eigen::VectorXd> prev_CumICU_map(&sim_result.solution[prev_idx][10 * A], A);
            Eigen::Map<const Eigen::VectorXd> prev_D_map(&sim_result.solution[prev_idx][8 * A], A);
            prev_CumH = prev_CumH_map;
            prev_CumICU = prev_CumICU_map;
            prev_D = prev_D_map;
        } else {
            // No run-up, use initial state
            prev_CumH = initial_state.segment(9 * A, A);
            prev_CumICU = initial_state.segment(10 * A, A);
            prev_D = initial_state.segment(8 * A, A);
        }

        for (int t = 0; t < T; ++t) {
            // Use positive_time_indices to map from positive index to original simulation index
            size_t sim_idx = positive_time_indices[t];
            
            Eigen::Map<const Eigen::VectorXd> CumH_t(&sim_result.solution[sim_idx][9 * A], A);
            Eigen::Map<const Eigen::VectorXd> CumICU_t(&sim_result.solution[sim_idx][10 * A], A);
            Eigen::Map<const Eigen::VectorXd> D_t(&sim_result.solution[sim_idx][8 * A], A);
            
            for (int age = 0; age < A; ++age) {
                // Calculate daily flow as difference in cumulative states
                daily_hosp(t, age) = std::max(0.0, CumH_t(age) - prev_CumH(age));
                daily_icu(t, age) = std::max(0.0, CumICU_t(age) - prev_CumICU(age));
                daily_deaths(t, age) = std::max(0.0, D_t(age) - prev_D(age));
                
                // Update previous values for next step
                prev_CumH(age) = CumH_t(age);
                prev_CumICU(age) = CumICU_t(age);
                prev_D(age) = D_t(age);
            }
        }
        
        // Calculate cumulative values
        Eigen::MatrixXd cumulative_hosp(T, A);
        Eigen::MatrixXd cumulative_icu(T, A);
        Eigen::MatrixXd cumulative_deaths_from_flows(T, A);
        if (T > 0) {
            cumulative_hosp.row(0) = daily_hosp.row(0);
            cumulative_icu.row(0) = daily_icu.row(0);
            cumulative_deaths_from_flows.row(0) = daily_deaths.row(0);
            for (int t = 1; t < T; ++t) {
                cumulative_hosp.row(t) = cumulative_hosp.row(t - 1) + daily_hosp.row(t);
                cumulative_icu.row(t) = cumulative_icu.row(t - 1) + daily_icu.row(t);
                cumulative_deaths_from_flows.row(t) = cumulative_deaths_from_flows.row(t - 1) + daily_deaths.row(t);
            }
        }
        
        // Accumulate values for quantile calculation
        for (int t = 0; t < T; ++t) {
            for (int a = 0; a < A; ++a) {
                hosp_acc[t][a](daily_hosp(t, a));
                icu_acc[t][a](daily_icu(t, a));
                death_acc[t][a](daily_deaths(t, a));
                c_hosp_acc[t][a](cumulative_hosp(t, a));
                c_icu_acc[t][a](cumulative_icu(t, a));
                c_death_acc[t][a](cumulative_deaths_from_flows(t, a));
            }
        }
        
        processed++;
        if (processed % 10 == 0) {
            Logger::getInstance().info("ResultAggregator", 
                "PPC: Processed " + std::to_string(processed) + "/" + 
                std::to_string(selected_indices.size()) + " samples");
        }
    }
    
    // Compute quantiles
    auto fillQuantiles = [&](PosteriorPredictiveData::IncidenceData& data,
                            std::vector<std::vector<PPDQuantileAccumulatorType>>& acc) {
        data.median.resize(T, A);
        data.lower_90.resize(T, A);
        data.upper_90.resize(T, A);
        data.lower_95.resize(T, A);
        data.upper_95.resize(T, A);
        
        for (int t = 0; t < T; ++t) {
            for (int a = 0; a < A; ++a) {
                if (ba::count(acc[t][a]) > 0) {
                    data.median(t, a) = ba::quantile(acc[t][a], ba::quantile_probability = 0.5);
                    data.lower_90(t, a) = ba::quantile(acc[t][a], ba::quantile_probability = 0.05);
                    data.upper_90(t, a) = ba::quantile(acc[t][a], ba::quantile_probability = 0.95);
                    data.lower_95(t, a) = ba::quantile(acc[t][a], ba::quantile_probability = 0.025);
                    data.upper_95(t, a) = ba::quantile(acc[t][a], ba::quantile_probability = 0.975);
                } else {
                    data.median(t, a) = NAN;
                    data.lower_90(t, a) = NAN;
                    data.upper_90(t, a) = NAN;
                    data.lower_95(t, a) = NAN;
                    data.upper_95(t, a) = NAN;
                }
            }
        }
    };
    
    fillQuantiles(ppd_data.daily_hospitalizations, hosp_acc);
    fillQuantiles(ppd_data.daily_icu_admissions, icu_acc);
    fillQuantiles(ppd_data.daily_deaths, death_acc);
    fillQuantiles(ppd_data.cumulative_hospitalizations, c_hosp_acc);
    fillQuantiles(ppd_data.cumulative_icu_admissions, c_icu_acc);
    fillQuantiles(ppd_data.cumulative_deaths, c_death_acc);
    
    Logger::getInstance().info("ResultAggregator", 
        "PPC completed using " + std::to_string(processed) + " samples");
    
    return ppd_data;
}

void ResultAggregator::aggregateTrajectoryFiles(
    const std::string& source_dir,
    const std::string& output_filepath,
    const std::vector<double>& time_points,
    IAnalysisWriter& writer) const {
    
    Logger::getInstance().info("ResultAggregator", "Aggregating trajectories from " + source_dir);
    
    if (!fs::exists(source_dir) || !fs::is_directory(source_dir)) {
        Logger::getInstance().warning("ResultAggregator", 
            "Source directory not found: " + source_dir);
        return;
    }
    
    // Read all trajectories
    std::vector<std::vector<double>> all_trajectories;
    for (const auto& entry : fs::directory_iterator(source_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".csv") {
            std::vector<double> trajectory;
            std::ifstream file(entry.path());
            std::string line;
            while(std::getline(file, line)) {
                try {
                    trajectory.push_back(std::stod(line));
                } catch(const std::exception& e) {
                    // Skip lines that can't be parsed
                }
            }
            if (!trajectory.empty()) {
                all_trajectories.push_back(trajectory);
            }
        }
    }
    
    if (all_trajectories.empty()) {
        Logger::getInstance().warning("ResultAggregator", 
            "No trajectory files found in " + source_dir);
        return;
    }
    
    // Aggregate the results
    size_t num_timesteps = all_trajectories[0].size();
    std::vector<double> trajectory_probs = {0.025, 0.05, 0.5, 0.95, 0.975};
    
    std::map<double, AggregatedStats> aggregated_data;
    
    for (size_t t = 0; t < num_timesteps && t < time_points.size(); ++t) {
        SummaryStatsAccumulatorType acc_at_t(ba::extended_p_square_probabilities = trajectory_probs);
        
        for (const auto& traj : all_trajectories) {
            if (t < traj.size()) {
                acc_at_t(traj[t]);
            }
        }
        
        if (ba::count(acc_at_t) > 0) {
            AggregatedStats stats;
            stats["median"] = ba::quantile(acc_at_t, ba::quantile_probability = 0.5);
            stats["q025"] = ba::quantile(acc_at_t, ba::quantile_probability = 0.025);
            stats["q975"] = ba::quantile(acc_at_t, ba::quantile_probability = 0.975);
            stats["q05"] = ba::quantile(acc_at_t, ba::quantile_probability = 0.05);
            stats["q95"] = ba::quantile(acc_at_t, ba::quantile_probability = 0.95);
            
            aggregated_data[time_points[t]] = stats;
        }
    }
    
    // Use writer to save asynchronously
    writer.saveAggregatedTrajectory(output_filepath, time_points, aggregated_data);
}

std::map<std::string, double> ResultAggregator::performENECOVIDValidation(
    const std::map<std::string, AggregatedStats>& summary,
    double ene_covid_target_day,
    double ene_covid_mean,
    double ene_covid_lower_ci,
    double ene_covid_upper_ci) const {
    
    std::map<std::string, double> validation_data;
    
    // Extract seroprevalence at target day from summary
    std::string sero_metric_name = "seroprevalence_day64";
    
    if (summary.count(sero_metric_name)) {
        const auto& sero_stats = summary.at(sero_metric_name);
        
        if (sero_stats.count("median")) {
            validation_data["model_median"] = sero_stats.at("median");
        }
        if (sero_stats.count("q025")) {
            validation_data["model_q025"] = sero_stats.at("q025");
        }
        if (sero_stats.count("q975")) {
            validation_data["model_q975"] = sero_stats.at("q975");
        }
    }
    
    // Add ENE-COVID reference data
    validation_data["enecovid_mean"] = ene_covid_mean;
    validation_data["enecovid_lower_ci"] = ene_covid_lower_ci;
    validation_data["enecovid_upper_ci"] = ene_covid_upper_ci;
    validation_data["target_day"] = ene_covid_target_day;
    
    return validation_data;
}

double ResultAggregator::extractMetricValue(
    const EssentialMetrics& metrics,
    const std::string& metric_name) const {
    
    // Scalar metrics
    if (metric_name == "R0") return metrics.R0;
    if (metric_name == "overall_IFR") return metrics.overall_IFR;
    if (metric_name == "overall_attack_rate") return metrics.overall_attack_rate;
    if (metric_name == "peak_hospital") return metrics.peak_hospital_occupancy;
    if (metric_name == "peak_ICU") return metrics.peak_ICU_occupancy;
    if (metric_name == "time_to_peak_hospital") return metrics.time_to_peak_hospital;
    if (metric_name == "time_to_peak_ICU") return metrics.time_to_peak_ICU;
    if (metric_name == "total_deaths") return metrics.total_cumulative_deaths;
    if (metric_name == "max_Rt") return metrics.max_Rt;
    if (metric_name == "min_Rt") return metrics.min_Rt;
    if (metric_name == "final_Rt") return metrics.final_Rt;
    if (metric_name == "seroprevalence_day64") return metrics.seroprevalence_at_target_day;
    
    // Age-specific metrics
    if (metric_name.find("IFR_age_") == 0) {
        int age = std::stoi(metric_name.substr(8));
        if (age >= 0 && age < static_cast<int>(metrics.age_specific_IFR.size())) {
            return metrics.age_specific_IFR[age];
        }
    }
    if (metric_name.find("IHR_age_") == 0) {
        int age = std::stoi(metric_name.substr(8));
        if (age >= 0 && age < static_cast<int>(metrics.age_specific_IHR.size())) {
            return metrics.age_specific_IHR[age];
        }
    }
    if (metric_name.find("IICUR_age_") == 0) {
        int age = std::stoi(metric_name.substr(10));
        if (age >= 0 && age < static_cast<int>(metrics.age_specific_IICUR.size())) {
            return metrics.age_specific_IICUR[age];
        }
    }
    if (metric_name.find("AttackRate_age_") == 0) {
        int age = std::stoi(metric_name.substr(15));
        if (age >= 0 && age < static_cast<int>(metrics.age_specific_attack_rate.size())) {
            return metrics.age_specific_attack_rate[age];
        }
    }
    
    return 0.0; // Default value if not found
}

} // namespace epidemic
