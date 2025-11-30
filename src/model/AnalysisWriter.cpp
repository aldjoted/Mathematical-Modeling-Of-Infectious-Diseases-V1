#include "model/AnalysisWriter.hpp"
#include "model/ModelConstants.hpp"
#include "utils/FileUtils.hpp"
#include "utils/Logger.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <numeric>

namespace epidemic {

AnalysisWriter::AnalysisWriter() {
    // FIX Bug 5: Add exception handling to ensure thread safety
    // Use try-catch to handle exceptions during thread creation
    try {
        // Start worker thread
        worker_thread_ = std::thread(&AnalysisWriter::workerLoop, this);
        Logger::getInstance().info("AnalysisWriter", "Async writer initialized with worker thread");
    } catch (const std::exception& e) {
        Logger::getInstance().error("AnalysisWriter", 
            std::string("Failed to start worker thread: ") + e.what());
        throw; // Re-throw to notify caller
    }
}

AnalysisWriter::~AnalysisWriter() {
    // Signal worker to stop
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_flag_ = true;
    }
    queue_cv_.notify_one();
    
    // Wait for worker to finish
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
    
    Logger::getInstance().info("AnalysisWriter", "Async writer shut down");
}

void AnalysisWriter::workerLoop() {
    while (true) {
        std::function<void()> task;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            // Wait for task or stop signal
            queue_cv_.wait(lock, [this] {
                return stop_flag_ || !task_queue_.empty();
            });
            
            // Exit if stop requested and queue is empty
            if (stop_flag_ && task_queue_.empty()) {
                break;
            }
            
            // Get next task
            if (!task_queue_.empty()) {
                task = std::move(task_queue_.front());
                task_queue_.pop();
            }
        }
        
        // Execute task outside lock
        if (task) {
            try {
                task();
            } catch (const std::exception& e) {
                Logger::getInstance().error("AnalysisWriter", 
                    std::string("Error in async task: ") + e.what());
            }
            
            // FIX Bug 10: Notify waiters that a task completed (queue may now be empty)
            queue_cv_.notify_all();
        }
    }
}

void AnalysisWriter::enqueueTask(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.push(std::move(task));
    }
    queue_cv_.notify_one();
}

void AnalysisWriter::waitForCompletion() {
    // FIX Bug 10: Use condition variable instead of busy-wait loop
    std::unique_lock<std::mutex> lock(queue_mutex_);
    
    // Wait until queue is empty
    // The condition variable is notified whenever a task completes
    queue_cv_.wait(lock, [this] {
        return task_queue_.empty();
    });
    
    Logger::getInstance().info("AnalysisWriter", "All async I/O tasks completed");
}

size_t AnalysisWriter::getPendingTaskCount() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return task_queue_.size();
}

// Public async methods (enqueue tasks)

void AnalysisWriter::saveVectorToCSV(
    const std::string& filepath, 
    const std::vector<double>& data) {
    
    // Deep copy data for thread safety
    enqueueTask([this, filepath, data]() {
        writeVectorToCSV(filepath, data);
    });
}

void AnalysisWriter::saveParameterPosteriors(
    const std::string& output_dir,
    const std::vector<Eigen::VectorXd>& param_samples,
    const std::vector<std::string>& param_names,
    int burn_in,
    int thinning) {
    
    // Deep copy for thread safety
    enqueueTask([this, output_dir, param_samples, param_names, burn_in, thinning]() {
        writeParameterPosteriors(output_dir, param_samples, param_names, burn_in, thinning);
    });
}

void AnalysisWriter::savePosteriorPredictiveData(
    const std::string& output_dir,
    const PosteriorPredictiveData& ppd_data) {
    
    // Deep copy for thread safety
    enqueueTask([this, output_dir, ppd_data]() {
        writePosteriorPredictiveData(output_dir, ppd_data);
    });
}

void AnalysisWriter::saveBatchMetrics(
    const std::string& filepath,
    const std::vector<EssentialMetrics>& batch_metrics,
    int num_age_classes) {
    
    enqueueTask([this, filepath, batch_metrics, num_age_classes]() {
        writeBatchMetrics(filepath, batch_metrics, num_age_classes);
    });
}

void AnalysisWriter::saveAggregatedSummary(
    const std::string& filepath,
    const std::map<std::string, AggregatedStats>& summary) {
    
    enqueueTask([this, filepath, summary]() {
        writeAggregatedSummary(filepath, summary);
    });
}

void AnalysisWriter::saveScenarioComparison(
    const std::string& filepath,
    const std::vector<std::pair<std::string, EssentialMetrics>>& scenarios) {
    
    enqueueTask([this, filepath, scenarios]() {
        writeScenarioComparison(filepath, scenarios);
    });
}

void AnalysisWriter::saveEneCovidValidation(
    const std::string& filepath,
    const std::map<std::string, double>& ene_covid_data) {
    
    enqueueTask([this, filepath, ene_covid_data]() {
        writeEneCovidValidation(filepath, ene_covid_data);
    });
}

void AnalysisWriter::saveAggregatedTrajectory(
    const std::string& filepath,
    const std::vector<double>& time_points,
    const std::map<double, AggregatedStats>& aggregated_data) {
    
    enqueueTask([this, filepath, time_points, aggregated_data]() {
        writeAggregatedTrajectory(filepath, time_points, aggregated_data);
    });
}

// Private write methods (executed by worker thread)

void AnalysisWriter::writeVectorToCSV(
    const std::string& filepath, 
    const std::vector<double>& data) {
    
    std::ofstream file(filepath);
    if (!file.is_open()) {
        Logger::getInstance().error("AnalysisWriter", "Failed to open file: " + filepath);
        return;
    }
    
    file << std::setprecision(10);
    for (const auto& value : data) {
        file << value << "\n";
    }
    file.close();
}

void AnalysisWriter::writeParameterPosteriors(
    const std::string& output_dir,
    const std::vector<Eigen::VectorXd>& param_samples,
    const std::vector<std::string>& param_names,
    int burn_in,
    int thinning) {
    
    // Save posterior samples
    std::string samples_filepath = FileUtils::joinPaths(output_dir, "posterior_samples.csv");
    std::ofstream sfile(samples_filepath);
    
    if (!sfile.is_open()) {
        Logger::getInstance().error("AnalysisWriter", "Failed to open: " + samples_filepath);
        return;
    }
    
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
    std::string summary_filepath = FileUtils::joinPaths(output_dir, "posterior_summary.csv");
    std::ofstream sumfile(summary_filepath);
    
    if (!sumfile.is_open()) {
        Logger::getInstance().error("AnalysisWriter", "Failed to open: " + summary_filepath);
        return;
    }
    
    // UPDATED HEADER FOR CLARITY
    sumfile << "parameter,mean,median,std_dev,lower_95_ci,upper_95_ci\n";
    sumfile << std::fixed << std::setprecision(8);
    
    // Process one parameter at a time
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
            
            // 95% CI Calculation
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
    }
    
    sumfile.close();
    Logger::getInstance().info("AnalysisWriter", "Parameter posteriors saved to: " + output_dir);
}

void AnalysisWriter::writePosteriorPredictiveData(
    const std::string& output_dir,
    const PosteriorPredictiveData& ppd_data) {
    
    // FIX Bug 6: Get actual age class count from the data instead of hardcoded constant
    int num_age_classes = 0;
    if (ppd_data.daily_hospitalizations.median.cols() > 0) {
        num_age_classes = ppd_data.daily_hospitalizations.median.cols();
    } else if (ppd_data.daily_deaths.median.cols() > 0) {
        num_age_classes = ppd_data.daily_deaths.median.cols();
    } else {
        // Fallback to default only if no data available
        num_age_classes = epidemic::constants::DEFAULT_NUM_AGE_CLASSES;
        Logger::getInstance().warning("AnalysisWriter", 
            "Could not determine age classes from PPC data, using default: " + 
            std::to_string(num_age_classes));
    }
    
    auto saveIncidenceData = [&](const PosteriorPredictiveData::IncidenceData& data,
                                const std::string& base_name) {
        auto saveMatrix = [&](const Eigen::MatrixXd& matrix, const std::string& suffix) {
            std::string filepath = FileUtils::joinPaths(output_dir, base_name + "_" + suffix + ".csv");
            
            std::ofstream file(filepath);
            if (!file.is_open()) {
                Logger::getInstance().error("AnalysisWriter", "Failed to open: " + filepath);
                return;
            }
            
            // Use dynamic age class count
            int actual_cols = std::min(num_age_classes, static_cast<int>(matrix.cols()));
            
            file << "time";
            for (int age = 0; age < actual_cols; ++age) {
                file << ",age_" << age;
            }
            file << "\n";
            
            for (size_t t = 0; t < ppd_data.time_points.size(); ++t) {
                file << ppd_data.time_points[t];
                for (int age = 0; age < actual_cols; ++age) {
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
    
    Logger::getInstance().info("AnalysisWriter", 
        "Posterior predictive check data saved to: " + output_dir);
}

void AnalysisWriter::writeBatchMetrics(
    const std::string& filepath,
    const std::vector<EssentialMetrics>& batch_metrics,
    int num_age_classes) {
    
    std::ofstream file(filepath);
    if (!file.is_open()) {
        Logger::getInstance().error("AnalysisWriter", "Failed to open: " + filepath);
        return;
    }
    
    // Write header
    file << "sample_idx,R0,overall_IFR,overall_attack_rate,peak_hospital,peak_ICU,"
         << "time_to_peak_hospital,time_to_peak_ICU,total_deaths,"
         << "max_Rt,min_Rt,final_Rt,seroprevalence_day64";
    
    for (int age = 0; age < num_age_classes; ++age) {
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
        
        for (int age = 0; age < num_age_classes; ++age) {
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

void AnalysisWriter::writeAggregatedSummary(
    const std::string& filepath,
    const std::map<std::string, AggregatedStats>& summary) {
    
    std::ofstream file(filepath);
    if (!file.is_open()) {
        Logger::getInstance().error("AnalysisWriter", "Failed to open: " + filepath);
        return;
    }
    
    file << "metric,mean,median,std_dev,q025,q975\n";
    file << std::fixed << std::setprecision(8);
    
    for (const auto& [metric_name, stats] : summary) {
        file << metric_name;
        
        // Expected keys: mean, median, std_dev, q025, q975
        if (stats.count("mean")) file << "," << stats.at("mean");
        else file << ",";
        
        if (stats.count("median")) file << "," << stats.at("median");
        else file << ",";
        
        if (stats.count("std_dev")) file << "," << stats.at("std_dev");
        else file << ",";
        
        if (stats.count("q025")) file << "," << stats.at("q025");
        else file << ",";
        
        if (stats.count("q975")) file << "," << stats.at("q975");
        else file << ",";
        
        file << "\n";
    }
    
    file.close();
    Logger::getInstance().info("AnalysisWriter", "Aggregated summary saved to: " + filepath);
}

void AnalysisWriter::writeScenarioComparison(
    const std::string& filepath,
    const std::vector<std::pair<std::string, EssentialMetrics>>& scenarios) {
    
    std::ofstream file(filepath);
    if (!file.is_open()) {
        Logger::getInstance().error("AnalysisWriter", "Failed to open: " + filepath);
        return;
    }
    
    // Write header
    file << "scenario,R0,overall_IFR,overall_attack_rate,peak_hospital,peak_ICU,"
         << "time_to_peak_hospital,time_to_peak_ICU,total_deaths,seroprevalence_day64";
    
    // Add kappa headers (use first scenario as template)
    if (!scenarios.empty() && !scenarios[0].second.kappa_values.empty()) {
        for (const auto& kappa_pair : scenarios[0].second.kappa_values) {
            file << "," << kappa_pair.first;
        }
    }
    file << "\n";
    
    // Write data
    for (const auto& [scenario_name, metrics] : scenarios) {
        file << scenario_name << "," << metrics.R0 << "," << metrics.overall_IFR << ","
             << metrics.overall_attack_rate << "," << metrics.peak_hospital_occupancy << ","
             << metrics.peak_ICU_occupancy << "," << metrics.time_to_peak_hospital << ","
             << metrics.time_to_peak_ICU << "," << metrics.total_cumulative_deaths << ","
             << metrics.seroprevalence_at_target_day;
        
        for (const auto& kappa_pair : metrics.kappa_values) {
            file << "," << kappa_pair.second;
        }
        file << "\n";
    }
    
    file.close();
    Logger::getInstance().info("AnalysisWriter", "Scenario comparison saved to: " + filepath);
}

void AnalysisWriter::writeEneCovidValidation(
    const std::string& filepath,
    const std::map<std::string, double>& ene_covid_data) {
    
    std::ofstream file(filepath);
    if (!file.is_open()) {
        Logger::getInstance().error("AnalysisWriter", "Failed to open: " + filepath);
        return;
    }
    
    file << "source,median_seroprevalence,lower_95ci,upper_95ci,target_day\n";
    file << std::fixed << std::setprecision(5);
    
    // Write model results
    if (ene_covid_data.count("model_median")) {
        file << "Model," << ene_covid_data.at("model_median") << ","
             << ene_covid_data.at("model_q025") << ","
             << ene_covid_data.at("model_q975") << ","
             << ene_covid_data.at("target_day") << "\n";
    }
    
    // Write ENE-COVID reference
    if (ene_covid_data.count("enecovid_mean")) {
        file << "ENE_COVID," << ene_covid_data.at("enecovid_mean") << ","
             << ene_covid_data.at("enecovid_lower_ci") << ","
             << ene_covid_data.at("enecovid_upper_ci") << ","
             << ene_covid_data.at("target_day") << "\n";
    }
    
    file.close();
    Logger::getInstance().info("AnalysisWriter", "ENE-COVID validation saved to: " + filepath);
}

void AnalysisWriter::writeAggregatedTrajectory(
    const std::string& filepath,
    const std::vector<double>& time_points,
    const std::map<double, AggregatedStats>& aggregated_data) {
    
    std::ofstream file(filepath);
    if (!file.is_open()) {
        Logger::getInstance().error("AnalysisWriter", "Failed to open: " + filepath);
        return;
    }
    
    file << "time,median,q025,q975,q05,q95\n";
    file << std::fixed << std::setprecision(6);
    
    for (const auto& time : time_points) {
        if (aggregated_data.count(time)) {
            const auto& stats = aggregated_data.at(time);
            file << time << ","
                 << stats.at("median") << ","
                 << stats.at("q025") << ","
                 << stats.at("q975") << ","
                 << stats.at("q05") << ","
                 << stats.at("q95") << "\n";
        }
    }
    
    file.close();
}

} // namespace epidemic