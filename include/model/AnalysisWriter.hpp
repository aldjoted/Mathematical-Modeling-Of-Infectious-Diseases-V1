#ifndef ANALYSIS_WRITER_HPP
#define ANALYSIS_WRITER_HPP

#include "model/interfaces/IAnalysisWriter.hpp"
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

namespace epidemic {

/**
 * @brief Asynchronous file writer implementation
 * 
 * This class uses a worker thread to perform file I/O operations asynchronously.
 * All write methods return immediately after queueing the task.
 */
class AnalysisWriter : public IAnalysisWriter {
public:
    /**
     * @brief Construct and start the worker thread
     */
    AnalysisWriter();
    
    /**
     * @brief Destructor - stops worker thread and waits for completion
     */
    ~AnalysisWriter();
    
    // Disable copy and move
    AnalysisWriter(const AnalysisWriter&) = delete;
    AnalysisWriter& operator=(const AnalysisWriter&) = delete;
    AnalysisWriter(AnalysisWriter&&) = delete;
    AnalysisWriter& operator=(AnalysisWriter&&) = delete;
    
    void saveVectorToCSV(
        const std::string& filepath, 
        const std::vector<double>& data
    ) override;
    
    void saveParameterPosteriors(
        const std::string& output_dir,
        const std::vector<Eigen::VectorXd>& param_samples,
        const std::vector<std::string>& param_names,
        int burn_in,
        int thinning
    ) override;
    
    void savePosteriorPredictiveData(
        const std::string& output_dir,
        const PosteriorPredictiveData& ppd_data
    ) override;
    
    void saveBatchMetrics(
        const std::string& filepath,
        const std::vector<EssentialMetrics>& batch_metrics,
        int num_age_classes
    ) override;
    
    void saveAggregatedSummary(
        const std::string& filepath,
        const std::map<std::string, AggregatedStats>& summary
    ) override;
    
    void saveScenarioComparison(
        const std::string& filepath,
        const std::vector<std::pair<std::string, EssentialMetrics>>& scenarios
    ) override;
    
    void saveEneCovidValidation(
        const std::string& filepath,
        const std::map<std::string, double>& ene_covid_data
    ) override;
    
    void saveAggregatedTrajectory(
        const std::string& filepath,
        const std::vector<double>& time_points,
        const std::map<double, AggregatedStats>& aggregated_data
    ) override;
    
    void waitForCompletion() override;
    
    size_t getPendingTaskCount() const override;
    
private:
    // Worker thread
    std::thread worker_thread_;
    
    // Task queue
    std::queue<std::function<void()>> task_queue_;
    
    // Synchronization primitives
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // Control flags
    std::atomic<bool> stop_flag_{false};
    
    /**
     * @brief Worker thread main loop
     */
    void workerLoop();
    
    /**
     * @brief Enqueue a task for async execution
     * @param task Function to execute
     */
    void enqueueTask(std::function<void()> task);
    
    // Helper methods for actual file writing (executed by worker thread)
    void writeVectorToCSV(const std::string& filepath, const std::vector<double>& data);
    void writeParameterPosteriors(
        const std::string& output_dir,
        const std::vector<Eigen::VectorXd>& param_samples,
        const std::vector<std::string>& param_names,
        int burn_in,
        int thinning
    );
    void writePosteriorPredictiveData(
        const std::string& output_dir,
        const PosteriorPredictiveData& ppd_data
    );
    void writeBatchMetrics(
        const std::string& filepath,
        const std::vector<EssentialMetrics>& batch_metrics,
        int num_age_classes
    );
    void writeAggregatedSummary(
        const std::string& filepath,
        const std::map<std::string, AggregatedStats>& summary
    );
    void writeScenarioComparison(
        const std::string& filepath,
        const std::vector<std::pair<std::string, EssentialMetrics>>& scenarios
    );
    void writeEneCovidValidation(
        const std::string& filepath,
        const std::map<std::string, double>& ene_covid_data
    );
    void writeAggregatedTrajectory(
        const std::string& filepath,
        const std::vector<double>& time_points,
        const std::map<double, AggregatedStats>& aggregated_data
    );
};

} // namespace epidemic

#endif // ANALYSIS_WRITER_HPP
