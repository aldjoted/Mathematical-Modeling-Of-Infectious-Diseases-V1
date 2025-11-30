#ifndef ANALYSIS_TYPES_HPP
#define ANALYSIS_TYPES_HPP

#include <vector>
#include <map>
#include <string>
#include <Eigen/Dense>

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
 * @brief Aggregated statistics for a metric across samples
 */
using AggregatedStats = std::map<std::string, double>; // e.g., "mean", "median", "q025", "q975"

} // namespace epidemic

#endif // ANALYSIS_TYPES_HPP
