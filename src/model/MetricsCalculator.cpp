#include "model/MetricsCalculator.hpp"
#include "model/ReproductionNumberCalculator.hpp"
#include "utils/Logger.hpp"
#include <cmath>

namespace epidemic {

EssentialMetrics MetricsCalculator::calculateEssentialMetrics(
    const SimulationResult& sim_result,
    std::shared_ptr<AgeSEPAIHRDModel> model,
    const SEPAIHRDParameters& params,
    const Eigen::VectorXd& initial_state,
    const std::vector<double>& time_points) const {
    
    EssentialMetrics metrics;
    int num_age_classes = params.N.size();
    
    metrics.age_specific_IFR.resize(num_age_classes);
    metrics.age_specific_IHR.resize(num_age_classes);
    metrics.age_specific_IICUR.resize(num_age_classes);
    metrics.age_specific_attack_rate.resize(num_age_classes);
    
    if (!sim_result.isValid()) {
        Logger::getInstance().warning("MetricsCalculator", "Invalid simulation result");
        return metrics;
    }
    
    // Calculate R0
    ReproductionNumberCalculator rn_calc(model);
    metrics.R0 = rn_calc.calculateR0();
    
    // Initial infections include all individuals who have been infected (everyone except S)
    Eigen::Map<const Eigen::VectorXd> E0(&initial_state[1 * num_age_classes], num_age_classes);
    Eigen::Map<const Eigen::VectorXd> P0(&initial_state[2 * num_age_classes], num_age_classes);
    Eigen::Map<const Eigen::VectorXd> A0(&initial_state[3 * num_age_classes], num_age_classes);
    Eigen::Map<const Eigen::VectorXd> I0(&initial_state[4 * num_age_classes], num_age_classes);
    Eigen::Map<const Eigen::VectorXd> H0(&initial_state[5 * num_age_classes], num_age_classes);
    Eigen::Map<const Eigen::VectorXd> ICU0(&initial_state[6 * num_age_classes], num_age_classes);
    Eigen::Map<const Eigen::VectorXd> R0_vec(&initial_state[7 * num_age_classes], num_age_classes);
    
    Eigen::VectorXd initial_infections = E0 + P0 + A0 + I0 + H0 + ICU0 + R0_vec;
    
    Eigen::VectorXd cumulative_infections = initial_infections;
    Eigen::VectorXd cumulative_hosp = Eigen::VectorXd::Zero(num_age_classes);
    Eigen::VectorXd cumulative_icu = Eigen::VectorXd::Zero(num_age_classes);
    double total_population = params.N.sum();
    
    // Variables for tracking peaks
    metrics.peak_hospital_occupancy = 0.0;
    metrics.peak_ICU_occupancy = 0.0;
    
    // Target day for seroprevalence (day 64 = May 4th)
    const double target_day = 64.0;
    long target_idx = -1;
    double min_diff = std::numeric_limits<double>::max();
    
    for (size_t i = 0; i < time_points.size(); ++i) {
        double diff = std::abs(time_points[i] - target_day);
        if (diff < min_diff) {
            min_diff = diff;
            target_idx = i;
        }
    }
    
    if (min_diff > 1.0) {
        Logger::getInstance().warning("MetricsCalculator", 
            "Target day 64 not found exactly. Closest is " + std::to_string(time_points[target_idx]));
    }
    
    // Process simulation timestep by timestep
    for (size_t t = 0; t < time_points.size(); ++t) {
        double time_t = time_points[t];
        double dt = (t > 0) ? (time_t - time_points[t-1]) : 1.0;
        
        // Extract current state
        Eigen::Map<const Eigen::VectorXd> S_t(&sim_result.solution[t][0 * num_age_classes], num_age_classes);
        Eigen::Map<const Eigen::VectorXd> P_t(&sim_result.solution[t][2 * num_age_classes], num_age_classes);
        Eigen::Map<const Eigen::VectorXd> A_t(&sim_result.solution[t][3 * num_age_classes], num_age_classes);
        Eigen::Map<const Eigen::VectorXd> I_t(&sim_result.solution[t][4 * num_age_classes], num_age_classes);
        Eigen::Map<const Eigen::VectorXd> H_t(&sim_result.solution[t][5 * num_age_classes], num_age_classes);
        Eigen::Map<const Eigen::VectorXd> ICU_t(&sim_result.solution[t][6 * num_age_classes], num_age_classes);
        
        // Calculate Rt
        double Rt = rn_calc.calculateRt(S_t, time_t);
        metrics.max_Rt = std::max(metrics.max_Rt, Rt);
        metrics.min_Rt = std::min(metrics.min_Rt, Rt);
        if (t == time_points.size() - 1) {
            metrics.final_Rt = Rt;
        }
        
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
        double kappa_t = model->getNpiStrategy()->getReductionFactor(time_t);
        Eigen::VectorXd infectious_load = Eigen::VectorXd::Zero(num_age_classes);
        for (int j = 0; j < num_age_classes; ++j) {
            if (params.N(j) > 1e-9) {
                infectious_load(j) = (P_t(j) + A_t(j) + params.theta * I_t(j)) / params.N(j);
            }
        }
        Eigen::VectorXd lambda_t = params.beta * kappa_t * params.M_baseline * infectious_load;
        Eigen::VectorXd new_infections = lambda_t.array() * S_t.array() * dt;
        
        cumulative_infections += new_infections;
        
        // Seroprevalence at target day
        if (static_cast<long>(t) == target_idx) {
            double total_susceptible = S_t.sum();
            metrics.seroprevalence_at_target_day = (total_population - total_susceptible) / total_population;
        }
    }
    
    // Final metrics
    Eigen::Map<const Eigen::VectorXd> D_final(&sim_result.solution.back()[8 * num_age_classes], num_age_classes);
    Eigen::Map<const Eigen::VectorXd> D_initial(&initial_state[8 * num_age_classes], num_age_classes);
    Eigen::VectorXd cumulative_deaths = D_final - D_initial;

    // Calculate cumulative hospitalizations and ICU admissions from states
    Eigen::Map<const Eigen::VectorXd> CumH_final(&sim_result.solution.back()[9 * num_age_classes], num_age_classes);
    Eigen::Map<const Eigen::VectorXd> CumICU_final(&sim_result.solution.back()[10 * num_age_classes], num_age_classes);
    
    Eigen::Map<const Eigen::VectorXd> CumH_initial(&initial_state[9 * num_age_classes], num_age_classes);
    Eigen::Map<const Eigen::VectorXd> CumICU_initial(&initial_state[10 * num_age_classes], num_age_classes);
    
    cumulative_hosp = CumH_final - CumH_initial;
    cumulative_icu = CumICU_final - CumICU_initial;
    
    metrics.total_cumulative_deaths = cumulative_deaths.sum();
    metrics.overall_attack_rate = cumulative_infections.sum() / total_population;
    metrics.overall_IFR = (cumulative_infections.sum() > 1e-9) ? 
        cumulative_deaths.sum() / cumulative_infections.sum() : 0.0;
    
    // Age-specific metrics with bounds checking to prevent extreme ratio values
    const double MIN_INFECTIONS_FOR_RATIO = 1.0;
    const double MAX_RATIO = 1.0;
    
    for (int age = 0; age < num_age_classes; ++age) {
        metrics.age_specific_attack_rate[age] = (params.N(age) > 0) ? 
            cumulative_infections(age) / params.N(age) : 0.0;
        
        if (cumulative_infections(age) > MIN_INFECTIONS_FOR_RATIO) {
            double ifr = cumulative_deaths(age) / cumulative_infections(age);
            double ihr = cumulative_hosp(age) / cumulative_infections(age);
            double iicur = cumulative_icu(age) / cumulative_infections(age);
            
            metrics.age_specific_IFR[age] = std::max(0.0, std::min(ifr, MAX_RATIO));
            metrics.age_specific_IHR[age] = std::max(0.0, std::min(ihr, MAX_RATIO));
            metrics.age_specific_IICUR[age] = std::max(0.0, std::min(iicur, MAX_RATIO));
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

std::vector<double> MetricsCalculator::calculateRtTrajectory(
    const SimulationResult& sim_result,
    std::shared_ptr<AgeSEPAIHRDModel> model,
    const std::vector<double>& time_points) const {
    
    std::vector<double> rt_trajectory;
    rt_trajectory.reserve(time_points.size());
    
    if (!sim_result.isValid()) {
        Logger::getInstance().warning("MetricsCalculator", "Invalid simulation for Rt trajectory");
        return rt_trajectory;
    }
    
    ReproductionNumberCalculator rn_calc(model);
    int num_age_classes = model->getNumAgeClasses();
    
    for (size_t t = 0; t < time_points.size(); ++t) {
        Eigen::Map<const Eigen::VectorXd> S_t(&sim_result.solution[t][0], num_age_classes);
        double Rt = rn_calc.calculateRt(S_t, time_points[t]);
        rt_trajectory.push_back(Rt);
    }
    
    return rt_trajectory;
}

std::vector<double> MetricsCalculator::calculateSeroprevalenceTrajectory(
    const SimulationResult& sim_result,
    const SEPAIHRDParameters& params,
    const std::vector<double>& time_points,
    const Eigen::VectorXd& initial_state [[maybe_unused]]) const {
    
    std::vector<double> sero_trajectory;
    sero_trajectory.reserve(time_points.size());
    
    if (!sim_result.isValid()) {
        Logger::getInstance().warning("MetricsCalculator", "Invalid simulation for seroprevalence trajectory");
        return sero_trajectory;
    }
    
    int num_age_classes = params.N.size();
    double total_population = params.N.sum();
    
    for (size_t t = 0; t < time_points.size(); ++t) {
        Eigen::Map<const Eigen::VectorXd> S_t(&sim_result.solution[t][0], num_age_classes);
        
        double total_susceptible = S_t.sum();
        double seroprevalence = (total_population - total_susceptible) / total_population;
        
        sero_trajectory.push_back(seroprevalence);
    }
    
    return sero_trajectory;
}

} // namespace epidemic
