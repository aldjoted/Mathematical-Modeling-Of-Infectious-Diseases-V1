#include "model/objectives/SEPAIHRDGradientObjectiveFunction.hpp"
#include "model/AgeSEPAIHRDsimulator.hpp"
#include "model/parameters/SEPAIHRDParameterManager.hpp"
#include "sir_age_structured/SimulationResultProcessor.hpp"
#include "model/ModelConstants.hpp"
#include "utils/Logger.hpp"
#include <numeric>
#include <algorithm>
#include <future>
#include <vector>
#include <cmath>

namespace epidemic {

double SEPAIHRDGradientObjectiveFunction::evaluate_with_gradient(
    const Eigen::VectorXd& params, 
    Eigen::VectorXd& grad) const {
    
    const int n_params = params.size();
    grad.resize(n_params);

    // 1. Central Value (Main Thread)
    const double f_center = SEPAIHRDObjectiveFunction::calculate(params);

    if (!std::isfinite(f_center)) {
        epidemic::Logger::getInstance().warning("SEPAIHRDGradient", 
            "Central objective value is non-finite. Gradient set to zero.");
        grad.setZero();
        return f_center;
    }

    // 2. Parallel Gradient Calculation using CLONES
    // Cast to SEPAIHRDParameterManager to access getProposalSigmas and getParamBounds
    const auto& sepaihrd_manager = dynamic_cast<const SEPAIHRDParameterManager&>(parameterManager_);
    
    #pragma omp parallel for
    for (int i = 0; i < n_params; ++i) {
        // Use relative perturbation: eps_i = epsilon_ * max(|param_i|, epsilon_)
        // This ensures perturbation is proportional to parameter magnitude
        double param_scale = std::max(std::abs(params[i]), epsilon_);
        double eps_i = epsilon_ * param_scale;
        
        // A. Clone the model for this thread
        auto thread_model = model_->clone();
        
        // B. Create a thread-local ParameterManager for the cloned model
        // Pass the actual sigmas and bounds from the original parameter manager
        SEPAIHRDParameterManager temp_manager(thread_model, 
             sepaihrd_manager.getParameterNames(), 
             sepaihrd_manager.getProposalSigmas(),
             sepaihrd_manager.getParamBounds());
             
        Eigen::VectorXd params_plus = params;
        params_plus[i] += eps_i;

        try {
            temp_manager.updateModelParameters(params_plus);
        } catch (...) {
            grad[i] = 0.0;
            continue;
        }

        // C. Prepare Initial State (Logic from SEPAIHRDObjectiveFunction::calculate)
        // IMPORTANT: Apply multipliers FIRST, then recalculate S
        Eigen::VectorXd initial_state_for_run = initial_state_;
        int n_ages = thread_model->getNumAgeClasses();

        auto get_multiplier = [&](const std::string& name) -> double {
            int idx = -1;
            const auto& names = parameterManager_.getParameterNames();
            auto it = std::find(names.begin(), names.end(), name);
            if (it != names.end()) {
                idx = std::distance(names.begin(), it);
            }
            return (idx != -1) ? params_plus(idx) : 1.0;
        };

        double e0_mult = get_multiplier("E0_multiplier");
        double p0_mult = get_multiplier("P0_multiplier");
        double a0_mult = get_multiplier("A0_multiplier");
        double i0_mult = get_multiplier("I0_multiplier");
        double h0_mult = get_multiplier("H0_multiplier");
        double icu0_mult = get_multiplier("ICU0_multiplier");
        double r0_mult = get_multiplier("R0_multiplier");
        double d0_mult = get_multiplier("D0_multiplier");

        // Apply multipliers FIRST (before recalculating S)
        initial_state_for_run.segment(1 * n_ages, n_ages) *= e0_mult;
        initial_state_for_run.segment(2 * n_ages, n_ages) *= p0_mult;
        initial_state_for_run.segment(3 * n_ages, n_ages) *= a0_mult;
        initial_state_for_run.segment(4 * n_ages, n_ages) *= i0_mult;
        initial_state_for_run.segment(5 * n_ages, n_ages) *= h0_mult;
        initial_state_for_run.segment(6 * n_ages, n_ages) *= icu0_mult;
        initial_state_for_run.segment(7 * n_ages, n_ages) *= r0_mult;
        initial_state_for_run.segment(8 * n_ages, n_ages) *= d0_mult;

        // Now recalculate S to ensure population conservation
        const Eigen::VectorXd& N = thread_model->getPopulationSizes();
        bool valid_state = true;
        for (int k = 0; k < n_ages; ++k) {
            double sum_non_S = 0.0;
            for (int j = 1; j < constants::NUM_COMPARTMENTS_SEPAIHRD; ++j) {
                sum_non_S += initial_state_for_run(j * n_ages + k);
            }
            if (sum_non_S > N(k) || sum_non_S < 0) {
                valid_state = false; break;
            }
            initial_state_for_run(k) = N(k) - sum_non_S;
        }

        if (!valid_state) {
            grad[i] = 0.0;
            continue;
        }

        // D. Run Simulation
        AgeSEPAIHRDSimulator thread_sim(
            thread_model,
            solver_strategy_, 
            time_points_.front(),
            time_points_.back(),
            1.0,
            abs_err_,
            rel_err_
        );

        SimulationResult result = thread_sim.run(initial_state_for_run, time_points_);
        
        double f_plus = -std::numeric_limits<double>::infinity();
        
        if (result.isValid()) {
             // E. Calculate Likelihood using Cumulative Variables (Corrected Logic)
             int num_compartments = AgeSEPAIHRDSimulator::NUM_COMPARTMENTS;
             
             // Retrieve Cumulative Data
             Eigen::MatrixXd CumH_data = SimulationResultProcessor::getCompartmentData(result, *thread_model, "CumH", num_compartments);
             Eigen::MatrixXd CumICU_data = SimulationResultProcessor::getCompartmentData(result, *thread_model, "CumICU", num_compartments);
             Eigen::MatrixXd D_data = SimulationResultProcessor::getCompartmentData(result, *thread_model, "D", num_compartments);

             // Calculate Daily Incidence from Cumulative Data
             Eigen::MatrixXd sim_hosp(CumH_data.rows(), CumH_data.cols());
             Eigen::MatrixXd sim_icu(CumICU_data.rows(), CumICU_data.cols());
             Eigen::MatrixXd sim_deaths(D_data.rows(), D_data.cols());

             if (time_points_.size() > 0) {
                // Row 0: Cum(t0) - Initial_Cum
                // CumH is at compartment index 9, CumICU at index 10, D at index 8
                sim_hosp.row(0) = CumH_data.row(0) - initial_state_for_run.segment(n_ages * 9, n_ages).transpose();
                sim_icu.row(0) = CumICU_data.row(0) - initial_state_for_run.segment(n_ages * 10, n_ages).transpose();
                sim_deaths.row(0) = D_data.row(0) - initial_state_for_run.segment(n_ages * 8, n_ages).transpose();

                // Subsequent rows: Cum(t) - Cum(t-1)
                if (time_points_.size() > 1) {
                    sim_hosp.bottomRows(time_points_.size() - 1) = CumH_data.bottomRows(time_points_.size() - 1) - CumH_data.topRows(time_points_.size() - 1);
                    sim_icu.bottomRows(time_points_.size() - 1) = CumICU_data.bottomRows(time_points_.size() - 1) - CumICU_data.topRows(time_points_.size() - 1);
                    sim_deaths.bottomRows(time_points_.size() - 1) = D_data.bottomRows(time_points_.size() - 1) - D_data.topRows(time_points_.size() - 1);
                }
                
                // Ensure non-negativity
                sim_hosp = sim_hosp.cwiseMax(0.0);
                sim_icu = sim_icu.cwiseMax(0.0);
                sim_deaths = sim_deaths.cwiseMax(0.0);
             } else {
                 sim_hosp.setZero();
                 sim_icu.setZero();
                 sim_deaths.setZero();
             }

             // Compare against New Daily data (from CSV)
             double ll_hosp = calculateSingleLogLikelihood(sim_hosp, observed_data_.getNewHospitalizations(), "Hospitalizations");
             double ll_icu = calculateSingleLogLikelihood(sim_icu, observed_data_.getNewICU(), "ICU Admissions");
             double ll_deaths = calculateSingleLogLikelihood(sim_deaths, observed_data_.getNewDeaths(), "Deaths");
             
             f_plus = ll_hosp + ll_icu + ll_deaths;
             
             if (std::isnan(f_plus) || std::isinf(f_plus)) {
                 f_plus = std::numeric_limits<double>::lowest();
             }
        } else {
            f_plus = std::numeric_limits<double>::lowest();
        }

        if (std::isfinite(f_plus)) {
            grad[i] = (f_plus - f_center) / eps_i;
        } else {
            grad[i] = 0.0;
        }
    }

    return f_center;
}

} // namespace epidemic
