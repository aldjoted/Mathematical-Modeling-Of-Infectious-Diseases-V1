#include "model/objectives/SEPAIHRDObjectiveFunction.hpp"
#include "model/parameters/SEPAIHRDParameterManager.hpp"
#include "exceptions/Exceptions.hpp"
#include "sir_age_structured/SimulationResultProcessor.hpp"
#include "utils/Logger.hpp"
#include "model/ModelConstants.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <future>

// OpenMP Support
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace epidemic {

std::string formatParameters(const Eigen::VectorXd& params); // Forward decl (assumed helper)

SEPAIHRDObjectiveFunction::SEPAIHRDObjectiveFunction(
    std::shared_ptr<AgeSEPAIHRDModel> model,
    IParameterManager& parameterManager,
    ISimulationCache& cache,
    const CalibrationData& calibration_data,
    const std::vector<double>& time_points,
    const Eigen::VectorXd& initial_state,
    std::shared_ptr<IOdeSolverStrategy> solver_strategy,
    double abs_error,
    double rel_error)
    : parameterManager_(parameterManager), model_(model), cache_(cache),
      observed_data_(calibration_data), time_points_(time_points),
      initial_state_(initial_state), solver_strategy_(solver_strategy),
      abs_err_(abs_error), rel_err_(rel_error), simulator_(nullptr)
{
    // Preallocation is kept for single-threaded usage or reference, 
    // but calculate() will use local variables for thread safety.
    preallocateInternalMatrices();
    cached_sim_data_.invalidate();
}

void SEPAIHRDObjectiveFunction::preallocateInternalMatrices() const {
    if (model_ && !time_points_.empty()) {
        int rows = static_cast<int>(time_points_.size());
        int num_age_classes = model_->getNumAgeClasses();
        simulated_hospitalizations_.resize(rows, num_age_classes);
        simulated_icu_admissions_.resize(rows, num_age_classes);
        simulated_deaths_.resize(rows, num_age_classes);
    }
}

double SEPAIHRDObjectiveFunction::calculate(const Eigen::VectorXd& parameters) const {
    std::string cache_key = cache_.createCacheKey(parameters);
    double cached_val;
    if (cache_.getLikelihood(cache_key, cached_val)) return cached_val;

    // Thread-safe execution: Clone the model and use local simulator/matrices
    
    // 1. Clone the model to avoid race conditions on mutable members and parameters
    auto local_model = model_->clone();
    
    // 2. Update parameters on the cloned model
    try {
        // Try to cast to SEPAIHRDParameterManager to use the thread-safe update method
        auto* sepaihrd_manager = dynamic_cast<SEPAIHRDParameterManager*>(&parameterManager_);
        if (sepaihrd_manager) {
            sepaihrd_manager->updateModelParameters(parameters, local_model);
        } else {
            // Fallback: Update the shared model (NOT THREAD SAFE, but best effort if cast fails)
            // This path should ideally not be taken in the current architecture
            parameterManager_.updateModelParameters(parameters);
            // If we fall back, we might need to clone AFTER update, but that's still racy.
            // Assuming SEPAIHRDParameterManager is always used.
        }
    } catch (...) { return std::numeric_limits<double>::lowest(); }

    // 3. Prepare initial state (Apply multipliers)
    int n_ages = local_model->getNumAgeClasses();
    Eigen::VectorXd init_state = initial_state_;
    
    SEPAIHRDParameters current_model_params = local_model->getModelParameters();
    
    // Check if we're using run-up strategy (runup_days > 0)
    double runup_days = current_model_params.runup_days;
    double seed_exposed = current_model_params.seed_exposed;
    
    if (runup_days > 0 && seed_exposed > 0) {
        // Run-up strategy: seed E compartment, zero out others
        const Eigen::VectorXd& N = local_model->getPopulationSizes();
        double total_pop = N.sum();
        for (int i = 0; i < n_ages; ++i) {
            double age_fraction = N(i) / total_pop;
            init_state(i + n_ages) = seed_exposed * age_fraction;  // E compartment
            init_state(i + 2*n_ages) = 0.0;  // P
            init_state(i + 3*n_ages) = 0.0;  // A
            init_state(i + 4*n_ages) = 0.0;  // I
            init_state(i + 5*n_ages) = 0.0;  // H
            init_state(i + 6*n_ages) = 0.0;  // ICU
            init_state(i + 7*n_ages) = 0.0;  // R
            init_state(i + 8*n_ages) = 0.0;  // D
            init_state(i + 9*n_ages) = 0.0;  // CumH
            init_state(i + 10*n_ages) = 0.0; // CumICU
        }
    } else {
        // Original behavior: apply multipliers
        init_state.segment(1*n_ages, n_ages) *= current_model_params.E0_multiplier;
        init_state.segment(2*n_ages, n_ages) *= current_model_params.P0_multiplier;
        init_state.segment(3*n_ages, n_ages) *= current_model_params.A0_multiplier;
        init_state.segment(4*n_ages, n_ages) *= current_model_params.I0_multiplier;
        init_state.segment(5*n_ages, n_ages) *= current_model_params.H0_multiplier;
        init_state.segment(6*n_ages, n_ages) *= current_model_params.ICU0_multiplier;
        init_state.segment(7*n_ages, n_ages) *= current_model_params.R0_multiplier;
        init_state.segment(8*n_ages, n_ages) *= current_model_params.D0_multiplier;
    }

    const Eigen::VectorXd& N = local_model->getPopulationSizes();
    for (int i = 0; i < n_ages; ++i) {
        double sum = 0;
        for (int j=1; j<constants::NUM_COMPARTMENTS_SEPAIHRD; ++j) sum += init_state(j*n_ages+i);
        if (sum > N(i)) return std::numeric_limits<double>::lowest();
        init_state(i) = N(i) - sum;
    }

    // 4. Run Simulation with Local Simulator
    AgeSEPAIHRDSimulator local_simulator(local_model, solver_strategy_, time_points_.front(), time_points_.back(), 1.0, abs_err_, rel_err_);
    auto res = local_simulator.run(init_state, time_points_);
    
    if (!res.isValid()) {
        return std::numeric_limits<double>::lowest();
    }

    // 5. Process Results into Local Matrices
    int nc = AgeSEPAIHRDSimulator::NUM_COMPARTMENTS;
    // We can't use cached_sim_data_ here as it's shared.
    Eigen::MatrixXd D_data = SimulationResultProcessor::getCompartmentData(res, *local_model, "D", nc);
    Eigen::MatrixXd CumH_data = SimulationResultProcessor::getCompartmentData(res, *local_model, "CumH", nc);
    Eigen::MatrixXd CumICU_data = SimulationResultProcessor::getCompartmentData(res, *local_model, "CumICU", nc);

    // === RUN-UP STRATEGY: Slice results to discard t < 0 ===
    // Find the index where t >= 0 (the first observation time)
    int runup_offset = 0;
    for (size_t i = 0; i < time_points_.size(); ++i) {
        if (time_points_[i] >= 0.0) {
            runup_offset = static_cast<int>(i);
            break;
        }
    }
    
    // Number of observation points (t >= 0)
    int num_obs_points = static_cast<int>(time_points_.size()) - runup_offset;
    
    // Validate dimensions match observed data
    if (num_obs_points != observed_data_.getNewDeaths().rows()) {
        return std::numeric_limits<double>::lowest();
    }

    // Calculate Derived Metrics (Daily Incidence from Cumulative States)
    // Work with full simulation data first, then slice
    Eigen::MatrixXd full_sim_hosp(CumH_data.rows(), CumH_data.cols());
    Eigen::MatrixXd full_sim_icu(CumICU_data.rows(), CumICU_data.cols());
    
    if (!time_points_.empty()) {
        // First row: Cum(t0) - Cum_initial (where t0 is -runup_days)
        full_sim_hosp.row(0) = CumH_data.row(0) - init_state.segment(n_ages * 9, n_ages).transpose();
        full_sim_icu.row(0) = CumICU_data.row(0) - init_state.segment(n_ages * 10, n_ages).transpose();
        
        // Subsequent rows: Cum(t) - Cum(t-1)
        if (time_points_.size() > 1) {
             full_sim_hosp.bottomRows(time_points_.size()-1) = CumH_data.bottomRows(time_points_.size()-1) - CumH_data.topRows(time_points_.size()-1);
             full_sim_icu.bottomRows(time_points_.size()-1) = CumICU_data.bottomRows(time_points_.size()-1) - CumICU_data.topRows(time_points_.size()-1);
        }
        
        // Ensure non-negative
        full_sim_hosp = full_sim_hosp.cwiseMax(0.0);
        full_sim_icu = full_sim_icu.cwiseMax(0.0);
    } else {
        full_sim_hosp.setZero(CumH_data.rows(), CumH_data.cols());
        full_sim_icu.setZero(CumICU_data.rows(), CumICU_data.cols());
    }
    
    Eigen::MatrixXd full_sim_deaths;
    if (!time_points_.empty()) {
        full_sim_deaths.resize(D_data.rows(), D_data.cols());
        full_sim_deaths.row(0) = D_data.row(0) - init_state.segment(n_ages * 8, n_ages).transpose();
        if (time_points_.size() > 1) {
             full_sim_deaths.bottomRows(time_points_.size()-1) = D_data.bottomRows(time_points_.size()-1) - D_data.topRows(time_points_.size()-1);
        }
        full_sim_deaths = full_sim_deaths.cwiseMax(0.0);
    } else {
        full_sim_deaths.setZero(D_data.rows(), D_data.cols());
    }

    // Slice to get only t >= 0 data for likelihood calculation
    Eigen::MatrixXd local_sim_hosp = full_sim_hosp.bottomRows(num_obs_points);
    Eigen::MatrixXd local_sim_icu = full_sim_icu.bottomRows(num_obs_points);
    Eigen::MatrixXd local_sim_deaths = full_sim_deaths.bottomRows(num_obs_points);

    // 6. Calculate Likelihood (Parallelized)
    auto f_hosp = std::async(std::launch::async, [&]{ 
        return calculateSingleLogLikelihood(local_sim_hosp, observed_data_.getNewHospitalizations(), "H"); 
    });
    auto f_icu = std::async(std::launch::async, [&]{ 
        return calculateSingleLogLikelihood(local_sim_icu, observed_data_.getNewICU(), "ICU"); 
    });
    
    double ll_deaths = calculateSingleLogLikelihood(local_sim_deaths, observed_data_.getNewDeaths(), "D");
    double total = f_hosp.get() + f_icu.get() + ll_deaths;

    if (std::isnan(total) || std::isinf(total)) total = std::numeric_limits<double>::lowest();
    
    cache_.storeLikelihood(cache_key, total);
    return total;
}

const std::vector<std::string>& SEPAIHRDObjectiveFunction::getParameterNames() const {
    return parameterManager_.getParameterNames();
}

// *** OPENMP PARALLELIZED LOG-LIKELIHOOD ***
double SEPAIHRDObjectiveFunction::calculateSingleLogLikelihood(
    const Eigen::MatrixXd& simulated,
    const Eigen::MatrixXd& observed,
    const std::string& dataType) const
{
    (void)dataType; // Unused parameter
    if (simulated.rows() != observed.rows() || simulated.cols() != observed.cols()) {
        return std::numeric_limits<double>::lowest();
    }

    const double epsilon = 1e-10;
    const int rows = observed.rows();
    const int cols = observed.cols();
    
    double log_likelihood = 0.0;

    // Flattened loop for OpenMP
    #pragma omp parallel for reduction(+:log_likelihood)
    for (int i = 0; i < rows; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < cols; ++j) {
            double obs = observed(i, j);
            // Check validity mask equivalent (obs >= 0 && finite)
            if (obs >= 0 && std::isfinite(obs)) {
                double sim = simulated(i, j);
                if (sim < 0) sim = 0.0;
                sim += epsilon;
                
                // Poisson LL: y * log(m) - m
                row_sum += (obs * std::log(sim) - sim);
            }
        }
        log_likelihood += row_sum;
    }

    return log_likelihood;
}

void SEPAIHRDObjectiveFunction::ensureSimulatorExists() const {
    if (!simulator_) {
        if (time_points_.empty()) throw std::runtime_error("No time points");
        simulator_ = std::make_unique<AgeSEPAIHRDSimulator>(model_, solver_strategy_, time_points_.front(), time_points_.back(), 1.0, abs_err_, rel_err_);
    }
}

} // namespace epidemic
