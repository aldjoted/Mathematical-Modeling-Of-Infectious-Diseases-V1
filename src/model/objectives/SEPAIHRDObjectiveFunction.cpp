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

#include "sir_age_structured/caching/SimulationCache.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace epidemic {

std::string formatParameters(const Eigen::VectorXd& params);

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
    sepaihrd_manager_ = dynamic_cast<SEPAIHRDParameterManager*>(&parameterManager_);

    runup_offset_ = 0;
    for (size_t i = 0; i < time_points_.size(); ++i) {
        if (time_points_[i] >= 0.0) {
            runup_offset_ = static_cast<int>(i);
            break;
        }
    }
    num_obs_points_ = static_cast<int>(time_points_.size()) - runup_offset_;

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
    SimulationCache* fast_cache = dynamic_cast<SimulationCache*>(&cache_);
    size_t fast_key = 0;

    if (fast_cache) {
        fast_key = fast_cache->computeHash(parameters);
        double cached_val;
        if (fast_cache->getLikelihood(fast_key, cached_val)) return cached_val;
    }

    std::string cache_key;
    if (!fast_cache) {
        cache_key = cache_.createCacheKey(parameters);
        double cached_val;
        if (cache_.getLikelihood(cache_key, cached_val)) return cached_val;
    }

    // Thread-local context: safe for parallel objective evaluations (e.g., PSO with OpenMP)
    struct ThreadLocalContext {
        std::shared_ptr<AgeSEPAIHRDModel> model;
        std::unique_ptr<AgeSEPAIHRDSimulator> simulator;
        Eigen::VectorXd init_state;
        Eigen::VectorXd age_fraction;
        double total_pop = 0.0;

        Eigen::MatrixXd sim_hosp;
        Eigen::MatrixXd sim_icu;
        Eigen::MatrixXd sim_deaths;
    };

    thread_local const SEPAIHRDObjectiveFunction* owner = nullptr;
    thread_local ThreadLocalContext ctx;
    if (owner != this) {
        owner = this;
        ctx = ThreadLocalContext{};
    }

    if (!ctx.model) {
        ctx.model = model_->clone();
        ctx.init_state = initial_state_;
        const Eigen::VectorXd& N = ctx.model->getPopulationSizes();
        ctx.total_pop = N.sum();
        if (ctx.total_pop > 0.0) {
            ctx.age_fraction = N / ctx.total_pop;
        } else {
            ctx.age_fraction = Eigen::VectorXd::Zero(N.size());
        }

        if (time_points_.empty()) {
            return std::numeric_limits<double>::lowest();
        }
        ctx.simulator = std::make_unique<AgeSEPAIHRDSimulator>(
            ctx.model, solver_strategy_, time_points_.front(), time_points_.back(), 1.0, abs_err_, rel_err_);
    }
    
    try {
        if (!sepaihrd_manager_) {
            return std::numeric_limits<double>::lowest();
        }
        sepaihrd_manager_->updateModelParameters(parameters, ctx.model);
    } catch (...) { return std::numeric_limits<double>::lowest(); }

    int n_ages = ctx.model->getNumAgeClasses();
    Eigen::VectorXd& init_state = ctx.init_state;
    init_state = initial_state_;
    
    double runup_days = ctx.model->getRunupDays();
    double seed_exposed = ctx.model->getSeedExposed();
    
    if (runup_days > 0 && seed_exposed > 0) {
        for (int i = 0; i < n_ages; ++i) {
            init_state(i + n_ages) = seed_exposed * ctx.age_fraction(i);
            init_state(i + 2*n_ages) = 0.0;
            init_state(i + 3*n_ages) = 0.0;
            init_state(i + 4*n_ages) = 0.0;  // I
            init_state(i + 5*n_ages) = 0.0;  // H
            init_state(i + 6*n_ages) = 0.0;  // ICU
            init_state(i + 7*n_ages) = 0.0;  // R
            init_state(i + 8*n_ages) = 0.0;  // D
            init_state(i + 9*n_ages) = 0.0;  // CumH
            init_state(i + 10*n_ages) = 0.0; // CumICU
        }
    } else {
        init_state.segment(1*n_ages, n_ages) *= ctx.model->getE0Multiplier();
        init_state.segment(2*n_ages, n_ages) *= ctx.model->getP0Multiplier();
        init_state.segment(3*n_ages, n_ages) *= ctx.model->getA0Multiplier();
        init_state.segment(4*n_ages, n_ages) *= ctx.model->getI0Multiplier();
        init_state.segment(5*n_ages, n_ages) *= ctx.model->getH0Multiplier();
        init_state.segment(6*n_ages, n_ages) *= ctx.model->getICU0Multiplier();
        init_state.segment(7*n_ages, n_ages) *= ctx.model->getR0Multiplier();
        init_state.segment(8*n_ages, n_ages) *= ctx.model->getD0Multiplier();
    }

    const Eigen::VectorXd& N = ctx.model->getPopulationSizes();
    for (int i = 0; i < n_ages; ++i) {
        double sum = 0;
        for (int j = 1; j < constants::NUM_POPULATION_COMPARTMENTS_SEPAIHRD; ++j) {
            sum += init_state(j * n_ages + i);
        }
        if (sum > N(i)) return std::numeric_limits<double>::lowest();
        init_state(i) = N(i) - sum;
    }

    auto res = ctx.simulator->run(init_state, time_points_);
    
    if (!res.isValid()) {
        return std::numeric_limits<double>::lowest();
    }

    int nc = AgeSEPAIHRDSimulator::NUM_COMPARTMENTS;
    Eigen::MatrixXd D_data = SimulationResultProcessor::getCompartmentData(res, *ctx.model, "D", nc);
    Eigen::MatrixXd CumH_data = SimulationResultProcessor::getCompartmentData(res, *ctx.model, "CumH", nc);
    Eigen::MatrixXd CumICU_data = SimulationResultProcessor::getCompartmentData(res, *ctx.model, "CumICU", nc);

    if (num_obs_points_ != observed_data_.getNewDeaths().rows()) {
        return std::numeric_limits<double>::lowest();
    }

    // Thread-local incidence matrices for safe parallel evaluation
    if (ctx.sim_hosp.rows() != CumH_data.rows() || ctx.sim_hosp.cols() != CumH_data.cols()) {
        ctx.sim_hosp.resize(CumH_data.rows(), CumH_data.cols());
    }
    if (ctx.sim_icu.rows() != CumICU_data.rows() || ctx.sim_icu.cols() != CumICU_data.cols()) {
        ctx.sim_icu.resize(CumICU_data.rows(), CumICU_data.cols());
    }
    if (ctx.sim_deaths.rows() != D_data.rows() || ctx.sim_deaths.cols() != D_data.cols()) {
        ctx.sim_deaths.resize(D_data.rows(), D_data.cols());
    }
    
    if (!time_points_.empty()) {
        ctx.sim_hosp.row(0) = CumH_data.row(0) - init_state.segment(n_ages * 9, n_ages).transpose();
        ctx.sim_icu.row(0) = CumICU_data.row(0) - init_state.segment(n_ages * 10, n_ages).transpose();
        
        if (time_points_.size() > 1) {
             ctx.sim_hosp.bottomRows(time_points_.size()-1) = CumH_data.bottomRows(time_points_.size()-1) - CumH_data.topRows(time_points_.size()-1);
             ctx.sim_icu.bottomRows(time_points_.size()-1) = CumICU_data.bottomRows(time_points_.size()-1) - CumICU_data.topRows(time_points_.size()-1);
        }
        
        ctx.sim_hosp = ctx.sim_hosp.cwiseMax(0.0);
        ctx.sim_icu = ctx.sim_icu.cwiseMax(0.0);
    } else {
        ctx.sim_hosp.setZero(CumH_data.rows(), CumH_data.cols());
        ctx.sim_icu.setZero(CumICU_data.rows(), CumICU_data.cols());
    }

    if (!time_points_.empty()) {
        ctx.sim_deaths.row(0) = D_data.row(0) - init_state.segment(n_ages * 8, n_ages).transpose();
        if (time_points_.size() > 1) {
             ctx.sim_deaths.bottomRows(time_points_.size()-1) = D_data.bottomRows(time_points_.size()-1) - D_data.topRows(time_points_.size()-1);
        }
        ctx.sim_deaths = ctx.sim_deaths.cwiseMax(0.0);
    } else {
        ctx.sim_deaths.setZero(D_data.rows(), D_data.cols());
    }

    // Compute log-likelihoods for each data stream
    const auto local_sim_hosp = ctx.sim_hosp.bottomRows(num_obs_points_);
    const auto local_sim_icu = ctx.sim_icu.bottomRows(num_obs_points_);
    const auto local_sim_deaths = ctx.sim_deaths.bottomRows(num_obs_points_);

    double ll_hosp = calculateSingleLogLikelihood(local_sim_hosp, observed_data_.getNewHospitalizations(), "H");
    double ll_icu = calculateSingleLogLikelihood(local_sim_icu, observed_data_.getNewICU(), "ICU");
    double ll_deaths = calculateSingleLogLikelihood(local_sim_deaths, observed_data_.getNewDeaths(), "D");
    double total = ll_hosp + ll_icu + ll_deaths;

    if (std::isnan(total) || std::isinf(total)) total = std::numeric_limits<double>::lowest();
    
    if (fast_cache) {
        fast_cache->storeLikelihood(fast_key, total);
    } else {
        cache_.storeLikelihood(cache_key, total);
    }
    return total;
}

const std::vector<std::string>& SEPAIHRDObjectiveFunction::getParameterNames() const {
    return parameterManager_.getParameterNames();
}

double SEPAIHRDObjectiveFunction::calculateSingleLogLikelihood(
    const Eigen::MatrixXd& simulated,
    const Eigen::MatrixXd& observed,
    const std::string& dataType) const
{
    (void)dataType;
    if (simulated.rows() != observed.rows() || simulated.cols() != observed.cols()) {
        return std::numeric_limits<double>::lowest();
    }

    const double epsilon = 1e-10;
    const int rows = observed.rows();
    const int cols = observed.cols();
    
    double log_likelihood = 0.0;

    bool use_parallel = false;
#if defined(_OPENMP)
    // Avoid nested parallelism when already inside parallel region
    use_parallel = (!omp_in_parallel()) && (static_cast<long long>(rows) * cols >= 256);
#endif

    #pragma omp parallel for reduction(+:log_likelihood) if(use_parallel)
    for (int i = 0; i < rows; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < cols; ++j) {
            const double obs = observed(i, j);
            if (obs >= 0.0 && std::isfinite(obs)) {
                double sim = simulated(i, j);
                if (sim < 0.0) sim = 0.0;
                sim += epsilon;
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
