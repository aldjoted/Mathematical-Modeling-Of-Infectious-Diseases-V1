#include "model/objectives/SEPAIHRDObjectiveFunction.hpp"
#include "exceptions/Exceptions.hpp"
#include "sir_age_structured/SimulationResultProcessor.hpp"
#include "utils/Logger.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <sstream>
#include <future>

namespace epidemic {

// Helper function to format parameter vector for logging
std::string formatParameters(const Eigen::VectorXd& params) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < params.size(); ++i) {
        oss << params[i] << (i == params.size() - 1 ? "" : ", ");
    }
    oss << "]";
    return oss.str();
}


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
    : model_(model),
      parameterManager_(parameterManager),
      cache_(cache),
      observed_data_(calibration_data),
      time_points_(time_points),
      initial_state_(initial_state),
      solver_strategy_(solver_strategy),
      abs_err_(abs_error),
      rel_err_(rel_error),
      simulator_(nullptr)
{
    const std::string F_NAME = "SEPAIHRDObjectiveFunction::Constructor";
    if (!model_) {
        Logger::getInstance().error(F_NAME, "Model pointer cannot be null.");
        THROW_INVALID_PARAM("SEPAIHRDObjectiveFunction", "Model pointer cannot be null.");
    }
    if (time_points_.empty()) {
        Logger::getInstance().error(F_NAME, "Time points vector cannot be empty.");
        THROW_INVALID_PARAM("SEPAIHRDObjectiveFunction", "Time points vector cannot be empty.");
    }
    if (initial_state_.size() == 0) {
        Logger::getInstance().error(F_NAME, "Initial state vector cannot be empty.");
        THROW_INVALID_PARAM("SEPAIHRDObjectiveFunction", "Initial state vector cannot be empty.");
    }
    if (!solver_strategy_) {
        Logger::getInstance().error(F_NAME, "Solver strategy cannot be null.");
        THROW_INVALID_PARAM("SEPAIHRDObjectiveFunction", "Solver strategy cannot be null.");
    }
    preallocateInternalMatrices();
    cached_sim_data_.invalidate();
    Logger::getInstance().debug(F_NAME, "Initialization successful.");
}

void SEPAIHRDObjectiveFunction::preallocateInternalMatrices() {
    const std::string F_NAME = "SEPAIHRDObjectiveFunction::preallocateInternalMatrices";
    if (model_ && !time_points_.empty()) {
        int rows = static_cast<int>(time_points_.size());
        int num_age_classes = model_->getNumAgeClasses();
        if (rows > 0 && num_age_classes > 0) {
            simulated_hospitalizations_.resize(rows, num_age_classes);
            simulated_icu_admissions_.resize(rows, num_age_classes);
            simulated_deaths_.resize(rows, num_age_classes);
            //Logger::getInstance().debug(F_NAME, "Preallocated internal matrices: " +
            //    std::to_string(rows) + "x" + std::to_string(num_age_classes));
        } else {
            Logger::getInstance().warning(F_NAME, "Cannot preallocate matrices: rows or age_classes is zero. Rows: " + std::to_string(rows) + ", Age Classes: " + std::to_string(num_age_classes));
        }
    } else {
        Logger::getInstance().error(F_NAME, "Cannot preallocate matrices: model is null or time_points_ is empty.");
        THROW_INVALID_PARAM("SEPAIHRDObjectiveFunction", "Model is null or time_points_ is empty.");
    }
}


double SEPAIHRDObjectiveFunction::calculate(const Eigen::VectorXd& parameters) {
    const std::string F_NAME = "SEPAIHRDObjectiveFunction::calculate";
    std::string cache_key_likelihood = cache_.createCacheKey(parameters);
    double cached_likelihood_value;

    if (cache_.getLikelihood(cache_key_likelihood, cached_likelihood_value)) {
        //Logger::getInstance().debug(F_NAME, "Likelihood " + std::to_string(cached_likelihood_value) + " retrieved from main cache for key: " + cache_key_likelihood);
        return cached_likelihood_value;
    }
    
    //Logger::getInstance().debug(F_NAME, "Calculating likelihood for parameters: " + formatParameters(parameters) + ". Main cache key: " + cache_key_likelihood);

    try {
        parameterManager_.updateModelParameters(parameters);
    } catch (const std::exception& e) {
        Logger::getInstance().warning(F_NAME, "Failed to update model parameters. Error: " + std::string(e.what()) + ". Parameters: " + formatParameters(parameters));
        cached_sim_data_.invalidate();
        return std::numeric_limits<double>::lowest();
    }
    
    if (!cached_sim_data_.isValid(parameters)) {
        //Logger::getInstance().debug(F_NAME, "Internal simulation data cache miss or invalid. Running simulation for parameters: " + formatParameters(parameters));
        ensureSimulatorExists();
        SimulationResult simulation_result;
        try {
            //Logger::getInstance().debug(F_NAME, "Running simulation.");
            simulation_result = simulator_->run(initial_state_, time_points_);
            if (!simulation_result.isValid()) {
                Logger::getInstance().warning(F_NAME, "Produced an invalid simulation result for parameters: " + formatParameters(parameters));
                cached_sim_data_.invalidate();
                return std::numeric_limits<double>::lowest();
            }
            //Logger::getInstance().debug(F_NAME, "Simulation successful. Caching I, H, D data.");

            int num_compartments = AgeSEPAIHRDSimulator::NUM_COMPARTMENTS;
            cached_sim_data_.I_data = SimulationResultProcessor::getCompartmentData(
                simulation_result, *model_, "I", num_compartments);
            cached_sim_data_.H_data = SimulationResultProcessor::getCompartmentData(
                simulation_result, *model_, "H", num_compartments);
            cached_sim_data_.D_data = SimulationResultProcessor::getCompartmentData(
                simulation_result, *model_, "D", num_compartments);
            cached_sim_data_.parameters_cache_key = parameters;
            cached_sim_data_.populated = true;

        } catch (const std::exception& e) {
            Logger::getInstance().warning(F_NAME, "Simulation failed. Error: " + std::string(e.what()) + ". Parameters: " + formatParameters(parameters));
            cached_sim_data_.invalidate();
            return std::numeric_limits<double>::lowest();
        }
    } else {
        Logger::getInstance().debug(F_NAME, "Using cached internal simulation data (I,H,D) for parameters: " + formatParameters(parameters));
    }
    
    if (simulated_hospitalizations_.rows() != static_cast<int>(time_points_.size()) || 
        simulated_hospitalizations_.cols() != model_->getNumAgeClasses()) {
        Logger::getInstance().warning(F_NAME, "Internal matrices incorrectly sized. Re-allocating. This should ideally not happen if constructor/preallocation worked.");
        preallocateInternalMatrices(); 
        if (simulated_hospitalizations_.rows() != static_cast<int>(time_points_.size()) || 
            simulated_hospitalizations_.cols() != model_->getNumAgeClasses()) {
             Logger::getInstance().error(F_NAME, "Failed to resize internal matrices. Aborting calculation.");
             return std::numeric_limits<double>::lowest();
        }
    }
    
    try {
        //Logger::getInstance().debug(F_NAME, "Extracting and calculating model-predicted incidence data using (potentially cached) I, H, D.");
        int num_age_classes = model_->getNumAgeClasses();
        int rows = static_cast<int>(time_points_.size());

        Eigen::VectorXd h_rates = model_->getHospRate(); 
        Eigen::VectorXd icu_admission_rates = model_->getIcuRate(); 
        
        // Calculate incidence using element-wise multiplication with age-specific rates
        // This is equivalent to multiplying each column of the compartment data by the corresponding rate.
        // Using rowwise multiplication with transposed rates is a clean way to express this.
        // The .matrix() call converts the resulting Array back to a Matrix for assignment.
        simulated_hospitalizations_ = (cached_sim_data_.I_data.array().rowwise() * h_rates.transpose().array()).matrix();
        simulated_icu_admissions_ = (cached_sim_data_.H_data.array().rowwise() * icu_admission_rates.transpose().array()).matrix();
        
        simulated_deaths_.setZero();
        if (rows > 0) {
            // This is D_cumulative(time_points[0]) - D_cumulative(initial_state_before_time_points[0])
            simulated_deaths_.row(0) = cached_sim_data_.D_data.row(0) - 
                                       initial_state_.segment(num_age_classes * AgeSEPAIHRDSimulator::D_COMPARTMENT_OFFSET, num_age_classes).transpose();
            
            // This is D_cumulative(time_points[t]) - D_cumulative(time_points[t-1])
            if (rows > 1) {
                simulated_deaths_.bottomRows(rows - 1) = cached_sim_data_.D_data.bottomRows(rows - 1) - 
                                                         cached_sim_data_.D_data.topRows(rows - 1);
            }
            
            simulated_deaths_ = simulated_deaths_.cwiseMax(0.0); 
        }

        //Logger::getInstance().debug(F_NAME, "Model-predicted incidence data calculation successful.");

    } catch (const std::exception& e) {
        Logger::getInstance().warning(F_NAME, "Failed to extract or calculate model-predicted incidence data. Error: " + std::string(e.what()) + ". Parameters: " + formatParameters(parameters));
        return std::numeric_limits<double>::lowest();
    }
    
    const Eigen::MatrixXd& observed_hospitalizations = observed_data_.getNewHospitalizations();
    const Eigen::MatrixXd& observed_icu_admissions = observed_data_.getNewICU();
    const Eigen::MatrixXd& observed_deaths = observed_data_.getNewDeaths();
    
    // Parallel likelihood calculation
    //Logger::getInstance().debug(F_NAME, "Calculating log-likelihoods (potentially in parallel).");

    auto future_ll_hosp = std::async(std::launch::async, [&]() {
        //Logger::getInstance().debug(F_NAME, "Calculating log-likelihood for hospitalizations (async).");
        return calculateSingleLogLikelihood(simulated_hospitalizations_, observed_hospitalizations, "Hospitalizations");
    });

    auto future_ll_icu = std::async(std::launch::async, [&]() {
        //Logger::getInstance().debug(F_NAME, "Calculating log-likelihood for ICU admissions (async).");
        return calculateSingleLogLikelihood(simulated_icu_admissions_, observed_icu_admissions, "ICU Admissions");
    });

    //Logger::getInstance().debug(F_NAME, "Calculating log-likelihood for deaths (sync).");
    double ll_deaths = calculateSingleLogLikelihood(simulated_deaths_, observed_deaths, "Deaths");
    
    double ll_hosp = future_ll_hosp.get();
    double ll_icu = future_ll_icu.get();
    
    double total_ll = ll_hosp + ll_icu + ll_deaths;

    // Check for NaN or infinity in total_ll, which can happen if individual LLs are lowest()
    if (std::isnan(total_ll) || std::isinf(total_ll)) {
        Logger::getInstance().warning(F_NAME, "Total log-likelihood is NaN or Inf. Parameters: " + formatParameters(parameters) +
                                             ". Hosp: " + std::to_string(ll_hosp) +
                                             ", ICU: " + std::to_string(ll_icu) +
                                             ", Deaths: " + std::to_string(ll_deaths));
        if (ll_hosp <= std::numeric_limits<double>::lowest() / 2.0 ||
            ll_icu  <= std::numeric_limits<double>::lowest() / 2.0 ||
            ll_deaths <= std::numeric_limits<double>::lowest() / 2.0 ) {
             total_ll = std::numeric_limits<double>::lowest();
        }
    }
    
    /*std::ostringstream oss_ll;
    oss_ll << "Total LL: " << total_ll << " (Hosp: " << ll_hosp << ", ICU: " << ll_icu << ", Deaths: " << ll_deaths << ") for parameters: " << formatParameters(parameters);
    Logger::getInstance().debug(F_NAME, oss_ll.str());*/
    
    cache_.storeLikelihood(cache_key_likelihood, total_ll);
    /*Logger::getInstance().debug(F_NAME, "Likelihood " + std::to_string(total_ll) + " stored in main cache for key: " + cache_key_likelihood);*/
    
    return total_ll;
}

const std::vector<std::string>& SEPAIHRDObjectiveFunction::getParameterNames() const {
    return parameterManager_.getParameterNames();
}

double SEPAIHRDObjectiveFunction::calculateSingleLogLikelihood(
    const Eigen::MatrixXd& simulated,
    const Eigen::MatrixXd& observed,
    const std::string& dataTypeForLog) const
{
    const std::string F_NAME = "SEPAIHRDObjectiveFunction::calculateSingleLogLikelihood (" + dataTypeForLog + ")";
    if (simulated.rows() != observed.rows() || simulated.cols() != observed.cols()) {
        std::ostringstream oss;
        oss << "Dimension mismatch. Simulated (" << simulated.rows() << "x" << simulated.cols()
            << ") vs Observed (" << observed.rows() << "x" << observed.cols() << ").";
        Logger::getInstance().error(F_NAME, oss.str());
        THROW_INVALID_PARAM("calculateSingleLogLikelihood", "Dimension mismatch for " + dataTypeForLog + ". " + oss.str());
    }
    
    const double epsilon = 1e-10;
    
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> valid_mask_array = 
        (observed.array() >= 0) && observed.array().isFinite();
    
    long long total_elements = observed.size();
    long long valid_points = valid_mask_array.count();
    long long skipped_points = total_elements - valid_points;

    if (skipped_points > 0) {
        Logger::getInstance().debug(F_NAME, "Skipped " + std::to_string(skipped_points) + " invalid (negative or NaN/Inf) observed data points out of " + std::to_string(total_elements) + " total.");
    }

    if (valid_points == 0) {
        Logger::getInstance().warning(F_NAME, "No valid data points found for likelihood calculation.");
        return std::numeric_limits<double>::lowest() / 3.0;
    }

    Eigen::MatrixXd sim_safe = simulated.cwiseMax(0.0).array() + epsilon;
    
    double log_likelihood = 0.0;

    Eigen::MatrixXd term1 = observed.array() * sim_safe.array().log();
    Eigen::MatrixXd term2 = sim_safe;
    Eigen::MatrixXd log_likelihood_terms = term1 - term2;

    for (int i = 0; i < observed.rows(); ++i) {
        for (int j = 0; j < observed.cols(); ++j) {
            if (valid_mask_array(i, j)) {
                log_likelihood += log_likelihood_terms(i, j);
            }
        }
    }
    
    if (std::isnan(log_likelihood) || std::isinf(log_likelihood)) {
         Logger::getInstance().warning(F_NAME, "Log-likelihood is NaN or Inf after calculation. LL: " + std::to_string(log_likelihood) + ". Valid points: " + std::to_string(valid_points));
         return std::numeric_limits<double>::lowest() / 3.0;
    }

    //Logger::getInstance().debug(F_NAME, "Calculated LL: " + std::to_string(log_likelihood) + " from " + std::to_string(valid_points) + " valid points.");
    
    return log_likelihood;
}

void SEPAIHRDObjectiveFunction::ensureSimulatorExists() {
    const std::string F_NAME = "SEPAIHRDObjectiveFunction::ensureSimulatorExists";
    if (!simulator_) {
        Logger::getInstance().debug(F_NAME, "Simulator instance not found, creating new instance.");
        if (time_points_.empty()) {
             Logger::getInstance().fatal(F_NAME, "Cannot create simulator: time_points_ is empty.");
             throw SimulationException("SEPAIHRDObjectiveFunction::ensureSimulatorExists", "Time points vector is empty, cannot determine simulation range.");
        }
        try {
            double start_time = time_points_.front();
            double end_time = time_points_.back();
            double time_step = 1.0;
            if (time_points_.size() > 1) {
                time_step = time_points_[1] - time_points_[0];
                if (time_step <= 0) {
                    Logger::getInstance().warning(F_NAME, "Calculated time step is not positive (" + std::to_string(time_step) + "). Defaulting to 1.0.");
                    time_step = 1.0;
                }
            } else if (time_points_.size() == 1) {
                 Logger::getInstance().debug(F_NAME, "Only one time point provided. Setting time_step to 1.0 for simulator (may not be used if end_time == start_time).");
            }
            
            simulator_ = std::make_unique<AgeSEPAIHRDSimulator>(
                model_, 
                solver_strategy_, 
                start_time, 
                end_time, 
                time_step, 
                abs_err_, 
                rel_err_
            );
           // Logger::getInstance().info(F_NAME, "Simulator instance created successfully.");
        } catch (const std::exception& e) {
            Logger::getInstance().fatal(F_NAME, "Failed to create simulator instance. Error: " + std::string(e.what()));
            throw SimulationException("SEPAIHRDObjectiveFunction::ensureSimulatorExists",
                                      std::string("Failed to create simulator: ") + e.what());
        }
    }
}

} // namespace epidemic