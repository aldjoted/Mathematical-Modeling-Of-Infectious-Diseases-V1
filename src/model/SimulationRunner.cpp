#include "model/SimulationRunner.hpp"
#include "model/AgeSEPAIHRDsimulator.hpp"
#include "model/PieceWiseConstantNPIStrategy.hpp"
#include "utils/Logger.hpp"
#include <functional>
#include <sstream>

namespace epidemic {

SimulationRunner::SimulationRunner(
    std::shared_ptr<AgeSEPAIHRDModel> model_template,
    std::shared_ptr<IOdeSolverStrategy> solver)
    : model_template_(model_template),
      solver_(solver) {
    
    if (!model_template_) {
        throw std::invalid_argument("SimulationRunner: Model template cannot be null");
    }
    if (!solver_) {
        throw std::invalid_argument("SimulationRunner: Solver strategy cannot be null");
    }
}

SimulationResult SimulationRunner::runSimulation(
    const SEPAIHRDParameters& params,
    const Eigen::VectorXd& initial_state,
    const std::vector<double>& time_points) {
    
    total_calls_++;
    
    // Generate cache key from parameters
    Eigen::VectorXd param_vec = parametersToVector(params);
    size_t cache_key = hashParameterVector(param_vec);
    
    // Check cache
    auto it = cache_.find(cache_key);
    if (it != cache_.end()) {
        cache_hits_++;
        Logger::getInstance().debug("SimulationRunner", 
            "Cache hit! (Hit rate: " + std::to_string(cache_hits_) + "/" + 
            std::to_string(total_calls_) + ")");
        return it->second;
    }
    
    // Cache miss - run simulation
    auto run_npi_strategy = model_template_->getNpiStrategy()->clone();
    
    // Fix: Update NPI strategy with sampled kappa values
    auto piecewise_npi = std::dynamic_pointer_cast<PiecewiseConstantNpiStrategy>(run_npi_strategy);
    if (piecewise_npi) {
        const size_t expected_calibratable = piecewise_npi->getNumCalibratableNpiParams();
        if (expected_calibratable > 0) {
            std::vector<double> calibratable_values;
            if (params.kappa_values.size() == expected_calibratable) {
                calibratable_values = params.kappa_values;
            } else if (piecewise_npi->isBaselineFixed() &&
                       params.kappa_values.size() == expected_calibratable + 1) {
                calibratable_values.assign(
                    params.kappa_values.begin() + 1,
                    params.kappa_values.end());
            } else if (!params.kappa_values.empty()) {
                Logger::getInstance().warning(
                    "SimulationRunner",
                    "Mismatch between provided kappa_values (" +
                        std::to_string(params.kappa_values.size()) +
                        ") and expected calibratable NPI params (" +
                        std::to_string(expected_calibratable) +
                        "). Attempting to align using the last expected entries.");
                if (params.kappa_values.size() >= expected_calibratable) {
                    calibratable_values.assign(
                        params.kappa_values.end() - expected_calibratable,
                        params.kappa_values.end());
                }
            }

            if (calibratable_values.size() == expected_calibratable) {
                piecewise_npi->setCalibratableValues(calibratable_values);
            } else {
                Logger::getInstance().error(
                    "SimulationRunner",
                    "Unable to align kappa_values with calibratable parameter count; "
                    "skipping NPI update to avoid throwing.");
            }
        }
        // setKappaEndTimes is not available in PiecewiseConstantNpiStrategy
        // Assuming end times are fixed during calibration or handled via constructor
    }

    auto run_model = std::make_shared<AgeSEPAIHRDModel>(params, run_npi_strategy);
    
    AgeSEPAIHRDSimulator simulator(
        run_model, 
        solver_, 
        time_points.front(), 
        time_points.back(), 
        1.0,  // dt
        1e-6, // atol
        1e-6  // rtol
    );
    
    SimulationResult result = simulator.run(initial_state, time_points);
    
    // Store in cache
    if (result.isValid()) {
        cache_[cache_key] = result;
    } else {
        Logger::getInstance().warning("SimulationRunner", "Invalid simulation result - not cached");
    }
    
    return result;
}

void SimulationRunner::clearCache() {
    cache_.clear();
    cache_hits_ = 0;
    total_calls_ = 0;
    Logger::getInstance().info("SimulationRunner", "Cache cleared");
}

std::pair<size_t, size_t> SimulationRunner::getCacheStats() const {
    return {cache_hits_, total_calls_};
}

size_t SimulationRunner::hashParameterVector(const Eigen::VectorXd& param_vec) const {
    // FIX Bug 4: Use boost::hash_combine pattern for better collision resistance
    // Also include vector size in hash to catch length mismatches
    size_t hash = 0;
    
    // Helper lambda implementing boost::hash_combine pattern
    auto hash_combine = [](size_t& seed, size_t value) {
        // Golden ratio-based mixing (boost::hash_combine pattern)
        seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    };
    
    // Include vector size in hash to prevent collisions between different-sized vectors
    hash_combine(hash, static_cast<size_t>(param_vec.size()));
    
    for (int i = 0; i < param_vec.size(); ++i) {
        // Round to prevent floating-point precision issues causing cache misses
        // Use a precision of 1e-12 to group nearly identical values
        double rounded = std::round(param_vec(i) * 1e12) / 1e12;
        
        // Use index in hash to distinguish position
        size_t value_hash = std::hash<double>{}(rounded);
        hash_combine(hash, value_hash);
        
        // Also mix in the index for position-sensitivity
        hash_combine(hash, static_cast<size_t>(i));
    }
    
    return hash;
}

Eigen::VectorXd SimulationRunner::parametersToVector(const SEPAIHRDParameters& params) const {
    // Flatten all parameters into a single vector for hashing
    std::vector<double> flat_params;
    
    // Basic scalars
    flat_params.push_back(params.beta);
    flat_params.push_back(params.sigma);
    flat_params.push_back(params.gamma_p);
    flat_params.push_back(params.gamma_A);
    flat_params.push_back(params.gamma_I);
    flat_params.push_back(params.gamma_H);
    flat_params.push_back(params.gamma_ICU);
    flat_params.push_back(params.theta);
    flat_params.push_back(params.contact_matrix_scaling_factor);
    
    // Age-specific vectors
    for (int i = 0; i < params.p.size(); ++i) flat_params.push_back(params.p(i));
    for (int i = 0; i < params.h.size(); ++i) flat_params.push_back(params.h(i));
    for (int i = 0; i < params.icu.size(); ++i) flat_params.push_back(params.icu(i));
    for (int i = 0; i < params.d_H.size(); ++i) flat_params.push_back(params.d_H(i));
    for (int i = 0; i < params.d_ICU.size(); ++i) flat_params.push_back(params.d_ICU(i));
    for (int i = 0; i < params.a.size(); ++i) flat_params.push_back(params.a(i));
    for (int i = 0; i < params.h_infec.size(); ++i) flat_params.push_back(params.h_infec(i));
    for (int i = 0; i < params.N.size(); ++i) flat_params.push_back(params.N(i));
    
    // Contact matrix (flatten row-major)
    for (int i = 0; i < params.M_baseline.rows(); ++i) {
        for (int j = 0; j < params.M_baseline.cols(); ++j) {
            flat_params.push_back(params.M_baseline(i, j));
        }
    }
    
    // Kappa values
    for (const auto& kappa : params.kappa_values) {
        flat_params.push_back(kappa);
    }
    
    // Kappa end times
    for (const auto& t : params.kappa_end_times) {
        flat_params.push_back(t);
    }
    
    // Beta values  
    for (const auto& beta_val : params.beta_values) {
        flat_params.push_back(beta_val);
    }
    
    // Beta end times
    for (const auto& t : params.beta_end_times) {
        flat_params.push_back(t);
    }
    
    // Convert to Eigen vector
    Eigen::VectorXd param_vec(flat_params.size());
    for (size_t i = 0; i < flat_params.size(); ++i) {
        param_vec(i) = flat_params[i];
    }
    
    return param_vec;
}

} // namespace epidemic
