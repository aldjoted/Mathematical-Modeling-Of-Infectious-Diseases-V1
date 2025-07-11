#include "model/parameters/SEPAIHRDParameterManager.hpp"
#include "model/PieceWiseConstantNPIStrategy.hpp"
#include "exceptions/Exceptions.hpp"
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>

namespace epidemic {

SEPAIHRDParameterManager::SEPAIHRDParameterManager(
    std::shared_ptr<AgeSEPAIHRDModel> model,
    const std::vector<std::string>& params_to_calibrate,
    const std::map<std::string, double>& proposal_sigmas,
    const std::map<std::string, std::pair<double, double>>& param_bounds)
    : model_(model),
      param_names_(params_to_calibrate),
      proposal_sigmas_(proposal_sigmas),
      param_bounds_(param_bounds)
{
    if (!model_) {
        THROW_INVALID_PARAM("SEPAIHRDParameterManager", "Model pointer cannot be null.");
    }
    if (param_names_.empty()) {
        THROW_INVALID_PARAM("SEPAIHRDParameterManager", "Parameter names list (params_to_calibrate) cannot be empty.");
    }

    int num_age_groups = model_->getNumAgeClasses();

    auto validate_age_param = [&](const std::string& param_name, const std::string& prefix) {
        if (param_name.rfind(prefix, 0) == 0) {
            try {
                size_t idx = std::stoul(param_name.substr(prefix.length()));
                if (idx >= static_cast<size_t>(num_age_groups)) {
                    THROW_INVALID_PARAM("SEPAIHRDParameterManager", "Invalid age index for parameter " + param_name);
                }
            } catch (const std::exception& e) {
                THROW_INVALID_PARAM("SEPAIHRDParameterManager", "Could not parse age index from: " + param_name);
            }
            return true;
        }
        return false;
    };

    for (const auto& name : param_names_) {
        if (proposal_sigmas_.find(name) == proposal_sigmas_.end()) {
            THROW_INVALID_PARAM("SEPAIHRDParameterManager", "Missing proposal sigma for parameter: " + name);
        }
        if (param_bounds_.find(name) == param_bounds_.end()) {
            THROW_INVALID_PARAM("SEPAIHRDParameterManager", "Missing bounds for parameter: " + name);
        }
        validate_age_param(name, "a_") || validate_age_param(name, "h_infec_") || validate_age_param(name, "p_") || validate_age_param(name, "h_") || validate_age_param(name, "icu_") || validate_age_param(name, "d_H_") || validate_age_param(name, "d_ICU_");

        if (name.rfind("beta_", 0) == 0) {
            try {
                std::stoul(name.substr(5)); // "beta_".length()
            } catch (const std::exception& e) {
                 THROW_INVALID_PARAM("SEPAIHRDParameterManager", "Could not parse index from beta parameter: " + name);
            }
        }
        else if (name.rfind("kappa_", 0) == 0) {
            auto npi_strat_base = model_->getNpiStrategy();
            if (!npi_strat_base) {
                THROW_INVALID_PARAM("SEPAIHRDParameterManager", "Cannot calibrate NPI parameter '" + name + "' as model has no NPI strategy.");
            }
            auto piecewise_npi_strat = std::dynamic_pointer_cast<PiecewiseConstantNpiStrategy>(npi_strat_base);
            if (!piecewise_npi_strat) {
                throw epidemic::ModelException("SEPAIHRDParameterManager", "NPI strategy is not a PiecewiseConstantNpiStrategy, cannot validate kappa names specifically.");
            }

            bool found_calibratable_kappa = false;
            for (size_t k_idx = 0; k_idx < piecewise_npi_strat->getNumCalibratableNpiParams(); ++k_idx) {
                if (piecewise_npi_strat->getNpiParamName(k_idx) == name) {
                    found_calibratable_kappa = true;
                    break;
                }
            }
            if (!found_calibratable_kappa) {
                if (piecewise_npi_strat->isBaselineFixed() && (name == "kappa_1" || name == "kappa_baseline")) {
                     THROW_INVALID_PARAM("SEPAIHRDParameterManager", "Parameter '" + name + "' refers to the fixed baseline kappa and cannot be in params_to_calibrate.");
                } else {
                     THROW_INVALID_PARAM("SEPAIHRDParameterManager", "NPI parameter '" + name + "' in params_to_calibrate is not recognized as a calibratable NPI parameter by the strategy.");
                }
            }
        }
    }
}

Eigen::VectorXd SEPAIHRDParameterManager::getCurrentParameters() const {
    Eigen::VectorXd current_params_vec(param_names_.size());
    const auto& model_params_struct = model_->getModelParameters();

    for (size_t i = 0; i < param_names_.size(); ++i) {
        const std::string& name = param_names_[i];
        if (name == "beta") current_params_vec[i] = model_params_struct.beta;
        else if (name.rfind("beta_", 0) == 0) { // Handle beta_1, beta_2, etc.
             try {
                size_t beta_idx = std::stoul(name.substr(5)) - 1; // "beta_1" -> index 0
                if (beta_idx < model_params_struct.beta_values.size()) {
                    current_params_vec[i] = model_params_struct.beta_values[beta_idx];
                } else {
                     throw ModelException("SEPAIHRDParameterManager::getCurrentParameters", "Beta name '" + name + "' implies out-of-bounds index.");
                }
            } catch (const std::exception& e) {
                throw ModelException("SEPAIHRDParameterManager::getCurrentParameters", "Error processing beta parameter '" + name + "': " + e.what());
            }
        }
        else if (name == "theta") current_params_vec[i] = model_params_struct.theta;
        else if (name == "sigma") current_params_vec[i] = model_params_struct.sigma;
        else if (name == "gamma_p") current_params_vec[i] = model_params_struct.gamma_p;
        else if (name == "gamma_A") current_params_vec[i] = model_params_struct.gamma_A;
        else if (name == "gamma_I") current_params_vec[i] = model_params_struct.gamma_I;
        else if (name == "gamma_H") current_params_vec[i] = model_params_struct.gamma_H;
        else if (name == "gamma_ICU") current_params_vec[i] = model_params_struct.gamma_ICU;
        else if (name == "E0_multiplier") current_params_vec[i] = model_params_struct.E0_multiplier;
        else if (name == "P0_multiplier") current_params_vec[i] = model_params_struct.P0_multiplier;
        else if (name == "A0_multiplier") current_params_vec[i] = model_params_struct.A0_multiplier;
        else if (name == "I0_multiplier") current_params_vec[i] = model_params_struct.I0_multiplier;
        else if (name == "H0_multiplier") current_params_vec[i] = model_params_struct.H0_multiplier;
        else if (name == "ICU0_multiplier") current_params_vec[i] = model_params_struct.ICU0_multiplier;
        else if (name == "R0_multiplier") current_params_vec[i] = model_params_struct.R0_multiplier;
        else if (name == "D0_multiplier") current_params_vec[i] = model_params_struct.D0_multiplier;
        else if (name.rfind("a_", 0) == 0) current_params_vec[i] = model_params_struct.a(std::stoul(name.substr(2)));
        else if (name.rfind("h_infec_", 0) == 0) current_params_vec[i] = model_params_struct.h_infec(std::stoul(name.substr(8)));
        else if (name.rfind("p_", 0) == 0) current_params_vec[i] = model_params_struct.p(std::stoul(name.substr(2)));
        else if (name.rfind("h_", 0) == 0) current_params_vec[i] = model_params_struct.h(std::stoul(name.substr(2)));
        else if (name.rfind("icu_", 0) == 0) current_params_vec[i] = model_params_struct.icu(std::stoul(name.substr(4)));
        else if (name.rfind("d_H_", 0) == 0) current_params_vec[i] = model_params_struct.d_H(std::stoul(name.substr(4)));
        else if (name.rfind("d_ICU_", 0) == 0) current_params_vec[i] = model_params_struct.d_ICU(std::stoul(name.substr(6)));
        else if (name.rfind("kappa_", 0) == 0) {
            try {
                size_t overall_kappa_idx = std::stoul(name.substr(6)) - 1;
                if (overall_kappa_idx < model_params_struct.kappa_values.size()) {
                    current_params_vec[i] = model_params_struct.kappa_values[overall_kappa_idx];
                } else {
                    throw ModelException("SEPAIHRDParameterManager::getCurrentParameters", "NPI name '" + name + "' implies out-of-bounds index.");
                }
            } catch (const std::exception& e) {
                throw ModelException("SEPAIHRDParameterManager::getCurrentParameters", "Error processing NPI parameter '" + name + "': " + e.what());
            }
        } else {
            THROW_INVALID_PARAM("SEPAIHRDParameterManager::getCurrentParameters", "Unknown parameter name: " + name);
        }
    }
    return current_params_vec;
}

void SEPAIHRDParameterManager::updateModelParameters(const Eigen::VectorXd& parameters_from_optimizer) {
    if (static_cast<size_t>(parameters_from_optimizer.size()) != param_names_.size()) {
        THROW_INVALID_PARAM("SEPAIHRDParameterManager::updateModelParameters", "Parameter vector size mismatch.");
    }

    Eigen::VectorXd constrained_params = applyConstraints(parameters_from_optimizer);
    
    // Get a mutable copy of the model's current parameters
    SEPAIHRDParameters updated_params = model_->getModelParameters();
    
    auto npi_strat_base = model_->getNpiStrategy();
    PiecewiseConstantNpiStrategy* piecewise_npi_strat = nullptr;
    if (npi_strat_base) {
        piecewise_npi_strat = dynamic_cast<PiecewiseConstantNpiStrategy*>(npi_strat_base.get());
    }

    // Prepare to collect NPI values if they are being calibrated
    bool npi_values_need_update = false;
    std::vector<double> collected_calibratable_npi_values;
    if (piecewise_npi_strat) {
        collected_calibratable_npi_values = piecewise_npi_strat->getCalibratableValues();
    }

    for (size_t i = 0; i < param_names_.size(); ++i) {
        const std::string& name = param_names_[i];
        double value = constrained_params[i];
        
        if (name == "beta") { updated_params.beta = value; }
        else if (name.rfind("beta_", 0) == 0) {
            try {
                size_t beta_idx = std::stoul(name.substr(5)) - 1; // "beta_1" -> index 0
                if (beta_idx < updated_params.beta_values.size()) {
                    updated_params.beta_values[beta_idx] = value;
                } else {
                    THROW_INVALID_PARAM("updateModelParameters", "Beta index out of range for name: " + name);
                }
            } catch (const std::exception& e) {
                THROW_INVALID_PARAM("updateModelParameters", "Invalid beta parameter name: " + name);
            }
        }
        else if (name == "theta") { updated_params.theta = value; }
        else if (name == "sigma") { updated_params.sigma = value; }
        else if (name == "gamma_p") { updated_params.gamma_p = value; }
        else if (name == "gamma_A") { updated_params.gamma_A = value; }
        else if (name == "gamma_I") { updated_params.gamma_I = value; }
        else if (name == "gamma_H") { updated_params.gamma_H = value; }
        else if (name == "gamma_ICU") { updated_params.gamma_ICU = value; }
        else if (name.rfind("a_", 0) == 0) { size_t idx = std::stoul(name.substr(2)); updated_params.a(idx) = value; }
        else if (name.rfind("h_infec_", 0) == 0) { size_t idx = std::stoul(name.substr(8)); updated_params.h_infec(idx) = value; }
        else if (name.rfind("p_", 0) == 0) { size_t idx = std::stoul(name.substr(2)); updated_params.p(idx) = value; }
        else if (name.rfind("h_", 0) == 0) { size_t idx = std::stoul(name.substr(2)); updated_params.h(idx) = value; }
        else if (name.rfind("icu_", 0) == 0) { size_t idx = std::stoul(name.substr(4)); updated_params.icu(idx) = value; }
        else if (name.rfind("d_H_", 0) == 0) { size_t idx = std::stoul(name.substr(4)); updated_params.d_H(idx) = value; }
        else if (name.rfind("d_ICU_", 0) == 0) { size_t idx = std::stoul(name.substr(6)); updated_params.d_ICU(idx) = value; }
        else if (name == "E0_multiplier") { updated_params.E0_multiplier = value; }
        else if (name == "P0_multiplier") { updated_params.P0_multiplier = value; }
        else if (name == "A0_multiplier") { updated_params.A0_multiplier = value; }
        else if (name == "I0_multiplier") { updated_params.I0_multiplier = value; }
        else if (name == "H0_multiplier") { updated_params.H0_multiplier = value; }
        else if (name == "ICU0_multiplier") { updated_params.ICU0_multiplier = value; }
        else if (name == "R0_multiplier") { updated_params.R0_multiplier = value; }
        else if (name == "D0_multiplier") { updated_params.D0_multiplier = value; }       
        else if (name.rfind("kappa_", 0) == 0) {
            if (piecewise_npi_strat) {
                bool found = false;
                for(size_t npi_cal_idx = 0; npi_cal_idx < piecewise_npi_strat->getNumCalibratableNpiParams(); ++npi_cal_idx) {
                    if (piecewise_npi_strat->getNpiParamName(npi_cal_idx) == name) {
                        collected_calibratable_npi_values[npi_cal_idx] = value;
                        npi_values_need_update = true;
                        found = true;
                        break; 
                    }
                }
                if (!found) {
                     std::cerr << "[Warning] SEPAIHRDParameterManager: NPI param '" << name << "' not found as calibratable." << std::endl;
                }
            } else {
                 throw ModelException("updateModelParameters", "Attempting to update kappa but NPI strategy is null/not piecewise.");
            }
        } else {
             std::cerr << "[Warning] SEPAIHRDParameterManager: Unknown parameter name '" << name << "' encountered." << std::endl;
        }
    }

    // Set the full parameter struct back to the model.
    // The model's setModelParameters method is responsible for updating all internal state,
    // including re-initializing the beta_strategy_.
    model_->setModelParameters(updated_params);

    // If any NPI parameters were changed, update the NPI strategy object.
    // This is a direct modification of the strategy object held by the model.
    if (npi_values_need_update && piecewise_npi_strat) {
        piecewise_npi_strat->setCalibratableValues(collected_calibratable_npi_values);
    }
}

double SEPAIHRDParameterManager::getSigmaForParamIndex(int index) const {
    if (index < 0 || static_cast<size_t>(index) >= param_names_.size()) {
        throw std::out_of_range("SEPAIHRDParameterManager::getSigmaForParamIndex: Index out of bounds.");
    }
    const std::string& name = param_names_[index];
    auto it = proposal_sigmas_.find(name);
    if (it == proposal_sigmas_.end()) {
        THROW_INVALID_PARAM("SEPAIHRDParameterManager::getSigmaForParamIndex", "Sigma not found for parameter: " + name);
    }
    return it->second;
}

Eigen::VectorXd SEPAIHRDParameterManager::applyConstraints(const Eigen::VectorXd& parameters) const {
    if (static_cast<size_t>(parameters.size()) != param_names_.size()) {
        THROW_INVALID_PARAM("SEPAIHRDParameterManager::applyConstraints", "Parameter vector size mismatch.");
    }
    Eigen::VectorXd constrained_params = parameters;
    for (size_t i = 0; i < param_names_.size(); ++i) {
        const std::string& name = param_names_[i];
        auto it = param_bounds_.find(name);
        if (it != param_bounds_.end()) {
            double min_bound = it->second.first;
            double max_bound = it->second.second;
            if (min_bound > max_bound) {
                 std::cerr << "[Warning] SEPAIHRDParameterManager: Min bound > Max bound for parameter " << name
                           << " (" << min_bound << " > " << max_bound << "). Using min bound only for max, and max for min if value is outside." << std::endl;
                if (constrained_params[i] < min_bound) constrained_params[i] = min_bound;
            }
            constrained_params[i] = std::max(min_bound, std::min(max_bound, parameters[i]));
        } else {
             std::cerr << "[Warning] SEPAIHRDParameterManager: Bounds not found for parameter during constraint application: " << name << ". Applying default non-negativity." << std::endl;
             constrained_params[i] = std::max(0.0, parameters[i]);
        }
    }
    return constrained_params;
}

int SEPAIHRDParameterManager::getIndexForParam(const std::string& name) const {
     auto it = std::find(param_names_.begin(), param_names_.end(), name);
     if (it != param_names_.end()) {
         return std::distance(param_names_.begin(), it);
     }
     return -1; 
}

double SEPAIHRDParameterManager::getLowerBoundForParamIndex(int idx) const {
    if (idx < 0 || static_cast<size_t>(idx) >= param_names_.size()) {
        THROW_OUT_OF_RANGE("SEPAIHRDParameterManager::getLowerBoundForParamIndex", "Index out of bounds.");
    }
    const auto& name = param_names_.at(idx);
    auto it = param_bounds_.find(name);
    if (it == param_bounds_.end()) {
         THROW_INVALID_PARAM("SEPAIHRDParameterManager::getLowerBoundForParamIndex", "Bounds not found for parameter: " + name);
    }
    return it->second.first;
}

double SEPAIHRDParameterManager::getUpperBoundForParamIndex(int idx) const {
    if (idx < 0 || static_cast<size_t>(idx) >= param_names_.size()) {
        THROW_OUT_OF_RANGE("SEPAIHRDParameterManager::getUpperBoundForParamIndex", "Index out of bounds.");
    }
    const auto& name = param_names_.at(idx);
     auto it = param_bounds_.find(name);
    if (it == param_bounds_.end()) {
         THROW_INVALID_PARAM("SEPAIHRDParameterManager::getUpperBoundForParamIndex", "Bounds not found for parameter: " + name);
    }
    return it->second.second;
}

} // namespace epidemic