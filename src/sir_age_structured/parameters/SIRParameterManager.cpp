#include "sir_age_structured/parameters/SIRParamerterManager.hpp"
#include <iostream>

namespace epidemic {

const double DEFAULT_Q_SGM = 0.05;
const double DEFAULT_SCALE_C_SGM = 0.05;
const double DEFAULT_GAMMA_SGM = 0.01;

SIRParameterManager::SIRParameterManager(
    std::shared_ptr<AgeSIRModel> model,
    const std::vector<std::string>& params_to_calibrate,
    const std::map<std::string, double>& proposal_sigmas)
    : model_(model), param_names_(params_to_calibrate), proposal_sigmas_(proposal_sigmas)
{
    if (!model_) {
        THROW_INVALID_PARAM("SIRParameterManager", "Model pointer is null.");
    }
    if (param_names_.empty()) {
        THROW_INVALID_PARAM("SIRParameterManager", "Parameter names list cannot be empty.");
    }

    int n_ages = model_->getNumAgeClasses();
    param_name_to_index_.reserve(param_names_.size());

    for (size_t i = 0; i < param_names_.size(); ++i) {
        const std::string& name = param_names_[i];
        param_name_to_index_[name] = i;

        if (name == "q" || name == "scale_C_total") {
            // Valid parameter, nothing else to check
        } else if (name.rfind("gamma_", 0) == 0) {
            try {
                int age_idx = std::stoi(name.substr(6));
                if (age_idx < 0 || age_idx >= n_ages) {
                    throw ModelException("SIRParameterManager", 
                        "Invalid age index in parameter name '" + name + 
                        "'. Max index: " + std::to_string(n_ages - 1));
                }
            } catch (const std::invalid_argument& e) {
                throw ModelException("SIRParameterManager", 
                    "Could not parse age index from parameter name '" + name + 
                    "': Invalid argument");
            } catch (const std::out_of_range& e) {
                throw ModelException("SIRParameterManager", 
                    "Could not parse age index from parameter name '" + name + 
                    "': Index out of range");
            }
        } else {
            throw ModelException("SIRParameterManager", 
                "Parameter name '" + name + "' not recognized for AgeSIRModel calibration.");
        }

        if (proposal_sigmas_.find(name) == proposal_sigmas_.end()) {
            if (name == "q") proposal_sigmas_[name] = DEFAULT_Q_SGM;
            else if (name == "scale_C_total") proposal_sigmas_[name] = DEFAULT_SCALE_C_SGM;
            else if (name.rfind("gamma_", 0) == 0) proposal_sigmas_[name] = DEFAULT_GAMMA_SGM;
            
            std::cerr << "[ParamManager] Warning: No proposal sigma provided for '" 
                      << name << "', using default." << std::endl;
        }
    }
}

const std::vector<std::string>& SIRParameterManager::getParameterNames() const {
    return param_names_;
}

size_t SIRParameterManager::getParameterCount() const {
    return param_names_.size();
}

Eigen::VectorXd SIRParameterManager::getCurrentParameters() const {
    Eigen::VectorXd current_vec(param_names_.size());
    double current_q = model_->getTransmissibility();
    double current_scale_C = model_->getContactScaleFactor();
    Eigen::VectorXd current_gamma = model_->getRecoveryRate();

    for(size_t i = 0; i < param_names_.size(); ++i) {
        const std::string& name = param_names_[i];
        if (name == "q") {
            current_vec[i] = current_q;
        } else if (name == "scale_C_total") {
            current_vec[i] = current_scale_C;
        } else if (name.rfind("gamma_", 0) == 0) {
            int age_idx = std::stoi(name.substr(6)); 
            if (age_idx >= 0 && age_idx < current_gamma.size()) {
                current_vec[i] = current_gamma[age_idx];
            } else {
                throw ModelException("SIRParameterManager::getCurrentParameters", 
                    "Internal Error: Invalid age index encountered for '" + name + "'.");
            }
        }
    }
    return current_vec;
}

void SIRParameterManager::updateModelParameters(const Eigen::VectorXd& parameters) {
    if (static_cast<size_t>(parameters.size()) != param_names_.size()) {
        THROW_INVALID_PARAM("SIRParameterManager::updateModelParameters", 
            "Parameter vector size mismatch: expected " + std::to_string(param_names_.size()) + 
            ", got " + std::to_string(parameters.size()));
    }

    Eigen::VectorXd constrained_params = applyConstraints(parameters);

    bool gamma_needs_update = false;
    Eigen::VectorXd temp_gamma = model_->getRecoveryRate();

    for(size_t i = 0; i < param_names_.size(); ++i) {
        const std::string& name = param_names_[i];
        double value = constrained_params[i];

        if (name == "q") {
            model_->setTransmissibility(value);
        } else if (name == "scale_C_total") {
            model_->setContactScaleFactor(value);
        } else if (name.rfind("gamma_", 0) == 0) {
            int age_idx = std::stoi(name.substr(6));
            if (age_idx >= 0 && age_idx < temp_gamma.size()) {
                if (temp_gamma[age_idx] != value) {
                    temp_gamma[age_idx] = value;
                    gamma_needs_update = true;
                }
            } else {
                throw ModelException("SIRParameterManager::updateModelParameters", 
                    "Internal Error: Invalid age index encountered for '" + name + "'.");
            }
        }
    }
    
    if (gamma_needs_update) {
        model_->setRecoveryRate(temp_gamma);
    }
}

Eigen::VectorXd SIRParameterManager::applyConstraints(const Eigen::VectorXd& parameters) const {
    if (static_cast<size_t>(parameters.size()) != param_names_.size()) {
        THROW_INVALID_PARAM("SIRParameterManager::applyConstraints", 
            "Parameter vector size mismatch: expected " + std::to_string(param_names_.size()) + 
            ", got " + std::to_string(parameters.size()));
    }
    
    Eigen::VectorXd constrained_params = parameters;
    
    for(size_t i = 0; i < param_names_.size(); ++i) {
        const std::string& name = param_names_[i];
        if (name == "q") {
            constrained_params[i] = std::max(1e-12, parameters[i]);
        } else if (name == "scale_C_total" || name.rfind("gamma_", 0) == 0) {
            constrained_params[i] = std::max(0.0, parameters[i]);
        }
    }
    
    return constrained_params;
}

int SIRParameterManager::getIndexForParam(const std::string& name) const {
    auto it = param_name_to_index_.find(name);
    if (it != param_name_to_index_.end()) {
        return static_cast<int>(it->second);
    }
    return -1; // Parameter not found
}

double SIRParameterManager::getSigmaForParamIndex(int index) const {
    if (index < 0 || static_cast<size_t>(index) >= param_names_.size()) {
        throw std::out_of_range("[ParamManager] Index out of bounds in getSigmaForParamIndex: " + 
            std::to_string(index));
    }
    
    const std::string& name = param_names_[index];
    auto it = proposal_sigmas_.find(name);
    if (it != proposal_sigmas_.end()) {
        return it->second;
    } else {
        THROW_INVALID_PARAM("SIRParameterManager::getSigmaForParamIndex", 
            "Internal Error: Sigma not found for parameter: " + name);
    }
}

double SIRParameterManager::getDefaultSigmaForParam(const std::string& param_name) const {
    auto it = proposal_sigmas_.find(param_name);
    if (it != proposal_sigmas_.end()) {
        return it->second;
    } else {
        THROW_INVALID_PARAM("SIRParameterManager::getDefaultSigmaForParam", 
            "Default sigma not found for parameter: " + param_name);
    }
}

} // namespace epidemic