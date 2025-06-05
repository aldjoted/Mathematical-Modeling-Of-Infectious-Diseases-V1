#include "sir_age_structured/AgeSIRModel.hpp"
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <numeric> 
using namespace Eigen;
using namespace epidemic;
using namespace std;

std::shared_ptr<AgeSIRModel> AgeSIRModel::create(
    const Eigen::VectorXd& N,
    const Eigen::MatrixXd& C,
    const Eigen::VectorXd& gamma,
    double q,
    double scale_C) {
    int num_classes = N.size();
    if (num_classes <= 0) {
        throw ModelConstructionException("AgeSIRModel::create", "Number of age classes must be positive.");
    }
    if (C.rows() != num_classes || C.cols() != num_classes) {
        throw ModelConstructionException("AgeSIRModel::create", "Contact matrix dimensions (" + std::to_string(C.rows()) + "x" + std::to_string(C.cols()) + ") must match number of age classes (" + std::to_string(num_classes) + ").");
    }
    if (gamma.size() != num_classes) {
        throw ModelConstructionException("AgeSIRModel::create", "Gamma vector size (" + std::to_string(gamma.size()) + ") must match number of age classes (" + std::to_string(num_classes) + ").");
    }
    if ((N.array() < 0).any() || (gamma.array() < 0).any() || q < 0 || scale_C < 0) {
        throw ModelConstructionException("AgeSIRModel::create", "Initial rates (gamma), transmissibility (q), scaling (scale_C), or populations (N) cannot be negative.");
    }
    if ((C.array() < 0).any()) {
        throw ModelConstructionException("AgeSIRModel::create", "Baseline contact matrix cannot contain negative values.");
    }

    struct MakeSharedEnabler : public AgeSIRModel {
        MakeSharedEnabler(const Eigen::VectorXd& N, const Eigen::MatrixXd& C, const Eigen::VectorXd& gamma, double q, double scale_C)
        : AgeSIRModel(N, C, gamma, q, scale_C) {}
    };
    return std::make_shared<MakeSharedEnabler>(N, C, gamma, q, scale_C);
}

AgeSIRModel::AgeSIRModel(const Eigen::VectorXd& N_,
    const Eigen::MatrixXd& C_baseline_,
    const Eigen::VectorXd& gamma_,
    double q_,
    double scale_C_initial)
: num_age_classes(N_.size()), N(N_), gamma(gamma_), q(q_),
C_baseline(C_baseline_), scale_C_total(scale_C_initial),
baseline_q(q_), baseline_scale_C_total(scale_C_initial) 
{
    try {
        validate_parameters_nolock();
    } catch (const InvalidParameterException& e) {
        throw ModelConstructionException("AgeSIRModel::AgeSIRModel (private)", "Internal Error: Validation failed within private constructor: " + std::string(e.what()));
    }
    update_C_current_nolock();
}

// --- Internal Non-Locking Methods ---
void AgeSIRModel::update_C_current_nolock() {
    if (scale_C_total < 0) {
         cerr << "Warning: scale_C_total is negative (" << scale_C_total << "). Clamping to 1.0" << std::endl;
         scale_C_total = 1.0;
    }
   C_current = scale_C_total * C_baseline;
}

void AgeSIRModel::validate_parameters_nolock() const {
    if (num_age_classes <= 0) THROW_INVALID_PARAM("AgeSIRModel::validate_parameters", "Number of age classes must be positive." + std::to_string(num_age_classes));
    if (N.size() != num_age_classes) THROW_INVALID_PARAM("AgeSIRModel::validate_parameters", "N vector size mismatch.");
    if (gamma.size() != num_age_classes) THROW_INVALID_PARAM("AgeSIRModel::validate_parameters", "Gamma vector size mismatch.");
    if (C_baseline.rows() != num_age_classes || C_baseline.cols() != num_age_classes) THROW_INVALID_PARAM("AgeSIRModel::validate_parameters", "Baseline contact matrix dimensions mismatch.");

    if ((N.array() < 0).any())  THROW_INVALID_PARAM("AgeSIRModel::validate_parameters", "Population sizes (N) cannot be negative."); 
    if ((gamma.array() < 0).any()) THROW_INVALID_PARAM("AgeSIRModel::validate_parameters", "Recovery rates (gamma) cannot be negative.");
    if (q < 0) THROW_INVALID_PARAM("AgeSIRModel::validate_parameters", "Transmissibility (q) cannot be negative.");
    if (scale_C_total < 0) THROW_INVALID_PARAM("AgeSIRModel::validate_parameters", "Contact scale factor (scale_C_total) cannot be negative.");
    if ((C_baseline.array() < 0).any()) THROW_INVALID_PARAM("AgeSIRModel::validate_parameters", "Baseline contact matrix entries cannot be negative.");
}

void AgeSIRModel::setRecoveryRate_nolock(const VectorXd& new_gamma) {
    if (new_gamma.size() != num_age_classes) {
        THROW_INVALID_PARAM("AgeSIRModel::setRecoveryRate", "Recovery rate vector size (" + std::to_string(new_gamma.size()) + ") must match the number of age classes (" + std::to_string(num_age_classes) + ").");
    }
     if ((new_gamma.array() < 0).any()) {
        THROW_INVALID_PARAM("AgeSIRModel::setRecoveryRate", "Recovery rates cannot be negative.");
     }
    gamma = new_gamma;
}

void AgeSIRModel::setTransmissibility_nolock(double new_q) {
     if (new_q < 0) {
        THROW_INVALID_PARAM("AgeSIRModel::setTransmissibility", "Transmissibility (q) cannot be negative. Got: " + std::to_string(new_q));
     }
    q = new_q;
}

void AgeSIRModel::setContactScaleFactor_nolock(double new_scale) {
     if (new_scale < 0) {
          THROW_INVALID_PARAM("AgeSIRModel::setContactScaleFactor", "Contact matrix scaling factor cannot be negative. Got: " + std::to_string(new_scale));
     }
     scale_C_total = new_scale;
     // Note: update_C_current_nolock is called after all parameters are set in setCalibratableParams
     // or at the end of the public setContactScaleFactor.
}

// --- Public Methods ---
void AgeSIRModel::computeDerivatives(const std::vector<double>& state,
            std::vector<double>& derivatives,
            [[maybe_unused]] double time) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (state.size() != static_cast<size_t>(getStateSize()) || derivatives.size() != static_cast<size_t>(getStateSize())) {
        THROW_INVALID_PARAM("AgeSIRModel::computeDerivatives", "State or derivative vector size mismatch. Expected " + std::to_string(getStateSize()) + ", got state=" + std::to_string(state.size()) + ", derivatives=" + std::to_string(derivatives.size()) + ".");
    }

    Map<const VectorXd> S_current(&state[0], num_age_classes);
    Map<const VectorXd> I_current(&state[num_age_classes], num_age_classes);
    Map<const VectorXd> R_current(&state[2 * num_age_classes], num_age_classes);

    Eigen::VectorXd I_over_N = Eigen::VectorXd::Zero(num_age_classes);
    for(int j=0; j<num_age_classes; ++j) {
        if (N(j) > 1e-9) {
        I_over_N(j) = I_current(j) / N(j);
        }
    }
    VectorXd lambda = q * (C_current * I_over_N);
    lambda = lambda.cwiseMax(0.0);

    VectorXd dS = -lambda.array() * S_current.array();
    VectorXd dI = lambda.array() * S_current.array() - gamma.array() * I_current.array();
    VectorXd dR = gamma.array() * I_current.array();

    dS = (S_current.array() < 1e-9 && dS.array() < 0).select(VectorXd::Zero(num_age_classes), dS);
    dI = (I_current.array() < 1e-9 && dI.array() < 0).select(VectorXd::Zero(num_age_classes), dI);
    dR = (R_current.array() < 1e-9 && dR.array() < 0).select(VectorXd::Zero(num_age_classes), dR);

    Map<VectorXd>(&derivatives[0], num_age_classes) = dS;
    Map<VectorXd>(&derivatives[num_age_classes], num_age_classes) = dI;
    Map<VectorXd>(&derivatives[2 * num_age_classes], num_age_classes) = dR;
}

void AgeSIRModel::applyIntervention(const std::string& name,
                                    [[maybe_unused]] double time,
                                    const VectorXd& params) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (name == "contact_reduction" || name == "social_distancing" || name == "lockdown") {
        if (params.size() != 1) {
            throw InterventionException("AgeSIRModel::applyIntervention", "Intervention '" + name + "' requires exactly 1 parameter (overall contact scaling factor). Got " + std::to_string(params.size()) + ".");
        }
        double scale_factor = params(0);
        if (scale_factor < 0.0) {
            throw InterventionException("AgeSIRModel::applyIntervention", "Contact scaling factor for intervention '" + name + "' cannot be negative. Got " + std::to_string(scale_factor) + ".");
        }
        setContactScaleFactor_nolock(scale_C_total * scale_factor);
        update_C_current_nolock();
        std::cout << "Applied intervention '" << name << "' setting overall contact scale to: "
                   << scale_C_total << std::endl;
    } else if (name == "mask_mandate" || name == "transmission_reduction") {
        if (params.size() != 1) {
            throw InterventionException("AgeSIRModel::applyIntervention", "Intervention '" + name + "' requires exactly 1 parameter (transmission reduction factor [0,1]). Got " + std::to_string(params.size()) + ".");
        }
        double reduction_factor = params(0);
        if (reduction_factor < 0.0 || reduction_factor > 1.0) {
            throw InterventionException("AgeSIRModel::applyIntervention", "Transmission reduction factor for intervention '" + name + "' must be between 0 and 1. Got " + std::to_string(reduction_factor) + ".");
        }
        setTransmissibility_nolock(q * (1.0 - reduction_factor));
        std::cout << "Applied intervention '" << name << "' reducing current q by " << reduction_factor*100 << "%"
                  << " (new q = " << q << ")" << std::endl;
    }
    else {
        throw InterventionException("AgeSIRModel::applyIntervention", "Unknown intervention type: '" + name + "'.");
    }
}

void AgeSIRModel::reset() {
    std::lock_guard<std::mutex> lock(mutex_); 

    q = baseline_q;
    scale_C_total = baseline_scale_C_total;
    update_C_current_nolock();
    std::cout << "SIR model parameters reset to baseline (q=" << q << ", scale_C_total=" << scale_C_total << ")." << std::endl;
}

int AgeSIRModel::getStateSize() const {
    return 3 * num_age_classes;
}

std::vector<std::string> AgeSIRModel::getStateNames() const {
    std::vector<std::string> names;
    for (int i = 0; i < num_age_classes; ++i) {
        names.push_back("S" + std::to_string(i));
    }
    for (int i = 0; i < num_age_classes; ++i) {
        names.push_back("I" + std::to_string(i));
    }
    for (int i = 0; i < num_age_classes; ++i) {
        names.push_back("R" + std::to_string(i));
    }
    return names;
}

void AgeSIRModel::setRecoveryRate(const VectorXd& new_gamma) {
    std::lock_guard<std::mutex> lock(mutex_);
    setRecoveryRate_nolock(new_gamma);
}

void AgeSIRModel::setTransmissibility(double new_q) {
    std::lock_guard<std::mutex> lock(mutex_);
    setTransmissibility_nolock(new_q);
}

void AgeSIRModel::setContactScaleFactor(double new_scale) {
    std::lock_guard<std::mutex> lock(mutex_);
    setContactScaleFactor_nolock(new_scale);
    update_C_current_nolock();
}

int AgeSIRModel::getNumAgeClasses() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return num_age_classes;
}

Eigen::VectorXd AgeSIRModel::getPopulationSizes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return N;
}

Eigen::MatrixXd AgeSIRModel::getCurrentContactMatrix() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return C_current;
}

Eigen::MatrixXd AgeSIRModel::getBaselineContactMatrix() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return C_baseline;
}

Eigen::VectorXd AgeSIRModel::getRecoveryRate() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return gamma;
}

double AgeSIRModel::getTransmissibility() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return q;
}

double AgeSIRModel::getContactScaleFactor() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return scale_C_total;
}