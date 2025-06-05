#include "model/AgeSEPAIHRDModel.hpp"
#include "exceptions/Exceptions.hpp"
#include "model/ModelConstants.hpp"
#include <stdexcept>
#include "utils/Logger.hpp"
#include <vector>
#include <string>
#include <mutex>
#include <numeric>
#include <algorithm>
#include <iostream>

namespace epidemic {

    AgeSEPAIHRDModel::AgeSEPAIHRDModel(const SEPAIHRDParameters& params, std::shared_ptr<INpiStrategy> npi_strategy_ptr)
        : num_age_classes(params.N.size()), N(params.N), M_baseline(params.M_baseline),
          beta(params.beta), theta(params.theta),
          sigma(params.sigma), gamma_p(params.gamma_p), gamma_A(params.gamma_A), gamma_I(params.gamma_I),
          gamma_H(params.gamma_H), gamma_ICU(params.gamma_ICU), p(params.p), h(params.h), icu(params.icu),
          d_H(params.d_H), d_ICU(params.d_ICU),
          npi_strategy(npi_strategy_ptr), baseline_beta(params.beta), baseline_theta(params.theta) {
    
        if (!params.validate()) {
            THROW_INVALID_PARAM("AgeSEPAIHRDModel", "Invalid SEPAIHRD parameters dimensions or sizes.");
        }
        if (!npi_strategy) {
             THROW_INVALID_PARAM("AgeSEPAIHRDModel", "NPI strategy pointer cannot be null.");
        }
        if (beta < 0 || theta < 0 || sigma < 0 || gamma_p < 0 || gamma_A < 0 || gamma_I < 0 || gamma_H < 0 || gamma_ICU < 0) {
             THROW_INVALID_PARAM("AgeSEPAIHRDModel", "Rate parameters cannot be negative.");
        }
        if ((p.array() < 0).any() || (p.array() > 1).any() ||
            (h.array() < 0).any() || (icu.array() < 0).any() ||
            (d_H.array() < 0).any() || (d_ICU.array() < 0).any()) {
             THROW_INVALID_PARAM("AgeSEPAIHRDModel", "Age-specific rate/probability parameters cannot be negative (p must be <= 1).");
        }
    }
    
    void AgeSEPAIHRDModel::computeDerivatives(const std::vector<double>& state,
                                             std::vector<double>& derivatives,
                                             double time) {
        std::lock_guard<std::mutex> lock(mutex_);
    
        int n = num_age_classes;
        if (state.size() != static_cast<size_t>(getStateSize()) || derivatives.size() != static_cast<size_t>(getStateSize())) {
             THROW_INVALID_PARAM("AgeSEPAIHRDModel::computeDerivatives", "State or derivatives vector size mismatch.");
        }
    
        Eigen::Map<const Eigen::VectorXd> S(&state[0*n], n);
        Eigen::Map<const Eigen::VectorXd> E(&state[1*n], n);
        Eigen::Map<const Eigen::VectorXd> P(&state[2*n], n);
        Eigen::Map<const Eigen::VectorXd> A(&state[3*n], n);
        Eigen::Map<const Eigen::VectorXd> I(&state[4*n], n);
        Eigen::Map<const Eigen::VectorXd> H(&state[5*n], n);
        Eigen::Map<const Eigen::VectorXd> ICU_(&state[6*n], n);

        double current_reduction_factor = npi_strategy->getReductionFactor(time);
        if (current_reduction_factor < 0) {
            throw SimulationException("AgeSEPAIHRDModel::computeDerivatives", "NPI reduction factor cannot be negative.");
        }
        
        Eigen::MatrixXd effective_contact = beta * current_reduction_factor * M_baseline;

        Eigen::VectorXd infectious_load_per_capita = Eigen::VectorXd::Zero(n);
        for (int j = 0; j < n; ++j) {
            if (N(j) > constants::MIN_POPULATION_FOR_DIVISION) {
                infectious_load_per_capita(j) = (P(j) + A(j) + theta * I(j)) / N(j);
            }
        }
        Eigen::VectorXd lambda = effective_contact * infectious_load_per_capita;
        lambda = lambda.cwiseMax(0.0);
    
        Eigen::VectorXd dS = -lambda.array() * S.array();
        Eigen::VectorXd dE = lambda.array() * S.array() - sigma * E.array();
        Eigen::VectorXd dP = sigma * E.array() - gamma_p * P.array();
        Eigen::VectorXd dA = p.array() * gamma_p * P.array() - gamma_A * A.array();
        Eigen::VectorXd dI = (1.0 - p.array()) * gamma_p * P.array() - (gamma_I + h.array()) * I.array();
        Eigen::VectorXd dH = h.array() * I.array() - (gamma_H + d_H.array() + icu.array()) * H.array();
        Eigen::VectorXd dICU = icu.array() * H.array() - (gamma_ICU + d_ICU.array()) * ICU_.array();
        Eigen::VectorXd dR = gamma_A * A.array() + gamma_I * I.array() + gamma_H * H.array() + gamma_ICU * ICU_.array();
        Eigen::VectorXd dD = d_H.array() * H.array() + d_ICU.array() * ICU_.array();
    
        Eigen::Map<Eigen::VectorXd>(&derivatives[0*n], n) = dS;
        Eigen::Map<Eigen::VectorXd>(&derivatives[1*n], n) = dE;
        Eigen::Map<Eigen::VectorXd>(&derivatives[2*n], n) = dP;
        Eigen::Map<Eigen::VectorXd>(&derivatives[3*n], n) = dA;
        Eigen::Map<Eigen::VectorXd>(&derivatives[4*n], n) = dI;
        Eigen::Map<Eigen::VectorXd>(&derivatives[5*n], n) = dH;
        Eigen::Map<Eigen::VectorXd>(&derivatives[6*n], n) = dICU;
        Eigen::Map<Eigen::VectorXd>(&derivatives[7*n], n) = dR;
        Eigen::Map<Eigen::VectorXd>(&derivatives[8*n], n) = dD;
    }
    
    void AgeSEPAIHRDModel::applyIntervention(const std::string& name,
            [[maybe_unused]] double time,
            const Eigen::VectorXd& params) {
        std::lock_guard<std::mutex> lock(mutex_);
    
        if (name == "mask_mandate" || name == "transmission_reduction") {
            if (params.size() != 1) THROW_INVALID_PARAM("applyIntervention", name + " requires 1 parameter (transmission_reduction [0,1]).");
            double transmission_reduction = params(0);
            if (transmission_reduction < 0.0 || transmission_reduction > 1.0) THROW_INVALID_PARAM("applyIntervention", "Transmission reduction must be between 0 and 1.");
            beta = baseline_beta * (1.0 - transmission_reduction);
            std::cout << "Applied intervention '" << name << "' reducing beta by " << transmission_reduction*100 << "%"
                      << " (new beta = " << beta << ")" << std::endl;
        }
        else if (name == "symptomatic_isolation") {
            if (params.size() != 1) THROW_INVALID_PARAM("applyIntervention", "Symptomatic isolation requires 1 parameter (isolation_factor for theta [0,1]).");
            double isolation_factor = params(0);
            if (isolation_factor < 0.0 || isolation_factor > 1.0) THROW_INVALID_PARAM("applyIntervention", "Isolation factor must be between 0 and 1.");
            theta = baseline_theta * isolation_factor;
             std::cout << "Applied intervention '" << name << "' scaling theta by " << isolation_factor
                       << " (new theta = " << theta << ")" << std::endl;
        }
        else {
            std::cerr << "[AgeSEPAIHRDModel] Warning: applyIntervention called with unhandled name: " << name << std::endl;
        }
    }
    
    void AgeSEPAIHRDModel::reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        beta = baseline_beta;
        theta = baseline_theta;
        std::cout << "SEPAIHRD model intervention parameters reset to baseline (beta=" << beta << ", theta=" << theta << ")." << std::endl;
    }
    
    int AgeSEPAIHRDModel::getStateSize() const {
        // num_age_classes is effectively constant after construction.
        return 9 * num_age_classes;
    }
    
    std::vector<std::string> AgeSEPAIHRDModel::getStateNames() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::string> names;
        names.reserve(9 * num_age_classes);
        std::vector<std::string> compartments = {"S", "E", "P", "A", "I", "H", "ICU", "R", "D"};
    
        for (const auto& comp : compartments) {
            for (int i = 0; i < num_age_classes; ++i) {
                names.push_back(comp + std::to_string(i));
            }
        }
        return names;
    }
    
    int AgeSEPAIHRDModel::getNumAgeClasses() const {
        return num_age_classes;
    }
    const Eigen::VectorXd& AgeSEPAIHRDModel::getPopulationSizes() const { std::lock_guard<std::mutex> lock(mutex_); return N; }
    const Eigen::MatrixXd& AgeSEPAIHRDModel::getContactMatrix() const { std::lock_guard<std::mutex> lock(mutex_); return M_baseline; }
    double AgeSEPAIHRDModel::getTransmissionRate() const { std::lock_guard<std::mutex> lock(mutex_); return beta; }
    double AgeSEPAIHRDModel::getReducedTransmissibility() const { std::lock_guard<std::mutex> lock(mutex_); return theta; }
    
    double AgeSEPAIHRDModel::getSigma() const { std::lock_guard<std::mutex> lock(mutex_); return sigma; }
    double AgeSEPAIHRDModel::getGammaP() const { std::lock_guard<std::mutex> lock(mutex_); return gamma_p; }
    double AgeSEPAIHRDModel::getGammaA() const { std::lock_guard<std::mutex> lock(mutex_); return gamma_A; }
    double AgeSEPAIHRDModel::getGammaI() const { std::lock_guard<std::mutex> lock(mutex_); return gamma_I; }
    double AgeSEPAIHRDModel::getGammaH() const { std::lock_guard<std::mutex> lock(mutex_); return gamma_H; }
    double AgeSEPAIHRDModel::getGammaICU() const { std::lock_guard<std::mutex> lock(mutex_); return gamma_ICU; }
    const Eigen::VectorXd& AgeSEPAIHRDModel::getProbAsymptomatic() const { std::lock_guard<std::mutex> lock(mutex_); return p; }
    const Eigen::VectorXd& AgeSEPAIHRDModel::getHospRate() const { std::lock_guard<std::mutex> lock(mutex_); return h; }
    const Eigen::VectorXd& AgeSEPAIHRDModel::getIcuRate() const { std::lock_guard<std::mutex> lock(mutex_); return icu; }
    const Eigen::VectorXd& AgeSEPAIHRDModel::getMortalityRateH() const { std::lock_guard<std::mutex> lock(mutex_); return d_H; }
    const Eigen::VectorXd& AgeSEPAIHRDModel::getMortalityRateICU() const { std::lock_guard<std::mutex> lock(mutex_); return d_ICU; }
    
    void AgeSEPAIHRDModel::setTransmissionRate(double new_beta) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (new_beta < 0.0) THROW_INVALID_PARAM("setTransmissionRate", "Transmission rate cannot be negative.");
        beta = new_beta;
    }
    
    void AgeSEPAIHRDModel::setReducedTransmissibility(double new_theta) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (new_theta < 0.0) THROW_INVALID_PARAM("setReducedTransmissibility", "Reduced transmissibility factor cannot be negative.");
        theta = new_theta;
    }
    
    std::shared_ptr<INpiStrategy> AgeSEPAIHRDModel::getNpiStrategy() const {
         std::lock_guard<std::mutex> lock(mutex_);
         return npi_strategy;
    }

    SEPAIHRDParameters AgeSEPAIHRDModel::getModelParameters() const {
        std::lock_guard<std::mutex> lock(mutex_);
        SEPAIHRDParameters params;
        params.N = N;
        params.M_baseline = M_baseline;
        params.contact_matrix_scaling_factor = 1.0;
        params.beta = beta;
        params.theta = theta;
        params.sigma = sigma;
        params.gamma_p = gamma_p;
        params.gamma_A = gamma_A;
        params.gamma_I = gamma_I;
        params.gamma_H = gamma_H;
        params.gamma_ICU = gamma_ICU;
        params.p = p;
        params.h = h;
        params.icu = icu;
        params.d_H = d_H;
        params.d_ICU = d_ICU;
        if(this->npi_strategy){
            params.kappa_end_times.clear();
            params.kappa_end_times.push_back(this->npi_strategy->getBaselinePeriodEndTime());
            const std::vector<double>& npi_times_after_baseline = this->npi_strategy->getEndTimes();
            params.kappa_end_times.insert(params.kappa_end_times.end(), npi_times_after_baseline.begin(), npi_times_after_baseline.end());
            params.kappa_values = this->npi_strategy->getValues();
        }
        return params;
    }

    void AgeSEPAIHRDModel::setModelParameters(const SEPAIHRDParameters& params) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (params.N.size() != num_age_classes ||
            params.p.size() != num_age_classes ||
            params.h.size() != num_age_classes ||
            params.icu.size() != num_age_classes ||
            params.d_H.size() != num_age_classes ||
            params.d_ICU.size() != num_age_classes ||
            params.M_baseline.rows() != num_age_classes ||
            params.M_baseline.cols() != num_age_classes) {
            THROW_INVALID_PARAM("setModelParameters", "Parameter dimensions do not match existing number of age classes.");
        }

        if (params.beta < 0 || params.theta < 0 || params.sigma < 0 || params.gamma_p < 0 || params.gamma_A < 0 || params.gamma_I < 0 || params.gamma_H < 0 || params.gamma_ICU < 0) {
             THROW_INVALID_PARAM("setModelParameters", "Rate parameters cannot be negative.");
        }
        if ((params.p.array() < 0).any() || (params.p.array() > 1).any() ||
            (params.h.array() < 0).any() || (params.icu.array() < 0).any() || 
            (params.d_H.array() < 0).any() || (params.d_ICU.array() < 0).any()) {
             THROW_INVALID_PARAM("setModelParameters", "Age-specific rate/probability parameters have invalid values (e.g., negative, or p > 1).");
        }
        if (!params.validate()) {
            THROW_INVALID_PARAM("setModelParameters", "SEPAIHRDParameters object failed its own validation.");
        }


        N = params.N;
        M_baseline = params.M_baseline;
        beta = params.beta;
        theta = params.theta;
        sigma = params.sigma;
        gamma_p = params.gamma_p;
        gamma_A = params.gamma_A;
        gamma_I = params.gamma_I;
        gamma_H = params.gamma_H;
        gamma_ICU = params.gamma_ICU;
        p = params.p;
        h = params.h;
        icu = params.icu;
        d_H = params.d_H;
        d_ICU = params.d_ICU;
        baseline_beta = params.beta;
        baseline_theta = params.theta;
    }

    bool AgeSEPAIHRDModel::areInitialDeathsZero() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return true;
    }
    
}