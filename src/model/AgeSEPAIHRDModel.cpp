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
#include "model/PieceWiseConstantNPIStrategy.hpp"

namespace epidemic {

    AgeSEPAIHRDModel::AgeSEPAIHRDModel(const SEPAIHRDParameters& params, std::shared_ptr<INpiStrategy> npi_strategy_ptr)
        : num_age_classes(params.N.size()), N(params.N), M_baseline(params.M_baseline),
          beta(params.beta), beta_end_times_(params.beta_end_times), beta_values_(params.beta_values), a(params.a), h_infec(params.h_infec), theta(params.theta),
          sigma(params.sigma), gamma_p(params.gamma_p), gamma_A(params.gamma_A), gamma_I(params.gamma_I),
          gamma_H(params.gamma_H), gamma_ICU(params.gamma_ICU), p(params.p), h(params.h), icu(params.icu),
          d_H(params.d_H), d_ICU(params.d_ICU),
          npi_strategy(npi_strategy_ptr), baseline_beta(params.beta), baseline_theta(params.theta), E0_multiplier(params.E0_multiplier), P0_multiplier(params.P0_multiplier), A0_multiplier(params.A0_multiplier), I0_multiplier(params.I0_multiplier), H0_multiplier(params.H0_multiplier), ICU0_multiplier(params.ICU0_multiplier), R0_multiplier(params.R0_multiplier), D0_multiplier(params.D0_multiplier) {
    
        if (!params.validate()) {
            THROW_INVALID_PARAM("AgeSEPAIHRDModel::constructor", "Invalid SEPAIHRD parameters dimensions or sizes.");
        }
        if (!npi_strategy) {
             THROW_INVALID_PARAM("AgeSEPAIHRDModel::constructor", "NPI strategy pointer cannot be null.");
        }
        if (beta < 0 || theta < 0 || sigma < 0 || gamma_p < 0 || gamma_A < 0 || gamma_I < 0 || gamma_H < 0 || gamma_ICU < 0) {
             THROW_INVALID_PARAM("AgeSEPAIHRDModel:::constructor", "Rate parameters cannot be negative.");
        }
        if ((p.array() < 0).any() || (p.array() > 1).any() ||
            (h.array() < 0).any() || (icu.array() < 0).any() ||
            (d_H.array() < 0).any() || (d_ICU.array() < 0).any()) {
             THROW_INVALID_PARAM("AgeSEPAIHRDModel::constructor", "Age-specific rate/probability parameters cannot be negative (p must be <= 1).");
        }

        if (!beta_values_.empty()) {
            if (beta_values_.size() != beta_end_times_.size()) {
                THROW_INVALID_PARAM("AgeSEPAIHRDModel::ctor", "Beta values and end times must have the same size.");
            }

            double beta_baseline_end_time = 0.0;
            auto* piecewise_npi_strategy = dynamic_cast<PiecewiseConstantNpiStrategy*>(npi_strategy.get());
            if (piecewise_npi_strategy) {
                if (!beta_end_times_.empty()){
                    beta_baseline_end_time = beta_end_times_.front();
                }
            }

            beta_strategy_ = std::make_unique<PiecewiseConstantParameterStrategy>(
                "beta",
                std::vector<double>(beta_end_times_.begin() + 1, beta_end_times_.end()),
                std::vector<double>(beta_values_.begin() + 1, beta_values_.end()),
                beta_values_.front(),
                beta_baseline_end_time
            );
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
        
        // 1. Calculate the total infectious pressure exerted BY each age group, scaled by their infectiousness `h_infec`.
        Eigen::VectorXd infectious_pressure = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd infectious_total = P.array() + A.array() + theta * I.array();

        for (int j = 0; j < n; ++j) {
            if (N(j) > constants::MIN_POPULATION_FOR_DIVISION) {
                infectious_pressure(j) = h_infec(j) * infectious_total(j) / N(j);
            }
        }    
        // 2. Get the NPI-adjusted contact rates.
        double current_beta = computeBeta(time);
        Eigen::MatrixXd effective_contact_matrix = current_reduction_factor * M_baseline;
        // 3. Calculate the force of infection (lambda) for each age group.
        Eigen::VectorXd lambda = current_beta * a.array() * (effective_contact_matrix * infectious_pressure).array();
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
            // This intervention overrides any time-varying beta schedule.
            beta_strategy_.reset(); 
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
        beta_strategy_.reset(); // Also reset the beta strategy
        std::cout << "SEPAIHRD model intervention parameters reset to baseline (beta=" << beta << ", theta=" << theta << ")." << std::endl;
    }
    
    int AgeSEPAIHRDModel::getStateSize() const {
        return 9 * num_age_classes;
    }
    
    std::vector<std::string> AgeSEPAIHRDModel::getStateNames() const {        
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
    const Eigen::VectorXd& AgeSEPAIHRDModel::getPopulationSizes() const { return N; }
    const Eigen::MatrixXd& AgeSEPAIHRDModel::getContactMatrix() const { return M_baseline; }
    double AgeSEPAIHRDModel::getTransmissionRate() const { return beta; }
    const Eigen::VectorXd& AgeSEPAIHRDModel::getSusceptibility() const { return a; }
    const Eigen::VectorXd& AgeSEPAIHRDModel::getInfectiousness() const { return h_infec; }
    double AgeSEPAIHRDModel::getReducedTransmissibility() const { return theta; }
    
    double AgeSEPAIHRDModel::getSigma() const { return sigma; }
    double AgeSEPAIHRDModel::getGammaP() const { return gamma_p; }
    double AgeSEPAIHRDModel::getGammaA() const { return gamma_A; }
    double AgeSEPAIHRDModel::getGammaI() const { return gamma_I; }
    double AgeSEPAIHRDModel::getGammaH() const { return gamma_H; }
    double AgeSEPAIHRDModel::getGammaICU() const { return gamma_ICU; }
    const Eigen::VectorXd& AgeSEPAIHRDModel::getProbAsymptomatic() const { return p; }
    const Eigen::VectorXd& AgeSEPAIHRDModel::getHospRate() const { return h; }
    const Eigen::VectorXd& AgeSEPAIHRDModel::getIcuRate() const { return icu; }
    const Eigen::VectorXd& AgeSEPAIHRDModel::getMortalityRateH() const { return d_H; }
    const Eigen::VectorXd& AgeSEPAIHRDModel::getMortalityRateICU() const { return d_ICU; }
    
    void AgeSEPAIHRDModel::setTransmissionRate(double new_beta) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (new_beta < 0.0) THROW_INVALID_PARAM("setTransmissionRate", "Transmission rate cannot be negative.");
        beta = new_beta;
        beta_strategy_.reset();
    }

    void AgeSEPAIHRDModel::setSusceptibility(const Eigen::VectorXd& new_a) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (new_a.size() != num_age_classes) THROW_INVALID_PARAM("setSusceptibility", "Susceptibility vector dimension mismatch.");
        if ((new_a.array() < 0).any()) THROW_INVALID_PARAM("setSusceptibility", "Susceptibility values cannot be negative.");
        a = new_a;
    }
    void AgeSEPAIHRDModel::setInfectiousness(const Eigen::VectorXd& new_h_infec) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (new_h_infec.size() != num_age_classes) THROW_INVALID_PARAM("setInfectiousness", "Infectiousness vector dimension mismatch.");
        if ((new_h_infec.array() < 0).any()) THROW_INVALID_PARAM("setInfectiousness", "Infectiousness values cannot be negative.");
        h_infec = new_h_infec;
    }

    void AgeSEPAIHRDModel::setReducedTransmissibility(double new_theta) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (new_theta < 0.0) THROW_INVALID_PARAM("setReducedTransmissibility", "Reduced transmissibility factor cannot be negative.");
        theta = new_theta;
    }
    
    std::shared_ptr<INpiStrategy> AgeSEPAIHRDModel::getNpiStrategy() const {
         return npi_strategy;
    }

    SEPAIHRDParameters AgeSEPAIHRDModel::getModelParameters() const {
        std::lock_guard<std::mutex> lock(mutex_);
        SEPAIHRDParameters params;
        params.N = N;
        params.M_baseline = M_baseline;
        params.contact_matrix_scaling_factor = 1.0;
        params.a = a;
        params.h_infec = h_infec;
        params.beta = beta;
        params.beta_end_times = beta_end_times_;
        params.beta_values = beta_values_;
        params.theta = theta;
        params.sigma = sigma;
        params.gamma_p = gamma_p;
        params.gamma_A = gamma_A;
        params.gamma_I = gamma_I;
        params.gamma_H = gamma_H;
        params.gamma_ICU = gamma_ICU;
        params.E0_multiplier = E0_multiplier;
        params.P0_multiplier = P0_multiplier;
        params.A0_multiplier = A0_multiplier;
        params.I0_multiplier = I0_multiplier;
        params.H0_multiplier = H0_multiplier;
        params.ICU0_multiplier = ICU0_multiplier;
        params.R0_multiplier = R0_multiplier;
        params.D0_multiplier = D0_multiplier;
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
            params.a.size() != num_age_classes ||
            params.h_infec.size() != num_age_classes ||
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
        a = params.a;
        h_infec = params.h_infec;
        beta = params.beta;
        theta = params.theta;
        sigma = params.sigma;
        gamma_p = params.gamma_p;
        gamma_A = params.gamma_A;
        gamma_I = params.gamma_I;
        gamma_H = params.gamma_H;
        gamma_ICU = params.gamma_ICU;
        E0_multiplier = params.E0_multiplier;
        P0_multiplier = params.P0_multiplier;
        A0_multiplier = params.A0_multiplier;
        I0_multiplier = params.I0_multiplier;
        H0_multiplier = params.H0_multiplier;
        ICU0_multiplier = params.ICU0_multiplier;
        R0_multiplier = params.R0_multiplier;
        D0_multiplier = params.D0_multiplier;
        p = params.p;
        h = params.h;
        icu = params.icu;
        d_H = params.d_H;
        d_ICU = params.d_ICU;
        
        // Update baselines for reset()
        baseline_beta = params.beta;
        baseline_theta = params.theta;
        beta_end_times_ = params.beta_end_times;
        beta_values_ = params.beta_values;

        if (!beta_values_.empty()) {
            if (beta_values_.size() != beta_end_times_.size()) {
                THROW_INVALID_PARAM("setModelParameters", "beta_end_times and beta_values must have the same size.");
            }
             if (beta_values_.empty()){
                beta_strategy_.reset();
            } else {
                double beta_baseline_end_time = beta_end_times_.front();
                double beta_baseline_value = beta_values_.front();
                std::vector<double> subsequent_end_times(beta_end_times_.begin() + 1, beta_end_times_.end());
                std::vector<double> subsequent_values(beta_values_.begin() + 1, beta_values_.end());
                beta_strategy_ = std::make_unique<PiecewiseConstantParameterStrategy>(
                    "beta",
                    subsequent_end_times,
                    subsequent_values,
                    beta_baseline_value,
                    beta_baseline_end_time
                );
            }
        } else {
            beta_strategy_.reset();
        }
    }

    bool AgeSEPAIHRDModel::areInitialDeathsZero() const {
        return true;
    }

    double AgeSEPAIHRDModel::computeBeta(double time) const {
        if (beta_strategy_) {
            return beta_strategy_->getValue(time);
        }
        return beta;
    }
    
}