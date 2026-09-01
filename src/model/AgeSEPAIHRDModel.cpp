#include "model/AgeSEPAIHRDModel.hpp"
#include "exceptions/Exceptions.hpp"
#include "model/ModelConstants.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include "model/PieceWiseConstantNPIStrategy.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(_MSC_VER)
    #define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define RESTRICT __restrict__
#else
    #define RESTRICT
#endif

namespace epidemic {

    AgeSEPAIHRDModel::AgeSEPAIHRDModel(const SEPAIHRDParameters& params, std::shared_ptr<INpiStrategy> npi_strategy_ptr)
        : num_age_classes(static_cast<int>(params.N.size())), 
          N(params.N), 
          M_baseline(params.M_baseline),
          beta(params.beta), beta_end_times_(params.beta_end_times), beta_values_(params.beta_values), 
          a(params.a), h_infec(params.h_infec), theta(params.theta),
          sigma(params.sigma), gamma_p(params.gamma_p), gamma_A(params.gamma_A), gamma_I(params.gamma_I),
          gamma_H(params.gamma_H), gamma_ICU(params.gamma_ICU), 
          p(params.p), h(params.h), icu(params.icu),
          d_H(params.d_H), d_ICU(params.d_ICU), 
          d_community(params.d_community.size() > 0 ? params.d_community : Eigen::VectorXd::Zero(params.N.size())),
          npi_strategy(std::move(npi_strategy_ptr)), 
          baseline_beta(params.beta), baseline_theta(params.theta),
          E0_multiplier(params.E0_multiplier), P0_multiplier(params.P0_multiplier),
          A0_multiplier(params.A0_multiplier), I0_multiplier(params.I0_multiplier),
          H0_multiplier(params.H0_multiplier), ICU0_multiplier(params.ICU0_multiplier),
          R0_multiplier(params.R0_multiplier), D0_multiplier(params.D0_multiplier),
          runup_days(params.runup_days), seed_exposed(params.seed_exposed) 
    {
        if (!params.validate()) THROW_INVALID_PARAM("AgeSEPAIHRDModel::constructor", "Invalid SEPAIHRD parameters.");
        if (!npi_strategy) THROW_INVALID_PARAM("AgeSEPAIHRDModel::constructor", "NPI strategy pointer cannot be null.");

        // Precompute 1/N to replace division with multiplication in hot path
        inv_N.resize(num_age_classes);
        for(int i = 0; i < num_age_classes; ++i) {
             inv_N[i] = (N[i] > constants::MIN_POPULATION_FOR_DIVISION) ? (1.0 / N[i]) : 0.0;
        }

        if (!beta_end_times_.empty() || !beta_values_.empty()) {
            if (beta_end_times_.size() != beta_values_.size() || beta_end_times_.empty()) {
                THROW_INVALID_PARAM("AgeSEPAIHRDModel::constructor",
                                    "beta_end_times and beta_values must be provided with matching non-zero sizes when using a beta schedule.");
            }
            beta_strategy_ = std::make_unique<PiecewiseConstantParameterStrategy>(
                "beta",
                std::vector<double>(beta_end_times_.begin() + 1, beta_end_times_.end()),
                std::vector<double>(beta_values_.begin() + 1, beta_values_.end()),
                beta_values_.front(),
                beta_end_times_.front()
            );
        }
        resizeWorkingVectors();
    }
    
    AgeSEPAIHRDModel::AgeSEPAIHRDModel(const AgeSEPAIHRDModel& other)
        : EpidemicModel(other),
          num_age_classes(other.num_age_classes), N(other.N), inv_N(other.inv_N), M_baseline(other.M_baseline),
          beta(other.beta), beta_end_times_(other.beta_end_times_), beta_values_(other.beta_values_),
          a(other.a), h_infec(other.h_infec), theta(other.theta), sigma(other.sigma),
          gamma_p(other.gamma_p), gamma_A(other.gamma_A), gamma_I(other.gamma_I),
          gamma_H(other.gamma_H), gamma_ICU(other.gamma_ICU), p(other.p), h(other.h),
          icu(other.icu), d_H(other.d_H), d_ICU(other.d_ICU), d_community(other.d_community),
          baseline_beta(other.baseline_beta), baseline_theta(other.baseline_theta),
          E0_multiplier(other.E0_multiplier), P0_multiplier(other.P0_multiplier),
          A0_multiplier(other.A0_multiplier), I0_multiplier(other.I0_multiplier),
          H0_multiplier(other.H0_multiplier), ICU0_multiplier(other.ICU0_multiplier),
          R0_multiplier(other.R0_multiplier), D0_multiplier(other.D0_multiplier),
          runup_days(other.runup_days), seed_exposed(other.seed_exposed) 
    {
        if (other.npi_strategy) this->npi_strategy = other.npi_strategy->clone();
        if (other.beta_strategy_) this->beta_strategy_ = std::make_unique<PiecewiseConstantParameterStrategy>(*other.beta_strategy_);
        resizeWorkingVectors();
    }

    std::shared_ptr<AgeSEPAIHRDModel> AgeSEPAIHRDModel::clone() const {
        return std::make_shared<AgeSEPAIHRDModel>(*this);
    }

    void AgeSEPAIHRDModel::resizeWorkingVectors() {
        // Single contiguous block improves cache locality
        size_t needed_size = static_cast<size_t>(2 * num_age_classes);
        if (workspace_.size() < needed_size) {
            workspace_.resize(needed_size);
        }
        cached_inf_pressure_ptr_ = workspace_.data();
        cached_lambda_ptr_ = workspace_.data() + num_age_classes;
    }

    void AgeSEPAIHRDModel::computeDerivatives(const std::vector<double>& state,
                                             std::vector<double>& derivatives,
                                             double time) {
        const int n = num_age_classes;
        const int expected_size = constants::NUM_COMPARTMENTS_SEPAIHRD * n;
        
        if (static_cast<int>(state.size()) < expected_size) {
            THROW_INVALID_PARAM("computeDerivatives", "State vector too small.");
        }
        
        if (static_cast<int>(derivatives.size()) != expected_size) {
            derivatives.resize(expected_size);
        }

        // Raw pointers with RESTRICT for SIMD optimization (SoA layout)
        const double* RESTRICT S_ptr   = &state[0 * n];
        const double* RESTRICT E_ptr   = &state[1 * n];
        const double* RESTRICT P_ptr   = &state[2 * n];
        const double* RESTRICT A_ptr   = &state[3 * n];
        const double* RESTRICT I_ptr   = &state[4 * n];
        const double* RESTRICT H_ptr   = &state[5 * n];
        const double* RESTRICT ICU_ptr = &state[6 * n];

        double* RESTRICT dS_ptr   = &derivatives[0 * n];
        double* RESTRICT dE_ptr   = &derivatives[1 * n];
        double* RESTRICT dP_ptr   = &derivatives[2 * n];
        double* RESTRICT dA_ptr   = &derivatives[3 * n];
        double* RESTRICT dI_ptr   = &derivatives[4 * n];
        double* RESTRICT dH_ptr   = &derivatives[5 * n];
        double* RESTRICT dICU_ptr = &derivatives[6 * n];
        double* RESTRICT dR_ptr   = &derivatives[7 * n];
        double* RESTRICT dD_ptr   = &derivatives[8 * n];
        double* RESTRICT dCumH_ptr   = &derivatives[9 * n];
        double* RESTRICT dCumICU_ptr = &derivatives[10 * n];

        const double* RESTRICT inv_N_ptr = inv_N.data();
        const double* RESTRICT h_infec_ptr = h_infec.data();
        const double* RESTRICT p_ptr = p.data();
        const double* RESTRICT h_ptr = h.data();
        const double* RESTRICT icu_ptr = icu.data();
        const double* RESTRICT dH_ptr_const = d_H.data();
        const double* RESTRICT dICU_ptr_const = d_ICU.data();
        const double* RESTRICT d_comm_ptr = d_community.data();
        const double* RESTRICT a_ptr = a.data();
        const double* RESTRICT M_ptr = M_baseline.data();

        double* RESTRICT inf_pressure_ptr = cached_inf_pressure_ptr_;
        double* RESTRICT lambda_ptr = cached_lambda_ptr_;

        const double local_theta = theta;

        // Infectious pressure: multiply by precomputed 1/N instead of division
        #pragma omp simd
        for (int i = 0; i < n; ++i) {
            double total_inf = P_ptr[i] + A_ptr[i] + local_theta * I_ptr[i];
            inf_pressure_ptr[i] = total_inf * h_infec_ptr[i] * inv_N_ptr[i];
        }

        // Force of infection: lambda = Beta * a .* (M * inf_pressure)
        // Column-major traversal for cache-friendly access
        #pragma omp simd
        for (int i = 0; i < n; ++i) {
            lambda_ptr[i] = 0.0;
        }

        for (int j = 0; j < n; ++j) {
            const double inf_j = inf_pressure_ptr[j];
            const double* RESTRICT M_col_ptr = &M_ptr[j * n];
            
            #pragma omp simd
            for (int i = 0; i < n; ++i) {
                lambda_ptr[i] += M_col_ptr[i] * inf_j;
            }
        }

        double current_beta = computeBeta(time);
        double reduction_factor = npi_strategy->getReductionFactor(time); 
        double beta_eff = current_beta * reduction_factor;

        #pragma omp simd
        for (int i = 0; i < n; ++i) {
            lambda_ptr[i] *= beta_eff * a_ptr[i];
        }

        // Local copies for register allocation
        const double local_sigma = sigma;
        const double local_gamma_p = gamma_p;
        const double local_gamma_A = gamma_A;
        const double local_gamma_I = gamma_I;
        const double local_gamma_H = gamma_H;
        const double local_gamma_ICU = gamma_ICU;

        // Compute all compartment derivatives in a single vectorized loop
        #pragma omp simd
        for (int i = 0; i < n; ++i) {
            double lambda_val = std::max(0.0, lambda_ptr[i]);
            
            double flow_SE = lambda_val * S_ptr[i];
            double flow_EP = local_sigma * E_ptr[i];
            double flow_P_out = local_gamma_p * P_ptr[i];
            
            double flow_PA = p_ptr[i] * flow_P_out;
            double flow_PI = flow_P_out - flow_PA;
            
            double flow_IH = h_ptr[i] * I_ptr[i];
            double flow_IR = local_gamma_I * I_ptr[i];
            double flow_ID_community = d_comm_ptr[i] * I_ptr[i];
            double I_out = flow_IR + flow_IH + flow_ID_community;

            double flow_H_ICU = icu_ptr[i] * H_ptr[i];
            double H_out = local_gamma_H * H_ptr[i] + dH_ptr_const[i] * H_ptr[i] + flow_H_ICU;
            double ICU_out = (local_gamma_ICU + dICU_ptr_const[i]) * ICU_ptr[i];

            dS_ptr[i]   = -flow_SE;
            dE_ptr[i]   = flow_SE - flow_EP;
            dP_ptr[i]   = flow_EP - flow_P_out;
            dA_ptr[i]   = flow_PA - local_gamma_A * A_ptr[i];
            dI_ptr[i]   = flow_PI - I_out;
            dH_ptr[i]   = flow_IH - H_out;
            dICU_ptr[i] = flow_H_ICU - ICU_out;
            
            dR_ptr[i]   = local_gamma_A * A_ptr[i] + flow_IR + local_gamma_H * H_ptr[i] + local_gamma_ICU * ICU_ptr[i];
            dD_ptr[i]   = dH_ptr_const[i] * H_ptr[i] + dICU_ptr_const[i] * ICU_ptr[i] + flow_ID_community;
            
            dCumH_ptr[i]   = flow_IH;
            dCumICU_ptr[i] = flow_H_ICU;
        }
    }
    
    void AgeSEPAIHRDModel::applyIntervention(const std::string& name, double time, const Eigen::VectorXd& params) {
        (void)time;
        if (name == "mask_mandate" || name == "transmission_reduction") {
             if (params.size() != 1) THROW_INVALID_PARAM("applyIntervention", name + " requires 1 parameter.");
             double frac = std::clamp(params(0), 0.0, 1.0);
             setTransmissionRate(baseline_beta * (1.0 - frac));
        } else if (name == "symptomatic_isolation") {
             if (params.size() != 1) THROW_INVALID_PARAM("applyIntervention", "Symptomatic isolation requires 1 parameter.");
             double frac = std::clamp(params(0), 0.0, 1.0);
             setReducedTransmissibility(baseline_theta * frac);
        }
    }
    
    void AgeSEPAIHRDModel::reset() {
        beta = baseline_beta;
        theta = baseline_theta;
        beta_strategy_.reset();
    }
    
    int AgeSEPAIHRDModel::getStateSize() const { return constants::NUM_COMPARTMENTS_SEPAIHRD * num_age_classes; }
    
    std::vector<std::string> AgeSEPAIHRDModel::getStateNames() const {        
        std::vector<std::string> names;
        names.reserve(constants::NUM_COMPARTMENTS_SEPAIHRD * num_age_classes);
        static const std::vector<std::string> compartments = {"S", "E", "P", "A", "I", "H", "ICU", "R", "D", "CumH", "CumICU"};
        for (const auto& comp : compartments) {
            for (int i = 0; i < num_age_classes; ++i) names.push_back(comp + std::to_string(i));
        }
        return names;
    }
    
    int AgeSEPAIHRDModel::getNumAgeClasses() const { return num_age_classes; }
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
    const Eigen::VectorXd& AgeSEPAIHRDModel::getCommunityMortalityRate() const { return d_community; }

    void AgeSEPAIHRDModel::setTransmissionRate(double new_beta) {
        if (new_beta < 0.0) THROW_INVALID_PARAM("setTransmissionRate", "Negative beta.");
        beta = new_beta;
        beta_strategy_.reset();
    }
    void AgeSEPAIHRDModel::setReducedTransmissibility(double new_theta) {
        if (new_theta < 0.0) THROW_INVALID_PARAM("setReducedTransmissibility", "Negative theta.");
        theta = new_theta;
    }
    void AgeSEPAIHRDModel::setSusceptibility(const Eigen::VectorXd& new_a) { a = new_a; }
    void AgeSEPAIHRDModel::setInfectiousness(const Eigen::VectorXd& new_h) { h_infec = new_h; }
    std::shared_ptr<INpiStrategy> AgeSEPAIHRDModel::getNpiStrategy() const { return npi_strategy; }

    SEPAIHRDParameters AgeSEPAIHRDModel::getModelParameters() const {
        SEPAIHRDParameters p_struct;
        p_struct.N = N; p_struct.M_baseline = M_baseline; p_struct.contact_matrix_scaling_factor = 1.0;
        p_struct.a = a; p_struct.h_infec = h_infec; p_struct.beta = beta; p_struct.theta = theta;
        p_struct.sigma = sigma; p_struct.gamma_p = gamma_p; p_struct.gamma_A = gamma_A;
        p_struct.gamma_I = gamma_I; p_struct.gamma_H = gamma_H; p_struct.gamma_ICU = gamma_ICU;
        p_struct.p = p; p_struct.h = h; p_struct.icu = icu; p_struct.d_H = d_H; p_struct.d_ICU = d_ICU;
        p_struct.d_community = d_community;
        p_struct.beta_end_times = beta_end_times_; p_struct.beta_values = beta_values_;
        
        p_struct.E0_multiplier = E0_multiplier;
        p_struct.P0_multiplier = P0_multiplier;
        p_struct.A0_multiplier = A0_multiplier;
        p_struct.I0_multiplier = I0_multiplier;
        p_struct.H0_multiplier = H0_multiplier;
        p_struct.ICU0_multiplier = ICU0_multiplier;
        p_struct.R0_multiplier = R0_multiplier;
        p_struct.D0_multiplier = D0_multiplier;
        
        p_struct.runup_days = runup_days;
        p_struct.seed_exposed = seed_exposed;

        if(this->npi_strategy){
            p_struct.kappa_end_times.push_back(this->npi_strategy->getBaselinePeriodEndTime());
            const auto& times = this->npi_strategy->getEndTimes();
            p_struct.kappa_end_times.insert(p_struct.kappa_end_times.end(), times.begin(), times.end());
            p_struct.kappa_values = this->npi_strategy->getValues();
        }
        return p_struct;
    }

    void AgeSEPAIHRDModel::setModelParameters(const SEPAIHRDParameters& params) {
        if (static_cast<int>(params.N.size()) != num_age_classes) 
            THROW_INVALID_PARAM("setModelParameters", "Size mismatch.");
        
        N = params.N; 
        
        // Update precomputed inverses
        for(int i = 0; i < num_age_classes; ++i) {
             inv_N[i] = (N[i] > constants::MIN_POPULATION_FOR_DIVISION) ? (1.0 / N[i]) : 0.0;
        }

        M_baseline = params.M_baseline; a = params.a; h_infec = params.h_infec;
        beta = params.beta; theta = params.theta; sigma = params.sigma;
        gamma_p = params.gamma_p; gamma_A = params.gamma_A; gamma_I = params.gamma_I;
        gamma_H = params.gamma_H; gamma_ICU = params.gamma_ICU;
        p = params.p; h = params.h; icu = params.icu; d_H = params.d_H; d_ICU = params.d_ICU;
        d_community = params.d_community.size() > 0 ? params.d_community : Eigen::VectorXd::Zero(num_age_classes);
        
        E0_multiplier = params.E0_multiplier; P0_multiplier = params.P0_multiplier;
        A0_multiplier = params.A0_multiplier; I0_multiplier = params.I0_multiplier;
        H0_multiplier = params.H0_multiplier; ICU0_multiplier = params.ICU0_multiplier;
        R0_multiplier = params.R0_multiplier; D0_multiplier = params.D0_multiplier;
        runup_days = params.runup_days; seed_exposed = params.seed_exposed;

        baseline_beta = params.beta; baseline_theta = params.theta;
        beta_end_times_ = params.beta_end_times; beta_values_ = params.beta_values;

        if (!beta_values_.empty() && !beta_end_times_.empty() && 
            beta_values_.size() == beta_end_times_.size()) {
             double beta_baseline_end_time = beta_end_times_.front();
             double beta_baseline_value = beta_values_.front();
             beta_strategy_ = std::make_unique<PiecewiseConstantParameterStrategy>(
                "beta", std::vector<double>(beta_end_times_.begin()+1, beta_end_times_.end()),
                std::vector<double>(beta_values_.begin()+1, beta_values_.end()),
                beta_baseline_value, beta_baseline_end_time);
        } else {
            beta_strategy_.reset();
        }
    }

    bool AgeSEPAIHRDModel::areInitialDeathsZero() const { return true; }
    double AgeSEPAIHRDModel::computeBeta(double time) const {
        return beta_strategy_ ? beta_strategy_->getValue(time) : beta;
    }
}