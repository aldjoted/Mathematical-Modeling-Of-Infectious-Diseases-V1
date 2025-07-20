#include "model/ReproductionNumberCalculator.hpp"
#include "model/parameters/SEPAIHRDParameters.hpp"
#include "exceptions/Exceptions.hpp"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>

namespace epidemic {
ReproductionNumberCalculator::ReproductionNumberCalculator(std::shared_ptr<AgeSEPAIHRDModel> model)
    : model_(std::move(model)),
      num_age_classes_(model_->getNumAgeClasses()),
      num_epi_states_in_ngm_system_(4)
{
    if (!model_) {
        throw std::invalid_argument("Model pointer cannot be null.");
    }
}

Eigen::MatrixXd ReproductionNumberCalculator::buildFMatrixForR0() const {
    SEPAIHRDParameters params = model_->getModelParameters();
    int n = num_age_classes_;
    int total_ngm_states = n * num_epi_states_in_ngm_system_;
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(total_ngm_states, total_ngm_states);

    double kappa_baseline = 1.0;
    if (model_->getNpiStrategy()) {
        kappa_baseline = model_->getNpiStrategy()->getReductionFactor(0.0);
        if (kappa_baseline < 0) {
             throw ModelException("ReproductionNumberCalculator::buildFMatrixForR0", "Baseline NPI reduction factor cannot be negative.");
        }
    }

    // For R0, we use the transmission rate at time t=0
    double beta_0 = model_->computeBeta(0.0);

    // Indices: E_i = i; P_i = n+i; A_i = 2n+i; I_i = 3n+i
    for (int i = 0; i < n; ++i) { // exposed_age_idx (row)
        for (int j = 0; j < n; ++j) { // infector_age_idx (column)
            if (params.N(j) < 1e-9) continue; // Avoid division by zero

            // Rate of new infections in E_i per individual in an infectious state in group j.
            // Formula: beta(0) * kappa(0) * M(i,j) * a(i) * h_infec(j) * (N(i)/N(j))
            double transmission_term = beta_0 * kappa_baseline * params.M_baseline(i, j) *
                                       params.a(i) * params.h_infec(j) * (params.N(i) / params.N(j));

            // New E_i infections from P_j
            F(i, n + j) = transmission_term;
            // New E_i infections from A_j
            F(i, 2 * n + j) = transmission_term;
            // New E_i infections from I_j (with reduced transmissibility theta)
            F(i, 3 * n + j) = params.theta * transmission_term;
        }
    }
    return F;
}


Eigen::MatrixXd ReproductionNumberCalculator::buildFMatrixForRt(const Eigen::VectorXd& S_current, double time) const {
    SEPAIHRDParameters params = model_->getModelParameters();
    int n = num_age_classes_;
    int total_ngm_states = n * num_epi_states_in_ngm_system_;
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(total_ngm_states, total_ngm_states);

    if (S_current.size() != n) {
        throw std::invalid_argument("[ReproductionNumberCalculator::buildFMatrixForRt] S_current vector size mismatch.");
    }
    
    double kappa_t = 1.0;
    if (model_->getNpiStrategy()) {
        kappa_t = model_->getNpiStrategy()->getReductionFactor(time);
        if (kappa_t < 0) {
            throw ModelException("ReproductionNumberCalculator::buildFMatrixForRt", "NPI reduction factor kappa_t cannot be negative.");
        }
    }

    // Use the potentially time-varying beta
    double beta_t = model_->computeBeta(time);

    for (int i = 0; i < n; ++i) { // exposed_age_idx
        for (int j = 0; j < n; ++j) { // infector_age_idx
            if (params.N(j) < 1e-9) continue;

            // Rate of new infections in E_i per individual in an infectious state in group j, at time t.
            // Formula: beta(t) * kappa(t) * M(i,j) * a(i) * h_infec(j) * (S_current(i)/N(j))
            double transmission_term = beta_t * kappa_t * params.M_baseline(i, j) *
                                       params.a(i) * params.h_infec(j) * (S_current(i) / params.N(j));
            
            transmission_term = std::max(0.0, transmission_term);

            // New E_i infections from P_j
            F(i, n + j) = transmission_term;
            // New E_i infections from A_j
            F(i, 2 * n + j) = transmission_term;
            // New E_i infections from I_j (with reduced transmissibility theta)
            F(i, 3 * n + j) = params.theta * transmission_term;
        }
    }
    return F;
}


Eigen::MatrixXd ReproductionNumberCalculator::buildVMatrix() const {
    SEPAIHRDParameters params = model_->getModelParameters();
    int n = num_age_classes_;
    int total_ngm_states = n * num_epi_states_in_ngm_system_;
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(total_ngm_states, total_ngm_states);

    // Order of states in matrix blocks: E, P, A, I
    for (int age = 0; age < n; ++age) {
        int e_idx = age;
        int p_idx = n + age;
        int a_idx = 2 * n + age;
        int i_idx = 3 * n + age;

        // Transitions related to E_age
        // Outflow from E_age at rate sigma
        V(e_idx, e_idx) = params.sigma;
        // Inflow to P_age from E_age
        V(p_idx, e_idx) = -params.sigma;

        // Transitions related to P_age
        // Outflow from P_age at rate gamma_p
        V(p_idx, p_idx) = params.gamma_p;
        // Inflow to A_age from P_age
        V(a_idx, p_idx) = -params.p(age) * params.gamma_p;
        // Inflow to I_age from P_age
        V(i_idx, p_idx) = -(1.0 - params.p(age)) * params.gamma_p;

        // Transitions related to A_age
        // Outflow from A_age (to Recovery) at rate gamma_A
        V(a_idx, a_idx) = params.gamma_A;

        // Transitions related to I_age
        // Outflow from I_age (to Recovery or Hospitalization)
        V(i_idx, i_idx) = params.gamma_I + params.h(age);
    }
    return V;
}


double ReproductionNumberCalculator::calculateR0() {
    Eigen::MatrixXd F = buildFMatrixForR0();
    Eigen::MatrixXd V = buildVMatrix();
    Eigen::MatrixXd V_inv = V.inverse();
    Eigen::MatrixXd K_ngm = F * V_inv;

    Eigen::EigenSolver<Eigen::MatrixXd> es(K_ngm);
    double max_eigenvalue_magnitude = 0.0;
    for (int i = 0; i < es.eigenvalues().size(); ++i) {
        // R0 is the largest REAL eigenvalue for many standard epidemiological models.
        // However, spectral radius (largest magnitude) is the general definition from NGM theory.
        max_eigenvalue_magnitude = std::max(max_eigenvalue_magnitude, std::abs(es.eigenvalues()[i]));
    }
    return max_eigenvalue_magnitude;
}


double ReproductionNumberCalculator::calculateRt(const Eigen::VectorXd& S_current, double time) {
    Eigen::MatrixXd F = buildFMatrixForRt(S_current, time);
    Eigen::MatrixXd V = buildVMatrix();
    Eigen::MatrixXd V_inv = V.inverse();
    Eigen::MatrixXd K_ngm = F * V_inv;

    Eigen::EigenSolver<Eigen::MatrixXd> es(K_ngm);
    double max_eigenvalue_magnitude = 0.0;
    for (int i = 0; i < es.eigenvalues().size(); ++i) {
        max_eigenvalue_magnitude = std::max(max_eigenvalue_magnitude, std::abs(es.eigenvalues()[i]));
    }
    return max_eigenvalue_magnitude;
}

} // namespace epidemic