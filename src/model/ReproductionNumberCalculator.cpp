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
      num_epi_states_in_ngm_system_(4) // E, P, A, I states
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

    double kappa_baseline = 1.0; // Default if no NPI strategy
    if (model_->getNpiStrategy()) {
        kappa_baseline = model_->getNpiStrategy()->getReductionFactor(0.0); // Kappa at time 0
        if (kappa_baseline < 0) {
             throw ModelException("ReproductionNumberCalculator::buildFMatrixForR0", "Baseline NPI reduction factor cannot be negative.");
        }
    }

    // Indices: E_a = a; P_a = n+a; A_a = 2n+a; I_a = 3n+a
    for (int exposed_age_idx = 0; exposed_age_idx < n; ++exposed_age_idx) { // Row index in F (for E_a)
        for (int infector_age_idx = 0; infector_age_idx < n; ++infector_age_idx) {
            if (params.N(infector_age_idx) < 1e-9) continue; // Avoid division by zero

            double contact_term = kappa_baseline * params.M_baseline(exposed_age_idx, infector_age_idx) *
                                  (params.N(exposed_age_idx) / params.N(infector_age_idx));

            // New E_a infections from P_b (infector_age_idx)
            F(exposed_age_idx, /*P start*/ n + infector_age_idx) = params.beta * contact_term;
            // New E_a infections from A_b
            F(exposed_age_idx, /*A start*/ 2 * n + infector_age_idx) = params.beta * contact_term;
            // New E_a infections from I_b
            F(exposed_age_idx, /*I start*/ 3 * n + infector_age_idx) = params.beta * params.theta * contact_term;
        }
    }
    return F;
}


Eigen::MatrixXd ReproductionNumberCalculator::buildFMatrixForRt(const Eigen::VectorXd& S_current, double kappa_t) const {
    SEPAIHRDParameters params = model_->getModelParameters();
    int n = num_age_classes_;
    int total_ngm_states = n * num_epi_states_in_ngm_system_;
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(total_ngm_states, total_ngm_states);

    if (S_current.size() != n) {
        throw std::invalid_argument("[ReproductionNumberCalculator::buildFMatrixForRt] S_current vector size mismatch.");
    }
    if (kappa_t < 0) {
        throw ModelException("ReproductionNumberCalculator::buildFMatrixForRt", "NPI reduction factor kappa_t cannot be negative.");
    }


    for (int exposed_age_idx = 0; exposed_age_idx < n; ++exposed_age_idx) {
        for (int infector_age_idx = 0; infector_age_idx < n; ++infector_age_idx) {
            if (params.N(infector_age_idx) < 1e-9) continue;

            double contact_term = kappa_t * params.M_baseline(exposed_age_idx, infector_age_idx) *
                                  (S_current(exposed_age_idx) / params.N(infector_age_idx));
            
            contact_term = std::max(0.0, contact_term); // Ensure non-negative due to S_current potentially being very small

            F(exposed_age_idx, n + infector_age_idx) = params.beta * contact_term;
            F(exposed_age_idx, 2 * n + infector_age_idx) = params.beta * contact_term;
            F(exposed_age_idx, 3 * n + infector_age_idx) = params.beta * params.theta * contact_term;
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

        // Transitions OUT OF E_age
        V(e_idx, e_idx) = params.sigma;
        // Transitions INTO P_age FROM E_age
        V(p_idx, e_idx) = -params.sigma;

        // Transitions OUT OF P_age
        V(p_idx, p_idx) = params.gamma_p;
        // Transitions INTO A_age FROM P_age
        V(a_idx, p_idx) = -params.p(age) * params.gamma_p;
        // Transitions INTO I_age FROM P_age
        V(i_idx, p_idx) = -(1.0 - params.p(age)) * params.gamma_p;

        // Transitions OUT OF A_age
        V(a_idx, a_idx) = params.gamma_A;

        // Transitions OUT OF I_age
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
    if (!model_->getNpiStrategy()) {
        throw ModelException("ReproductionNumberCalculator::calculateRt", "NPI strategy not set in model, cannot get kappa(t).");
    }
    double kappa_t = model_->getNpiStrategy()->getReductionFactor(time);

    Eigen::MatrixXd F = buildFMatrixForRt(S_current, kappa_t);
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