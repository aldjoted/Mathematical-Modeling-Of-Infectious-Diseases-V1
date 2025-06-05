#include "gtest/gtest.h"
#include "sir_age_structured/AgeSIRModel.hpp"
#include "exceptions/Exceptions.hpp"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <memory>

using namespace epidemic;
using namespace Eigen;

// Test fixture for AgeSIRModel tests
class AgeSIRModelTest : public ::testing::Test {
protected:
    int num_age_classes = 2;
    VectorXd N;
    MatrixXd C;
    VectorXd gamma;
    double q;
    double scale_C;
    std::shared_ptr<AgeSIRModel> model;

    void SetUp() override {
        N.resize(num_age_classes);
        N << 1000.0, 2000.0;

        C.resize(num_age_classes, num_age_classes);
        C << 0.5, 0.1,
             0.1, 0.4;

        gamma.resize(num_age_classes);
        gamma << 0.1, 0.15;

        q = 0.05;
        scale_C = 1.0;

        // Use the factory method for construction
        ASSERT_NO_THROW(model = AgeSIRModel::create(N, C, gamma, q, scale_C));
        ASSERT_NE(model, nullptr);
    }

    // Helper to create a state vector std::vector<double>
    std::vector<double> createStdVectorState(const VectorXd& S, const VectorXd& I, const VectorXd& R) {
        std::vector<double> state_vec;
        state_vec.reserve(S.size() + I.size() + R.size());
        state_vec.insert(state_vec.end(), S.data(), S.data() + S.size());
        state_vec.insert(state_vec.end(), I.data(), I.data() + I.size());
        state_vec.insert(state_vec.end(), R.data(), R.data() + R.size());
        return state_vec;
    }
};

// Test successful model creation
TEST_F(AgeSIRModelTest, CreationSuccess) {
    EXPECT_EQ(model->getNumAgeClasses(), num_age_classes);
    EXPECT_EQ(model->getStateSize(), 3 * num_age_classes);
    EXPECT_EQ(model->getTransmissibility(), q);
    EXPECT_EQ(model->getContactScaleFactor(), scale_C);
    EXPECT_EQ(model->getRecoveryRate(), gamma);
    EXPECT_EQ(model->getPopulationSizes(), N);
    EXPECT_EQ(model->getBaselineContactMatrix(), C);
    EXPECT_TRUE(model->getCurrentContactMatrix().isApprox(C * scale_C, 1e-9)); // Initially C_current = C * scale_C
}

// Test creation failures
TEST(AgeSIRModelCreationTest, CreationFailures) {
    int n = 2;
    VectorXd N(n); N << 1000, 2000;
    MatrixXd C(n, n); C << 0.5, 0.1, 0.1, 0.4;
    VectorXd gamma(n); gamma << 0.1, 0.15;
    double q = 0.05;
    double scale_C = 1.0;

    // Negative population
    VectorXd N_neg(n); N_neg << -1000, 2000;
    EXPECT_THROW(AgeSIRModel::create(N_neg, C, gamma, q, scale_C), ModelConstructionException);

    // Negative gamma
    VectorXd gamma_neg(n); gamma_neg << 0.1, -0.15;
    EXPECT_THROW(AgeSIRModel::create(N, C, gamma_neg, q, scale_C), ModelConstructionException);

    // Negative q
    EXPECT_THROW(AgeSIRModel::create(N, C, gamma, -0.05, scale_C), ModelConstructionException);

    // Negative scale_C
    EXPECT_THROW(AgeSIRModel::create(N, C, gamma, q, -1.0), ModelConstructionException);

    // Negative contact matrix entry
    MatrixXd C_neg(n, n); C_neg << 0.5, -0.1, 0.1, 0.4;
    EXPECT_THROW(AgeSIRModel::create(N, C_neg, gamma, q, scale_C), ModelConstructionException);


    // Dimension mismatch N vs C
    MatrixXd C_wrong_dim(n + 1, n + 1); C_wrong_dim.setZero();
    EXPECT_THROW(AgeSIRModel::create(N, C_wrong_dim, gamma, q, scale_C), ModelConstructionException);

    // Dimension mismatch N vs gamma
    VectorXd gamma_wrong_dim(n + 1); gamma_wrong_dim.setZero();
    EXPECT_THROW(AgeSIRModel::create(N, C, gamma_wrong_dim, q, scale_C), ModelConstructionException);

    // Zero age classes
    VectorXd N_zero(0);
    MatrixXd C_zero(0, 0);
    VectorXd gamma_zero(0);
    EXPECT_THROW(AgeSIRModel::create(N_zero, C_zero, gamma_zero, q, scale_C), ModelConstructionException);
}

// Test derivative calculation
TEST_F(AgeSIRModelTest, ComputeDerivatives) {
    VectorXd S(num_age_classes); S << 900, 1900;
    VectorXd I(num_age_classes); I << 100, 100;
    VectorXd R(num_age_classes); R << 0, 0;
    std::vector<double> state = createStdVectorState(S, I, R);
    std::vector<double> derivatives(state.size());

    ASSERT_NO_THROW((*model)(state, derivatives, 0.0)); // Use operator() which calls computeDerivatives

    // Expected lambda calculation:
    // I_over_N = [100/1000, 100/2000] = [0.1, 0.05]
    // C_current = [[0.5, 0.1], [0.1, 0.4]] * 1.0
    // C_current * I_over_N = [0.5*0.1 + 0.1*0.05, 0.1*0.1 + 0.4*0.05] = [0.055, 0.03]
    // lambda = q * [0.055, 0.03] = 0.05 * [0.055, 0.03] = [0.00275, 0.0015]

    // Expected derivatives:
    // dS = -lambda * S = [-0.00275 * 900, -0.0015 * 1900] = [-2.475, -2.85]
    // dI = lambda * S - gamma * I = [2.475 - 0.1*100, 2.85 - 0.15*100] = [2.475 - 10, 2.85 - 15] = [-7.525, -12.15]
    // dR = gamma * I = [0.1*100, 0.15*100] = [10, 15]

    EXPECT_NEAR(derivatives[0], -2.475, 1e-9); // dS0
    EXPECT_NEAR(derivatives[1], -2.85, 1e-9);  // dS1
    EXPECT_NEAR(derivatives[2], -7.525, 1e-9); // dI0
    EXPECT_NEAR(derivatives[3], -12.15, 1e-9); // dI1
    EXPECT_NEAR(derivatives[4], 10.0, 1e-9);   // dR0
    EXPECT_NEAR(derivatives[5], 15.0, 1e-9);   // dR1
}

// Test derivative calculation with zero susceptibles or infected
TEST_F(AgeSIRModelTest, ComputeDerivativesZeroPop) {
    VectorXd S(num_age_classes); S << 0, 1900; // S0 = 0
    VectorXd I(num_age_classes); I << 100, 0;  // I1 = 0
    VectorXd R(num_age_classes); R << 900, 100;
    std::vector<double> state = createStdVectorState(S, I, R);
    std::vector<double> derivatives(state.size());

    ASSERT_NO_THROW((*model)(state, derivatives, 0.0));

    // Expected lambda calculation:
    // I_over_N = [100/1000, 0/2000] = [0.1, 0.0]
    // C_current = [[0.5, 0.1], [0.1, 0.4]]
    // C_current * I_over_N = [0.5*0.1 + 0.1*0.0, 0.1*0.1 + 0.4*0.0] = [0.05, 0.01]
    // lambda = q * [0.05, 0.01] = 0.05 * [0.05, 0.01] = [0.0025, 0.0005]

    // Expected derivatives:
    // dS = -lambda * S = [-0.0025 * 0, -0.0005 * 1900] = [0, -0.95]
    // dI = lambda * S - gamma * I = [0.0025 * 0 - 0.1*100, 0.0005 * 1900 - 0.15*0] = [-10, 0.95]
    // dR = gamma * I = [0.1*100, 0.15*0] = [10, 0]

    // The model implementation clamps negative derivatives for compartments near zero
    EXPECT_GE(derivatives[0], 0.0); // dS0 should be >= 0 since S0 is 0
    EXPECT_NEAR(derivatives[1], -0.95, 1e-9);  // dS1
    EXPECT_NEAR(derivatives[2], -10.0, 1e-9);  // dI0
    EXPECT_NEAR(derivatives[3], 0.95, 1e-9);   // dI1
    EXPECT_NEAR(derivatives[4], 10.0, 1e-9);   // dR0
    EXPECT_NEAR(derivatives[5], 0.0, 1e-9);    // dR1
}


// Test parameter setters
TEST_F(AgeSIRModelTest, SetParameters) {
    // Set Recovery Rate
    VectorXd new_gamma(num_age_classes); new_gamma << 0.2, 0.25;
    ASSERT_NO_THROW(model->setRecoveryRate(new_gamma));
    EXPECT_EQ(model->getRecoveryRate(), new_gamma);
    VectorXd invalid_gamma(num_age_classes + 1); invalid_gamma.setZero();
    EXPECT_THROW(model->setRecoveryRate(invalid_gamma), InvalidParameterException);
    VectorXd neg_gamma(num_age_classes); neg_gamma << 0.1, -0.1;
    EXPECT_THROW(model->setRecoveryRate(neg_gamma), InvalidParameterException);


    // Set Transmissibility
    double new_q = 0.1;
    ASSERT_NO_THROW(model->setTransmissibility(new_q));
    EXPECT_EQ(model->getTransmissibility(), new_q);
    EXPECT_THROW(model->setTransmissibility(-0.1), InvalidParameterException);

    // Set Contact Scale Factor
    double new_scale_C = 0.5;
    ASSERT_NO_THROW(model->setContactScaleFactor(new_scale_C));
    EXPECT_EQ(model->getContactScaleFactor(), new_scale_C);
    EXPECT_TRUE(model->getCurrentContactMatrix().isApprox(C * new_scale_C, 1e-9));
    EXPECT_THROW(model->setContactScaleFactor(-0.5), InvalidParameterException);
}

// Test interventions
TEST_F(AgeSIRModelTest, ApplyInterventions) {
    double initial_q = model->getTransmissibility();
    double initial_scale_C = model->getContactScaleFactor();
    MatrixXd initial_C_current = model->getCurrentContactMatrix();

    // Contact Reduction
    VectorXd params_contact(1); params_contact << 0.7; // Scale current contacts by 0.7
    ASSERT_NO_THROW(model->applyIntervention("contact_reduction", 10.0, params_contact));
    EXPECT_NEAR(model->getContactScaleFactor(), initial_scale_C * 0.7, 1e-9);
    EXPECT_TRUE(model->getCurrentContactMatrix().isApprox(initial_C_current * 0.7, 1e-9));
    EXPECT_EQ(model->getTransmissibility(), initial_q); // q should be unchanged

    // Transmission Reduction (Mask Mandate)
    VectorXd params_trans(1); params_trans << 0.4; // Reduce transmission by 40% (reduction factor 0.4)
    double expected_q = model->getTransmissibility() * (1.0 - 0.4);
    double scale_C_before_mask = model->getContactScaleFactor();
    ASSERT_NO_THROW(model->applyIntervention("mask_mandate", 20.0, params_trans));
    EXPECT_NEAR(model->getTransmissibility(), expected_q, 1e-9);
    EXPECT_EQ(model->getContactScaleFactor(), scale_C_before_mask); // scale_C should be unchanged by this intervention

    // Test invalid intervention parameters
    VectorXd invalid_params_size(2); invalid_params_size << 0.5, 0.1;
    EXPECT_THROW(model->applyIntervention("contact_reduction", 30.0, invalid_params_size), InterventionException);

    VectorXd invalid_params_range_contact(1); invalid_params_range_contact << -0.1;
    EXPECT_THROW(model->applyIntervention("lockdown", 30.0, invalid_params_range_contact), InterventionException);

    VectorXd invalid_params_range_trans_neg(1); invalid_params_range_trans_neg << -0.1; // < 0
    EXPECT_THROW(model->applyIntervention("transmission_reduction", 30.0, invalid_params_range_trans_neg), InterventionException);
    VectorXd invalid_params_range_trans_high(1); invalid_params_range_trans_high << 1.5; // > 1
    EXPECT_THROW(model->applyIntervention("transmission_reduction", 30.0, invalid_params_range_trans_high), InterventionException);


    // Test unknown intervention
    VectorXd params_unknown(1); params_unknown << 0.5;
    EXPECT_THROW(model->applyIntervention("unknown_intervention", 40.0, params_unknown), InterventionException);
}

// Test reset functionality
TEST_F(AgeSIRModelTest, ResetModel) {
    double baseline_q = model->getTransmissibility();
    double baseline_scale_C = model->getContactScaleFactor();
    MatrixXd baseline_C_current = model->getCurrentContactMatrix();

    // Modify parameters via intervention
    VectorXd params_contact(1); params_contact << 0.7;
    VectorXd params_trans(1); params_trans << 0.4;
    model->applyIntervention("contact_reduction", 10.0, params_contact);
    model->applyIntervention("mask_mandate", 20.0, params_trans);
    EXPECT_NE(model->getTransmissibility(), baseline_q);
    EXPECT_NE(model->getContactScaleFactor(), baseline_scale_C);

    // Reset
    ASSERT_NO_THROW(model->reset());

    // Verify parameters are back to baseline
    EXPECT_EQ(model->getTransmissibility(), baseline_q);
    EXPECT_EQ(model->getContactScaleFactor(), baseline_scale_C);
    EXPECT_TRUE(model->getCurrentContactMatrix().isApprox(baseline_C_current, 1e-9));
}

// Test state names
TEST_F(AgeSIRModelTest, GetStateNames) {
    std::vector<std::string> expected_names = {"S0", "S1", "I0", "I1", "R0", "R1"};
    EXPECT_EQ(model->getStateNames(), expected_names);
}
