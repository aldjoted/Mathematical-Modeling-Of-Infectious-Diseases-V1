#include "gtest/gtest.h"
#include "sir_age_structured/ModelFactory.hpp"
#include "sir_age_structured/AgeSIRModel.hpp" // Include necessary model header
#include "exceptions/Exceptions.hpp"
#include <Eigen/Dense>
#include <vector>
#include <memory>

using namespace epidemic;
using namespace Eigen;

// Test fixture for ModelFactory tests
class ModelFactoryTest : public ::testing::Test {
protected:
    int num_age_classes = 2;
    VectorXd N;
    MatrixXd C;
    VectorXd gamma;
    double q;
    double scale_C;

    VectorXd S0;
    VectorXd I0;
    VectorXd R0;


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

        S0.resize(num_age_classes); S0 << 990, 1980;
        I0.resize(num_age_classes); I0 << 10, 20;
        R0.resize(num_age_classes); R0 << 0, 0;
    }
};

// Test successful AgeSIRModel creation via factory
TEST_F(ModelFactoryTest, CreateAgeSIRModelSuccess) {
    std::shared_ptr<AgeSIRModel> model;
    ASSERT_NO_THROW(model = ModelFactory::createAgeSIRModel(N, C, gamma, q, scale_C));
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->getNumAgeClasses(), num_age_classes);
    EXPECT_EQ(model->getTransmissibility(), q);
    EXPECT_EQ(model->getContactScaleFactor(), scale_C);
    EXPECT_EQ(model->getRecoveryRate(), gamma);
    EXPECT_EQ(model->getPopulationSizes(), N);
}

// Test AgeSIRModel creation failures via factory (delegated validation)
TEST_F(ModelFactoryTest, CreateAgeSIRModelFailures) {
    // Use invalid parameters similar to AgeSIRModelCreationTest
    VectorXd N_neg(num_age_classes); N_neg << -1000, 2000;
    EXPECT_THROW(ModelFactory::createAgeSIRModel(N_neg, C, gamma, q, scale_C), ModelConstructionException);

    MatrixXd C_wrong_dim(num_age_classes + 1, num_age_classes + 1); C_wrong_dim.setZero();
    EXPECT_THROW(ModelFactory::createAgeSIRModel(N, C_wrong_dim, gamma, q, scale_C), ModelConstructionException);

    EXPECT_THROW(ModelFactory::createAgeSIRModel(N, C, gamma, -0.05, scale_C), ModelConstructionException);

    // Test factory wrapping of standard exceptions
    // (Difficult to trigger std::exception directly without modifying AgeSIRModel::create internals,
    // but the factory aims to catch them)
}

// Test successful initial SIR state creation
TEST_F(ModelFactoryTest, CreateInitialSIRStateSuccess) {
    VectorXd initial_state;
    ASSERT_NO_THROW(initial_state = ModelFactory::createInitialSIRState(S0, I0, R0));

    ASSERT_EQ(initial_state.size(), 3 * num_age_classes);
    // Check segments
    EXPECT_EQ(initial_state.segment(0, num_age_classes), S0);
    EXPECT_EQ(initial_state.segment(num_age_classes, num_age_classes), I0);
    EXPECT_EQ(initial_state.segment(2 * num_age_classes, num_age_classes), R0);
}

// Test initial SIR state creation failures
TEST_F(ModelFactoryTest, CreateInitialSIRStateFailures) {
    // Dimension mismatch
    VectorXd I0_wrong_dim(num_age_classes + 1); I0_wrong_dim.setZero();
    EXPECT_THROW(ModelFactory::createInitialSIRState(S0, I0_wrong_dim, R0), InvalidParameterException);

    VectorXd R0_wrong_dim(num_age_classes - 1); R0_wrong_dim.setZero();
    // Ensure the size is non-negative before testing
    if (num_age_classes > 1) {
        EXPECT_THROW(ModelFactory::createInitialSIRState(S0, I0, R0_wrong_dim), InvalidParameterException);
    }


    // Negative values
    VectorXd S0_neg(num_age_classes); S0_neg << -10, 1980;
    EXPECT_THROW(ModelFactory::createInitialSIRState(S0_neg, I0, R0), InvalidParameterException);

    VectorXd I0_neg(num_age_classes); I0_neg << 10, -20;
    EXPECT_THROW(ModelFactory::createInitialSIRState(S0, I0_neg, R0), InvalidParameterException);

    VectorXd R0_neg(num_age_classes); R0_neg << 0, -1;
    EXPECT_THROW(ModelFactory::createInitialSIRState(S0, I0, R0_neg), InvalidParameterException);


    // Empty vectors
    VectorXd S0_empty(0);
    VectorXd I0_empty(0);
    VectorXd R0_empty(0);
    EXPECT_THROW(ModelFactory::createInitialSIRState(S0_empty, I0_empty, R0_empty), InvalidParameterException);
}

// Add tests for createInitialSEPAIHRDState if that model is used and defined
// TEST_F(ModelFactoryTest, CreateInitialSEPAIHRDStateSuccess) { ... }
// TEST_F(ModelFactoryTest, CreateInitialSEPAIHRDStateFailures) { ... }
