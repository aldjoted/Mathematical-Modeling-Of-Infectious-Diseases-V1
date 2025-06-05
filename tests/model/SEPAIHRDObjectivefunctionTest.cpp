#include <gtest/gtest.h>
#include "model/objectives/SEPAIHRDObjectiveFunction.hpp"
#include "model/AgeSEPAIHRDModel.hpp"
#include "model/interfaces/INpiStrategy.hpp"
#include "model/parameters/SEPAIHRDParameterManager.hpp"
#include "model/parameters/SEPAIHRDParameters.hpp"
#include "sir_age_structured/caching/SimulationCache.hpp"
#include "sir_age_structured/solvers/Dropri5SolverStrategy.hpp"
#include "model/PieceWiseConstantNPIStrategy.hpp"
#include "utils/GetCalibrationData.hpp"
#include "utils/Logger.hpp"
#include <memory>
#include <random>

class TestableSEPAIHRDObjectiveFunction : public epidemic::SEPAIHRDObjectiveFunction {
public:
    using epidemic::SEPAIHRDObjectiveFunction::SEPAIHRDObjectiveFunction;
    double publicCalculateSingleLogLikelihood(const Eigen::MatrixXd& simulated,
                                                const Eigen::MatrixXd& observed,
                                                const std::string& label) const {
        return calculateSingleLogLikelihood(simulated, observed, label);
    }
};

namespace epidemic {
namespace test {

/**
 * @brief Test fixture for SEPAIHRDObjectiveFunction
 * 
 * Provides common setup and helper methods for testing the objective function
 */
class SEPAIHRDObjectiveFunctionTest : public ::testing::Test {
protected:
    // Model components
    std::shared_ptr<AgeSEPAIHRDModel> model_;
    std::shared_ptr<INpiStrategy> npi_strategy_;
    std::shared_ptr<IOdeSolverStrategy> solver_strategy_;
    
    // Calibration components
    std::unique_ptr<SEPAIHRDParameterManager> parameter_manager_;
    std::unique_ptr<SimulationCache> cache_;
    std::unique_ptr<CalibrationData> calibration_data_;
    
    // Test parameters
    SEPAIHRDParameters test_params_;
    std::vector<double> time_points_;
    Eigen::VectorXd initial_state_;
    
    // Random number generator for test data
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_dist_;
    std::poisson_distribution<int> poisson_dist_;
    
    // Constants
    static constexpr int NUM_AGE_CLASSES = 4;
    static constexpr int NUM_TIME_POINTS = 30;
    static constexpr double ABS_ERROR = 1e-6;
    static constexpr double REL_ERROR = 1e-6;

    void SetUp() override {
        // Initialize random number generator
        rng_.seed(42); // Fixed seed for reproducibility
        uniform_dist_ = std::uniform_real_distribution<double>(0.0, 1.0);
        poisson_dist_ = std::poisson_distribution<int>(10);
        
        // Set up logger for tests
        Logger::getInstance().setLogLevel(LogLevel::WARNING);
        
        // Create test parameters
        setupTestParameters();
        
        // Create NPI strategy
        std::vector<double> npi_end_times_setup = {13.0, 63.0, 111.0, 305.0};
        std::vector<double> npi_values_setup = {1.0, 0.5, 0.7, 0.9};

        double baseline_kappa = npi_values_setup[0];
        double baseline_end_time = npi_end_times_setup[0];
        std::vector<double> npi_times_after_baseline = {npi_end_times_setup[1], npi_end_times_setup[2], npi_end_times_setup[3]};
        std::vector<double> npi_kappas_after_baseline = {npi_values_setup[1], npi_values_setup[2], npi_values_setup[3]};
        std::vector<std::string> npi_calib_names_for_strategy = {"kappa_1", "kappa_2", "kappa_3"};

        npi_strategy_ = std::make_shared<PiecewiseConstantNpiStrategy>(
            npi_times_after_baseline, 
            npi_kappas_after_baseline, 
            std::map<std::string, std::pair<double, double>>{},
            baseline_kappa,
            baseline_end_time,
            true,
            npi_calib_names_for_strategy
        );
        
        // Create model
        model_ = std::make_shared<AgeSEPAIHRDModel>(test_params_, npi_strategy_);
        
        // Create solver strategy
        solver_strategy_ = std::make_shared<Dopri5SolverStrategy>();
        
        // Create parameter manager
        std::vector<std::string> param_names = {"beta", "theta", "kappa_1", "kappa_2", "kappa_3"};
        std::vector<double> lower_bounds_vec = {0.01, 0.1, 0.1, 0.1, 0.1};
        std::vector<double> upper_bounds_vec = {1.0, 1.0, 1.5, 1.5, 1.5};
        std::vector<double> sigmas_vec = {0.01, 0.01, 0.05, 0.05, 0.05};
        
        std::map<std::string, double> proposal_sigmas_map;
        std::map<std::string, std::pair<double, double>> param_bounds_map;
        for (size_t i = 0; i < param_names.size(); ++i) {
            proposal_sigmas_map[param_names[i]] = sigmas_vec[i];
            param_bounds_map[param_names[i]] = {lower_bounds_vec[i], upper_bounds_vec[i]};
        }
        
        parameter_manager_ = std::make_unique<SEPAIHRDParameterManager>(
            model_, param_names, proposal_sigmas_map, param_bounds_map
        );
        
        // Create cache
        cache_ = std::make_unique<SimulationCache>(1000, 8);
        
        // Set up time points
        setupTimePoints();
        
        // Create synthetic calibration data
        setupSyntheticCalibrationData();
        
        // Set up initial state
        setupInitialState();
    }

    void TearDown() override {
        // Clean up is handled by smart pointers
    }

    /**
     * @brief Set up test parameters for the SEPAIHRD model
     */
    void setupTestParameters() {
        // Population by age group
        test_params_.N = Eigen::VectorXd(NUM_AGE_CLASSES);
        test_params_.N << 3000000, 4000000, 2000000, 1000000;
        
        // Contact matrix
        test_params_.M_baseline = Eigen::MatrixXd(NUM_AGE_CLASSES, NUM_AGE_CLASSES);
        test_params_.M_baseline << 7.0, 5.0, 2.0, 1.0,
                                   5.0, 8.0, 3.0, 1.5,
                                   2.0, 3.0, 4.0, 2.0,
                                   1.0, 1.5, 2.0, 3.0;
        
        test_params_.contact_matrix_scaling_factor = 1.0;
        
        // Transmission parameters
        test_params_.beta = 0.05;
        test_params_.theta = 0.5;
        
        // Disease progression rates
        test_params_.sigma = 1.0 / 3.0;    
        test_params_.gamma_p = 1.0 / 2.0;  
        test_params_.gamma_A = 1.0 / 5.0;   
        test_params_.gamma_I = 1.0 / 5.0;  
        test_params_.gamma_H = 1.0 / 10.0; 
        test_params_.gamma_ICU = 1.0 / 14.0;
        
        // Age-specific parameters
        test_params_.p = Eigen::VectorXd(NUM_AGE_CLASSES);
        test_params_.p << 0.4, 0.3, 0.2, 0.1; // Asymptomatic fraction
        
        test_params_.h = Eigen::VectorXd(NUM_AGE_CLASSES);
        test_params_.h << 0.01, 0.03, 0.08, 0.15; // Hospitalization rate
        
        test_params_.icu = Eigen::VectorXd(NUM_AGE_CLASSES);
        test_params_.icu << 0.05, 0.10, 0.25, 0.40; // ICU rate among hospitalized
        
        test_params_.d_H = Eigen::VectorXd(NUM_AGE_CLASSES);
        test_params_.d_H << 0.01, 0.02, 0.05, 0.10; // Hospital mortality
        
        test_params_.d_ICU = Eigen::VectorXd(NUM_AGE_CLASSES);
        test_params_.d_ICU << 0.20, 0.30, 0.40, 0.50; // ICU mortality
        
        // NPI parameters (will be overridden by strategy)
        test_params_.kappa_end_times = {13.0, 63.0, 111.0, 305.0};
        test_params_.kappa_values = {1.0, 0.5, 0.7, 0.9};
    }

    /**
     * @brief Set up time points for simulation
     */
    void setupTimePoints() {
        time_points_.clear();
        for (int i = 0; i < NUM_TIME_POINTS; ++i) {
            time_points_.push_back(static_cast<double>(i));
        }
    }

    /**
     * @brief Create synthetic calibration data for testing
     */
    void setupSyntheticCalibrationData() {
        // Create synthetic observed data matrices
        Eigen::MatrixXd new_hospitalizations(NUM_TIME_POINTS, NUM_AGE_CLASSES);
        Eigen::MatrixXd new_icu(NUM_TIME_POINTS, NUM_AGE_CLASSES);
        Eigen::MatrixXd new_deaths(NUM_TIME_POINTS, NUM_AGE_CLASSES);
        Eigen::MatrixXd new_cases(NUM_TIME_POINTS, NUM_AGE_CLASSES);
        
        // Generate synthetic data with some realistic patterns
        for (int t = 0; t < NUM_TIME_POINTS; ++t) {
            for (int a = 0; a < NUM_AGE_CLASSES; ++a) {
                // Create an epidemic curve pattern
                double base_rate = 100.0 * std::exp(-0.5 * std::pow((t - 15.0) / 10.0, 2));
                
                // Age-specific scaling
                double age_factor = (a + 1) * 0.5;
                
                // Add Poisson noise
                new_hospitalizations(t, a) = std::max(0, poisson_dist_(rng_) + 
                                                     static_cast<int>(base_rate * age_factor * 0.1));
                new_icu(t, a) = std::max(0, poisson_dist_(rng_) + 
                                        static_cast<int>(base_rate * age_factor * 0.03));
                new_deaths(t, a) = std::max(0, static_cast<int>(base_rate * age_factor * 0.01));
                new_cases(t, a) = std::max(0, poisson_dist_(rng_) + 
                                          static_cast<int>(base_rate * age_factor));
            }
        }
        
        // Create cumulative data (simplified - just cumsum)
        Eigen::MatrixXd cum_cases = new_cases;
        Eigen::MatrixXd cum_hosp = new_hospitalizations;
        Eigen::MatrixXd cum_icu = new_icu;
        Eigen::MatrixXd cum_deaths = new_deaths;
        
        for (int t = 1; t < NUM_TIME_POINTS; ++t) {
            cum_cases.row(t) += cum_cases.row(t-1);
            cum_hosp.row(t) += cum_hosp.row(t-1);
            cum_icu.row(t) += cum_icu.row(t-1);
            cum_deaths.row(t) += cum_deaths.row(t-1);
        }
        
        // Initial cumulative values (at t=0)
        Eigen::VectorXd initial_cum_cases = cum_cases.row(0);
        Eigen::VectorXd initial_cum_deaths = cum_deaths.row(0);
        Eigen::VectorXd initial_cum_hosp = cum_hosp.row(0);
        Eigen::VectorXd initial_cum_icu = cum_icu.row(0);
        
        calibration_data_ = std::make_unique<CalibrationData>(
            new_cases,
            new_hospitalizations,
            new_icu,
            new_deaths,
            test_params_.N,
            initial_cum_cases,
            initial_cum_deaths,
            initial_cum_hosp,
            initial_cum_icu,
            NUM_AGE_CLASSES
        );
    }

    /**
     * @brief Set up initial state for simulations
     */
    void setupInitialState() {
        initial_state_ = calibration_data_->getInitialSEPAIHRDState(
            test_params_.sigma,
            test_params_.gamma_p,
            test_params_.gamma_A,
            test_params_.gamma_I,
            test_params_.p,
            test_params_.h
        );
    }

    /**
     * @brief Create an objective function instance for testing
     */
    std::unique_ptr<SEPAIHRDObjectiveFunction> createObjectiveFunction() {
        return std::make_unique<SEPAIHRDObjectiveFunction>(
            model_,
            *parameter_manager_,
            *cache_,
            *calibration_data_,
            time_points_,
            initial_state_,
            solver_strategy_,
            ABS_ERROR,
            REL_ERROR
        );
    }
};

// Test basic construction and initialization
TEST_F(SEPAIHRDObjectiveFunctionTest, ConstructionTest) {
    ASSERT_NO_THROW({
        auto obj_func = createObjectiveFunction();
        EXPECT_NE(obj_func, nullptr);
    });
}

// Test parameter names retrieval
TEST_F(SEPAIHRDObjectiveFunctionTest, GetParameterNamesTest) {
    auto obj_func = createObjectiveFunction();
    const auto& param_names = obj_func->getParameterNames();
    
    EXPECT_EQ(param_names.size(), 5);
    EXPECT_EQ(param_names[0], "beta");
    EXPECT_EQ(param_names[1], "theta");
    EXPECT_EQ(param_names[2], "kappa_1");
}

// Test basic likelihood calculation
TEST_F(SEPAIHRDObjectiveFunctionTest, BasicCalculationTest) {
    auto obj_func = createObjectiveFunction();
    
    // Get current parameters
    Eigen::VectorXd current_params = parameter_manager_->getCurrentParameters();
    
    // Calculate likelihood
    double likelihood = obj_func->calculate(current_params);
    
    // Check that likelihood is finite and reasonable
    EXPECT_TRUE(std::isfinite(likelihood));
    EXPECT_GT(likelihood, std::numeric_limits<double>::lowest());
}

// Test caching behavior
TEST_F(SEPAIHRDObjectiveFunctionTest, CachingTest) {
    auto obj_func = createObjectiveFunction();
    
    Eigen::VectorXd params = parameter_manager_->getCurrentParameters();
    
    // First calculation - should miss cache
    double likelihood1 = obj_func->calculate(params);
    
    // Second calculation with same parameters - should hit cache
    double likelihood2 = obj_func->calculate(params);
    
    // Results should be identical
    EXPECT_DOUBLE_EQ(likelihood1, likelihood2);
    
    // Cache should have an entry
    EXPECT_GT(cache_->size(), 0);
}

// Test parameter sensitivity
TEST_F(SEPAIHRDObjectiveFunctionTest, ParameterSensitivityTest) {
    auto obj_func = createObjectiveFunction();
    
    Eigen::VectorXd params = parameter_manager_->getCurrentParameters();
    double baseline_likelihood = obj_func->calculate(params);
    
    // Perturb beta
    Eigen::VectorXd perturbed_params = params;
    perturbed_params(0) *= 1.1;
    double perturbed_likelihood = obj_func->calculate(perturbed_params);
    
    // Likelihood should change
    EXPECT_NE(baseline_likelihood, perturbed_likelihood);
}

// Test edge case: zero observed data
TEST_F(SEPAIHRDObjectiveFunctionTest, ZeroObservedDataTest) {
    Eigen::MatrixXd zero_matrix = Eigen::MatrixXd::Zero(NUM_TIME_POINTS, NUM_AGE_CLASSES);
    Eigen::VectorXd zero_vector = Eigen::VectorXd::Zero(NUM_AGE_CLASSES);
    
    auto zero_calibration_data = std::make_unique<CalibrationData>(
        zero_matrix, zero_matrix, zero_matrix, zero_matrix,
        test_params_.N,
        zero_vector, zero_vector, zero_vector, zero_vector,
        NUM_AGE_CLASSES
    );
    
    auto obj_func = std::make_unique<SEPAIHRDObjectiveFunction>(
        model_,
        *parameter_manager_,
        *cache_,
        *zero_calibration_data,
        time_points_,
        initial_state_,
        solver_strategy_,
        ABS_ERROR,
        REL_ERROR
    );
    
    Eigen::VectorXd params = parameter_manager_->getCurrentParameters();
    double likelihood = obj_func->calculate(params);
    
    EXPECT_TRUE(std::isfinite(likelihood));
}

// Test with very small time step
TEST_F(SEPAIHRDObjectiveFunctionTest, SmallTimeStepTest) {
    // Create dense time points
    std::vector<double> dense_time_points;
    for (double t = 0.0; t <= 10.0; t += 0.1) {
        dense_time_points.push_back(t);
    }
    
    // Create corresponding calibration data
    int num_dense_points = static_cast<int>(dense_time_points.size());
    Eigen::MatrixXd dense_data = Eigen::MatrixXd::Random(num_dense_points, NUM_AGE_CLASSES).cwiseAbs() * 10;
    Eigen::VectorXd zero_vector = Eigen::VectorXd::Zero(NUM_AGE_CLASSES);
    
    auto dense_calibration_data = std::make_unique<CalibrationData>(
        dense_data, dense_data, dense_data, dense_data,
        test_params_.N,
        zero_vector, zero_vector, zero_vector, zero_vector,
        NUM_AGE_CLASSES
    );
    
    auto obj_func = std::make_unique<SEPAIHRDObjectiveFunction>(
        model_,
        *parameter_manager_,
        *cache_,
        *dense_calibration_data,
        dense_time_points,
        initial_state_,
        solver_strategy_,
        ABS_ERROR,
        REL_ERROR
    );
    
    Eigen::VectorXd params = parameter_manager_->getCurrentParameters();
    
    ASSERT_NO_THROW({
        double likelihood = obj_func->calculate(params);
        EXPECT_TRUE(std::isfinite(likelihood));
    });
}

// Test with missing data (NaN values)
TEST_F(SEPAIHRDObjectiveFunctionTest, MissingDataTest) {
    // Create data with some NaN values
    Eigen::MatrixXd data_with_nan = Eigen::MatrixXd::Random(NUM_TIME_POINTS, NUM_AGE_CLASSES).cwiseAbs() * 10;
    
    // Insert some NaN values
    data_with_nan(5, 2) = std::numeric_limits<double>::quiet_NaN();
    data_with_nan(10, 0) = std::numeric_limits<double>::quiet_NaN();
    
    Eigen::VectorXd zero_vector = Eigen::VectorXd::Zero(NUM_AGE_CLASSES);
    
    auto nan_calibration_data = std::make_unique<CalibrationData>(
        data_with_nan, data_with_nan, data_with_nan, data_with_nan,
        test_params_.N,
        zero_vector, zero_vector, zero_vector, zero_vector,
        NUM_AGE_CLASSES
    );
    
    auto obj_func = std::make_unique<SEPAIHRDObjectiveFunction>(
        model_,
        *parameter_manager_,
        *cache_,
        *nan_calibration_data,
        time_points_,
        initial_state_,
        solver_strategy_,
        ABS_ERROR,
        REL_ERROR
    );
    
    Eigen::VectorXd params = parameter_manager_->getCurrentParameters();
    
    ASSERT_NO_THROW({
        double likelihood = obj_func->calculate(params);
        EXPECT_TRUE(std::isfinite(likelihood) || likelihood == std::numeric_limits<double>::lowest());
    });
}

// Test parallel calculation consistency
TEST_F(SEPAIHRDObjectiveFunctionTest, ParallelConsistencyTest) {
    auto obj_func = createObjectiveFunction();
    
    Eigen::VectorXd params = parameter_manager_->getCurrentParameters();
    
    // Calculate multiple times to test parallel execution
    std::vector<double> results;
    for (int i = 0; i < 5; ++i) {
        cache_->clear();
        results.push_back(obj_func->calculate(params));
    }
    
    // All results should be identical
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_DOUBLE_EQ(results[0], results[i]);
    }
}

TEST_F(SEPAIHRDObjectiveFunctionTest, ExtremeParameterValuesTest) {
    auto obj_func = createObjectiveFunction();
    
    Eigen::VectorXd small_beta_params = parameter_manager_->getCurrentParameters();
    small_beta_params(0) = 1e-6;
    
    double small_beta_likelihood = obj_func->calculate(small_beta_params);
    EXPECT_TRUE(std::isfinite(small_beta_likelihood));
    
    // Test with large beta (high transmission)
    Eigen::VectorXd large_beta_params = parameter_manager_->getCurrentParameters();
    large_beta_params(0) = 0.99;
    
    double large_beta_likelihood = obj_func->calculate(large_beta_params);
    EXPECT_TRUE(std::isfinite(large_beta_likelihood));
}

// Test likelihood calculation components
TEST_F(SEPAIHRDObjectiveFunctionTest, LikelihoodComponentsTest) {    
    auto obj_func = createObjectiveFunction();
    Eigen::VectorXd params = parameter_manager_->getCurrentParameters();
    
    // Calculate with full data
    double full_likelihood = obj_func->calculate(params);
    
    // Create data with only hospitalizations
    Eigen::MatrixXd hosp_only = calibration_data_->getNewHospitalizations();
    Eigen::MatrixXd zeros = Eigen::MatrixXd::Zero(NUM_TIME_POINTS, NUM_AGE_CLASSES);
    Eigen::VectorXd zero_vector = Eigen::VectorXd::Zero(NUM_AGE_CLASSES);
    
    auto hosp_only_data = std::make_unique<CalibrationData>(
        zeros, hosp_only, zeros, zeros,
        test_params_.N,
        zero_vector, zero_vector, zero_vector, zero_vector,
        NUM_AGE_CLASSES
    );
    
    auto hosp_only_func = std::make_unique<SEPAIHRDObjectiveFunction>(
        model_,
        *parameter_manager_,
        *cache_,
        *hosp_only_data,
        time_points_,
        initial_state_,
        solver_strategy_,
        ABS_ERROR,
        REL_ERROR
    );
    
    cache_->clear();
    double hosp_only_likelihood = hosp_only_func->calculate(params);
    
    EXPECT_NE(full_likelihood, hosp_only_likelihood);
}

// Test memory efficiency with large datasets
TEST_F(SEPAIHRDObjectiveFunctionTest, LargeDatasetTest) {
    int large_num_points = 365;
    std::vector<double> year_time_points;
    for (int i = 0; i < large_num_points; ++i) {
        year_time_points.push_back(static_cast<double>(i));
    }
    
    Eigen::MatrixXd large_data = Eigen::MatrixXd::Random(large_num_points, NUM_AGE_CLASSES).cwiseAbs() * 100;
    Eigen::VectorXd zero_vector = Eigen::VectorXd::Zero(NUM_AGE_CLASSES);
    
    auto large_calibration_data = std::make_unique<CalibrationData>(
        large_data, large_data, large_data, large_data,
        test_params_.N,
        zero_vector, zero_vector, zero_vector, zero_vector,
        NUM_AGE_CLASSES
    );
    
    auto obj_func = std::make_unique<SEPAIHRDObjectiveFunction>(
        model_,
        *parameter_manager_,
        *cache_,
        *large_calibration_data,
        year_time_points,
        initial_state_,
        solver_strategy_,
        ABS_ERROR,
        REL_ERROR
    );
    
    Eigen::VectorXd params = parameter_manager_->getCurrentParameters();
    
    ASSERT_NO_THROW({
        double likelihood = obj_func->calculate(params);
        EXPECT_TRUE(std::isfinite(likelihood));
    });
}

// Test individual compartment contributions
TEST_F(SEPAIHRDObjectiveFunctionTest, CompartmentContributionTest) {
    
    Eigen::MatrixXd zeros = Eigen::MatrixXd::Zero(NUM_TIME_POINTS, NUM_AGE_CLASSES);
    Eigen::MatrixXd hosp_data = calibration_data_->getNewHospitalizations();
    Eigen::VectorXd zero_vector = Eigen::VectorXd::Zero(NUM_AGE_CLASSES);
    
    // Create separate cache instances for each objective function to avoid cache collisions
    auto hosp_cache = std::make_unique<SimulationCache>(1000, 8);
    auto icu_cache = std::make_unique<SimulationCache>(1000, 8);
    auto deaths_cache = std::make_unique<SimulationCache>(1000, 8);
    auto total_cache = std::make_unique<SimulationCache>(1000, 8);
    
    // Hospitalizations only
    auto hosp_only_data = std::make_unique<CalibrationData>(
        zeros, hosp_data, zeros, zeros,
        test_params_.N, zero_vector, zero_vector, zero_vector, zero_vector,
        NUM_AGE_CLASSES
    );
    
    auto hosp_only_func = std::make_unique<SEPAIHRDObjectiveFunction>(
        model_, *parameter_manager_, *hosp_cache, *hosp_only_data,
        time_points_, initial_state_, solver_strategy_, ABS_ERROR, REL_ERROR
    );
    
    // ICU only
    auto icu_only_data = std::make_unique<CalibrationData>(
        zeros, zeros, calibration_data_->getNewICU(), zeros,
        test_params_.N, zero_vector, zero_vector, zero_vector, zero_vector,
        NUM_AGE_CLASSES
    );
    
    auto icu_only_func = std::make_unique<SEPAIHRDObjectiveFunction>(
        model_, *parameter_manager_, *icu_cache, *icu_only_data,
        time_points_, initial_state_, solver_strategy_, ABS_ERROR, REL_ERROR
    );
    
    // Deaths only
    auto deaths_only_data = std::make_unique<CalibrationData>(
        zeros, zeros, zeros, calibration_data_->getNewDeaths(),
        test_params_.N, zero_vector, zero_vector, zero_vector, zero_vector,
        NUM_AGE_CLASSES
    );
    
    auto deaths_only_func = std::make_unique<SEPAIHRDObjectiveFunction>(
        model_, *parameter_manager_, *deaths_cache, *deaths_only_data,
        time_points_, initial_state_, solver_strategy_, ABS_ERROR, REL_ERROR
    );
    
    // Full data with separate cache
    auto total_func = std::make_unique<SEPAIHRDObjectiveFunction>(
        model_, *parameter_manager_, *total_cache, *calibration_data_,
        time_points_, initial_state_, solver_strategy_, ABS_ERROR, REL_ERROR
    );
    
    Eigen::VectorXd params = parameter_manager_->getCurrentParameters();
    
    double ll_hosp = hosp_only_func->calculate(params);
    double ll_icu = icu_only_func->calculate(params);
    double ll_deaths = deaths_only_func->calculate(params);
    double ll_total = total_func->calculate(params);
    
    // Test that all likelihoods are finite
    EXPECT_TRUE(std::isfinite(ll_hosp));
    EXPECT_TRUE(std::isfinite(ll_icu));
    EXPECT_TRUE(std::isfinite(ll_deaths));
    EXPECT_TRUE(std::isfinite(ll_total));
    
    EXPECT_NE(ll_hosp, ll_icu);
    EXPECT_NE(ll_hosp, ll_deaths);
    EXPECT_NE(ll_icu, ll_deaths);
    
    EXPECT_NE(ll_total, ll_hosp);
    EXPECT_NE(ll_total, ll_icu);
    EXPECT_NE(ll_total, ll_deaths);
    
    double sum_individual = ll_hosp + ll_icu + ll_deaths;
    EXPECT_NE(ll_total, sum_individual);
    
    EXPECT_GE(ll_total, ll_hosp);
    EXPECT_GE(ll_total, ll_icu);
    EXPECT_GE(ll_total, ll_deaths);
}


TEST_F(SEPAIHRDObjectiveFunctionTest, ManualPoissonLikelihoodTest) {
    const int synthetic_time_points = 5;
    const int synthetic_age_classes = 2;
    const double epsilon = 1e-10;
    Eigen::VectorXd synthetic_N(synthetic_age_classes);
    synthetic_N << 1000, 2000;

    Eigen::MatrixXd synthetic_observed_hosp(synthetic_time_points, synthetic_age_classes);
    synthetic_observed_hosp << 5, 3,
                               2, 7,
                               4, 1,
                               6, 0,
                               3, 5;

    Eigen::MatrixXd synthetic_simulated_hosp(synthetic_time_points, synthetic_age_classes);
    synthetic_simulated_hosp << 4.8, 3.2,
                                2.1, 6.9,
                                3.9, 1.1,
                                5.8, 0.2,
                                3.1, 4.9;
    double manual_ll = 0.0;
    for (int i = 0; i < synthetic_observed_hosp.rows(); ++i) {
        for (int j = 0; j < synthetic_observed_hosp.cols(); ++j) {
            double obs = synthetic_observed_hosp(i, j);
            double sim = synthetic_simulated_hosp(i, j) + epsilon;
            manual_ll += obs * std::log(sim) - sim;
        }
    }

    Eigen::MatrixXd dummy_data = synthetic_observed_hosp;
    Eigen::VectorXd dummy_vector = Eigen::VectorXd::Zero(synthetic_age_classes);
    std::vector<double> synthetic_time_vector;
    for (int t = 0; t < synthetic_time_points; ++t) {
        synthetic_time_vector.push_back(static_cast<double>(t));
    }
    auto synthetic_calibration_data = std::make_unique<CalibrationData>(
        dummy_data, 
        synthetic_observed_hosp,
        dummy_data,
        dummy_data,
        synthetic_N,
        dummy_vector, dummy_vector, dummy_vector, dummy_vector,
        synthetic_age_classes
    );

    auto synthetic_cache = std::make_unique<SimulationCache>(100, 8);
    
    TestableSEPAIHRDObjectiveFunction testable_obj_func(
        model_,
        *parameter_manager_,
        *synthetic_cache,
        *synthetic_calibration_data,
        synthetic_time_vector,
        initial_state_,
        solver_strategy_,
        ABS_ERROR,
        REL_ERROR
    );

    double func_ll = testable_obj_func.publicCalculateSingleLogLikelihood(synthetic_simulated_hosp,
                                                                          synthetic_observed_hosp,
                                                                          "Hospitalizations");
    EXPECT_NEAR(manual_ll, func_ll, 1e-8)
        << "Manual LL (" << manual_ll << ") and function LL (" << func_ll << ") differ.";
}

} // namespace test
} // namespace epidemic