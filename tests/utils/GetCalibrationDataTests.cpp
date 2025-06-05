#include "gtest/gtest.h"
#include "utils/GetCalibrationData.hpp"
#include "exceptions/Exceptions.hpp"
#include <Eigen/Dense>
#include <fstream>
#include <vector>
#include <string>
#include <cstdio>

using namespace epidemic;

// Helper function to create a temporary CSV file for testing
std::string createTestCSVFile(const std::string& filename_prefix, const std::vector<std::string>& lines) {
    std::string filename = filename_prefix + "_" + std::to_string(std::time(nullptr)) + "_" + std::to_string(rand() % 1000) + ".csv";
    std::ofstream test_file(filename);
    if (!test_file.is_open()) {
        throw std::runtime_error("Failed to create test CSV file: " + filename);
    }
    for (const auto& line : lines) {
        test_file << line << std::endl;
    }
    test_file.close();
    return filename;
}

// Test fixture for CalibrationData tests
class CalibrationDataTest : public ::testing::Test {
protected:
    const int NUM_AGE_CLASSES = 4;
    std::string test_csv_filepath;
    std::vector<std::string> files_to_cleanup;


    void SetUp() override {
        srand(static_cast<unsigned int>(time(nullptr)));
    }

    void TearDown() override {
        for (const auto& filepath : files_to_cleanup) {
            std::remove(filepath.c_str());
        }
        files_to_cleanup.clear();
    }

    // Creates a CSV and registers it for cleanup
    std::string createManagedTestCSV(const std::string& prefix, const std::vector<std::string>& content) {
        std::string filepath = createTestCSVFile(prefix, content);
        files_to_cleanup.push_back(filepath);
        return filepath;
    }

    std::vector<std::string> getValidHeaderLines() {
        return {
            "date,"
            "new_confirmed_0_30,new_confirmed_30_60,new_confirmed_60_80,new_confirmed_80_plus,"
            "new_deceased_0_30,new_deceased_30_60,new_deceased_60_80,new_deceased_80_plus,"
            "new_hospitalized_patients_0_30,new_hospitalized_patients_30_60,new_hospitalized_patients_60_80,new_hospitalized_patients_80_plus,"
            "new_intensive_care_patients_0_30,new_intensive_care_patients_30_60,new_intensive_care_patients_60_80,new_intensive_care_patients_80_plus,"
            "population_0_30,population_30_60,population_60_80,population_80_plus,"
            "cumulative_confirmed_0_30,cumulative_confirmed_30_60,cumulative_confirmed_60_80,cumulative_confirmed_80_plus,"
            "cumulative_deceased_0_30,cumulative_deceased_30_60,cumulative_deceased_60_80,cumulative_deceased_80_plus,"
            "cumulative_hospitalized_patients_0_30,cumulative_hospitalized_patients_30_60,cumulative_hospitalized_patients_60_80,cumulative_hospitalized_patients_80_plus,"
            "cumulative_intensive_care_patients_0_30,cumulative_intensive_care_patients_30_60,cumulative_intensive_care_patients_60_80,cumulative_intensive_care_patients_80_plus"
        };
    }

    std::vector<std::string> getValidDataRowLines(const std::string& date, int start_val) { 
         std::string row_str = date + ",";
         for (int field_group = 0; field_group < 9; ++field_group) {
            for (int age_group = 0; age_group < NUM_AGE_CLASSES; ++age_group) {
                double val = 0.0;
                if (field_group < 4) {
                    val = start_val + (field_group * 10) + age_group;
                } else if (field_group == 4) {
                    val = 10000 * (start_val + 40 + age_group);
                } else {
                     val = start_val * 10 + ( (field_group-5) * 10 + 50) + age_group;
                }
                row_str += std::to_string(static_cast<int>(val));
                if (!(field_group == 8 && age_group == NUM_AGE_CLASSES -1)) {
                    row_str += ",";
                }
            }
         }
        return {row_str};
    }
};

TEST_F(CalibrationDataTest, ConstructorWithMatrices_PopulatesDataCorrectly) {
    int n_points = 2;
    Eigen::MatrixXd new_c(n_points, NUM_AGE_CLASSES);
    Eigen::MatrixXd new_h(n_points, NUM_AGE_CLASSES);
    Eigen::MatrixXd new_i(n_points, NUM_AGE_CLASSES);
    Eigen::MatrixXd new_d(n_points, NUM_AGE_CLASSES);
    Eigen::VectorXd pop(NUM_AGE_CLASSES);
    Eigen::VectorXd cum_c0(NUM_AGE_CLASSES);
    Eigen::VectorXd cum_d0(NUM_AGE_CLASSES);
    Eigen::VectorXd cum_h0(NUM_AGE_CLASSES);
    Eigen::VectorXd cum_i0(NUM_AGE_CLASSES);

    for (int r = 0; r < n_points; ++r) {
        for (int c = 0; c < NUM_AGE_CLASSES; ++c) {
            new_c(r, c) = (r + 1) * 10 + c;
            new_h(r, c) = (r + 1) * 20 + c;
            new_i(r, c) = (r + 1) * 30 + c;
            new_d(r, c) = (r + 1) * 40 + c;
        }
    }
    for (int c = 0; c < NUM_AGE_CLASSES; ++c) {
        pop(c) = 10000 * (c + 1);
        cum_c0(c) = 5 + c;
        cum_d0(c) = 6 + c;
        cum_h0(c) = 7 + c;
        cum_i0(c) = 8 + c;
    }

    CalibrationData cd(new_c, new_h, new_i, new_d, pop, cum_c0, cum_d0, cum_h0, cum_i0, NUM_AGE_CLASSES);

    EXPECT_EQ(cd.getNumDataPoints(), n_points);
    EXPECT_EQ(cd.getNumAgeClasses(), NUM_AGE_CLASSES);
    ASSERT_EQ(cd.getDates().size(), n_points);
    EXPECT_EQ(cd.getDates()[0], "mock_date_0");

    EXPECT_EQ(cd.getNewConfirmedCases(), new_c);
    EXPECT_EQ(cd.getNewHospitalizations(), new_h);
    EXPECT_EQ(cd.getNewICU(), new_i);
    EXPECT_EQ(cd.getNewDeaths(), new_d);
    EXPECT_EQ(cd.getPopulationByAgeGroup(), pop);

    // Test cumulative calculations
    EXPECT_EQ(cd.getCumulativeConfirmedCases().row(0), cum_c0.transpose());
    EXPECT_EQ(cd.getCumulativeConfirmedCases().row(1), (cum_c0 + new_c.row(0).transpose()).transpose());

    EXPECT_EQ(cd.getCumulativeDeaths().row(0), cum_d0.transpose());
    EXPECT_EQ(cd.getCumulativeDeaths().row(1), (cum_d0 + new_d.row(0).transpose()).transpose());

    EXPECT_EQ(cd.getCumulativeHospitalizations().row(0), cum_h0.transpose());
    EXPECT_EQ(cd.getCumulativeHospitalizations().row(1), (cum_h0 + new_h.row(0).transpose()).transpose());

    EXPECT_EQ(cd.getCumulativeICU().row(0), cum_i0.transpose());
    EXPECT_EQ(cd.getCumulativeICU().row(1), (cum_i0 + new_i.row(0).transpose()).transpose());
}

TEST_F(CalibrationDataTest, GetInitialActiveCases_ReturnsFirstRowCumulativeConfirmed) {
    Eigen::MatrixXd new_c = Eigen::MatrixXd::Ones(2, NUM_AGE_CLASSES);
    Eigen::VectorXd pop = Eigen::VectorXd::Constant(NUM_AGE_CLASSES, 1000);
    Eigen::VectorXd cum_c0(NUM_AGE_CLASSES);
    cum_c0 << 10, 12, 15, 18;

    CalibrationData cd(new_c, new_c, new_c, new_c, pop, cum_c0, cum_c0, cum_c0, cum_c0, NUM_AGE_CLASSES);
    EXPECT_EQ(cd.getInitialActiveCases(), cum_c0);
}

TEST_F(CalibrationDataTest, GetInitialActiveCases_ThrowsIfDataEmpty) {
    Eigen::MatrixXd empty_m(0, NUM_AGE_CLASSES);
    Eigen::VectorXd zero_vec = Eigen::VectorXd::Zero(NUM_AGE_CLASSES);
    Eigen::VectorXd valid_pop = Eigen::VectorXd::Constant(NUM_AGE_CLASSES, 100);
    CalibrationData cd(empty_m, empty_m, empty_m, empty_m, valid_pop,
                       zero_vec, zero_vec, zero_vec, zero_vec, NUM_AGE_CLASSES);
    EXPECT_THROW(cd.getInitialActiveCases(), std::runtime_error);
}

TEST_F(CalibrationDataTest, GetInitialSEPAIHRDState_CorrectlyCalculates) {
    int n_points = 1;
    Eigen::MatrixXd new_c(n_points, NUM_AGE_CLASSES);
    Eigen::MatrixXd new_h(n_points, NUM_AGE_CLASSES);
    Eigen::MatrixXd new_i(n_points, NUM_AGE_CLASSES);
    Eigen::MatrixXd new_d(n_points, NUM_AGE_CLASSES); 
    Eigen::VectorXd pop(NUM_AGE_CLASSES);
    Eigen::VectorXd cum_c0(NUM_AGE_CLASSES);
    Eigen::VectorXd cum_d0(NUM_AGE_CLASSES);
    Eigen::VectorXd cum_h0(NUM_AGE_CLASSES);
    Eigen::VectorXd cum_i0(NUM_AGE_CLASSES);

    new_c.row(0) << 5, 6, 7, 8;
    cum_h0 << 2, 3, 4, 5;
    cum_i0 << 1, 1, 2, 2;
    cum_d0 << 0, 1, 1, 2;
    pop << 1000, 2000, 1500, 1000;
    cum_c0 << 5, 6, 7, 8;

    CalibrationData cd(new_c, new_h, new_i, new_d, pop,
                       cum_c0, cum_d0, cum_h0, cum_i0, NUM_AGE_CLASSES);

    double sigma_rate = 1.0 / 5.2;
    double gamma_p_rate = 1.0 / 2.3;
    double gamma_a_rate = 1.0 / 7.0;
    double gamma_i_rate = 1.0 / 7.0;

    // Define age-specific parameters
    Eigen::VectorXd p_asymptomatic(NUM_AGE_CLASSES);
    Eigen::VectorXd h_hospitalization(NUM_AGE_CLASSES);
    p_asymptomatic << 0.5, 0.4, 0.3, 0.2;
    h_hospitalization << 0.01, 0.02, 0.05, 0.1;

    // Call with the required parameters
    Eigen::VectorXd initial_state = cd.getInitialSEPAIHRDState(
        sigma_rate, gamma_p_rate, gamma_a_rate, gamma_i_rate,
        p_asymptomatic, h_hospitalization
    );

    ASSERT_EQ(initial_state.size(), 9 * NUM_AGE_CLASSES);

    // Test that the observable compartments match the data
    EXPECT_DOUBLE_EQ(initial_state(4 * NUM_AGE_CLASSES + 0), 5); 
    EXPECT_DOUBLE_EQ(initial_state(5 * NUM_AGE_CLASSES + 0), 2);  
    EXPECT_DOUBLE_EQ(initial_state(6 * NUM_AGE_CLASSES + 0), 1);  // ICU0_0 from cum_i0
    EXPECT_DOUBLE_EQ(initial_state(8 * NUM_AGE_CLASSES + 0), 0);  // D0_0 from cum_d0

    // Test population conservation for each age group
    for (int age = 0; age < NUM_AGE_CLASSES; ++age) {
        double sum_comps = 0;
        for (int comp = 0; comp < 9; ++comp) {
            sum_comps += initial_state(comp * NUM_AGE_CLASSES + age);
        }
        EXPECT_NEAR(sum_comps, pop(age), 1e-6); // Allow small numerical error
    }

    // Test that all compartments are non-negative
    for (int i = 0; i < initial_state.size(); ++i) {
        EXPECT_GE(initial_state(i), 0.0) << "Compartment " << i << " should be non-negative";
    }
}

TEST_F(CalibrationDataTest, GetInitialSEPAIHRDState_ThrowsIfNoDataPoints) {
     CalibrationData cd(Eigen::MatrixXd(0,NUM_AGE_CLASSES), Eigen::MatrixXd(0,NUM_AGE_CLASSES),
                       Eigen::MatrixXd(0,NUM_AGE_CLASSES), Eigen::MatrixXd(0,NUM_AGE_CLASSES),
                       Eigen::VectorXd::Zero(NUM_AGE_CLASSES), Eigen::VectorXd::Zero(NUM_AGE_CLASSES),
                       Eigen::VectorXd::Zero(NUM_AGE_CLASSES), Eigen::VectorXd::Zero(NUM_AGE_CLASSES),
                       Eigen::VectorXd::Zero(NUM_AGE_CLASSES), NUM_AGE_CLASSES);
    
    // Define dummy parameters
    double sigma_rate = 1.0 / 5.2;
    double gamma_p_rate = 1.0 / 2.3;
    double gamma_a_rate = 1.0 / 7.0;
    double gamma_i_rate = 1.0 / 7.0;
    Eigen::VectorXd p_asymptomatic = Eigen::VectorXd::Constant(NUM_AGE_CLASSES, 0.5);
    Eigen::VectorXd h_hospitalization = Eigen::VectorXd::Constant(NUM_AGE_CLASSES, 0.05);

    EXPECT_THROW(cd.getInitialSEPAIHRDState(sigma_rate, gamma_p_rate, gamma_a_rate, gamma_i_rate,
                                            p_asymptomatic, h_hospitalization), std::runtime_error);
}

TEST_F(CalibrationDataTest, GetInitialSEPAIHRDState_ThrowsIfPopMismatch) {
    Eigen::MatrixXd valid_matrix = Eigen::MatrixXd::Ones(1, NUM_AGE_CLASSES);
    // Provide a population vector with one less element to trigger the mismatch.
    Eigen::VectorXd wrong_pop = Eigen::VectorXd::Zero(NUM_AGE_CLASSES - 1);
    EXPECT_THROW({
        CalibrationData cd(valid_matrix, valid_matrix, valid_matrix, valid_matrix,
                           wrong_pop,
                           Eigen::VectorXd::Ones(NUM_AGE_CLASSES),
                           Eigen::VectorXd::Ones(NUM_AGE_CLASSES),
                           Eigen::VectorXd::Ones(NUM_AGE_CLASSES),
                           Eigen::VectorXd::Ones(NUM_AGE_CLASSES),
                           NUM_AGE_CLASSES);
    }, std::invalid_argument);
}

TEST_F(CalibrationDataTest, GetInitialSEPAIHRDState_ThrowsIfParameterSizeMismatch) {
    Eigen::MatrixXd valid_matrix = Eigen::MatrixXd::Ones(1, NUM_AGE_CLASSES);
    Eigen::VectorXd valid_vector = Eigen::VectorXd::Ones(NUM_AGE_CLASSES);

    CalibrationData cd(valid_matrix, valid_matrix, valid_matrix, valid_matrix,
                       valid_vector, valid_vector, valid_vector, valid_vector, valid_vector, NUM_AGE_CLASSES);
    
    double sigma_rate = 1.0 / 5.2;
    double gamma_p_rate = 1.0 / 2.3;
    double gamma_a_rate = 1.0 / 7.0;
    double gamma_i_rate = 1.0 / 7.0;
    
    // Test with mismatched parameter vector sizes
    Eigen::VectorXd p_mismatch = Eigen::VectorXd::Constant(NUM_AGE_CLASSES - 1, 0.5);
    Eigen::VectorXd h_valid = Eigen::VectorXd::Constant(NUM_AGE_CLASSES, 0.05);

    EXPECT_THROW(cd.getInitialSEPAIHRDState(sigma_rate, gamma_p_rate, gamma_a_rate, gamma_i_rate,
                                            p_mismatch, h_valid), std::runtime_error);

    Eigen::VectorXd p_valid = Eigen::VectorXd::Constant(NUM_AGE_CLASSES, 0.5);
    Eigen::VectorXd h_mismatch = Eigen::VectorXd::Constant(NUM_AGE_CLASSES - 1, 0.05);

    EXPECT_THROW(cd.getInitialSEPAIHRDState(sigma_rate, gamma_p_rate, gamma_a_rate, gamma_i_rate,
                                            p_valid, h_mismatch), std::runtime_error);
}

TEST_F(CalibrationDataTest, GetInitialSEPAIHRDState_ThrowsIfRequiredMatricesEmpty) {
    std::vector<std::string> file_content_header_only = getValidHeaderLines();
    test_csv_filepath = createManagedTestCSV("header_only", file_content_header_only);
    
    EXPECT_THROW(CalibrationData cd(test_csv_filepath), std::runtime_error); // Should fail during construction
}

TEST_F(CalibrationDataTest, GetInitialSEPAIHRDState_HandlesLargeInitialValuesClampingCorrectly) {
    Eigen::MatrixXd new_c(1, NUM_AGE_CLASSES);
    Eigen::MatrixXd new_h_dummy(1, NUM_AGE_CLASSES);
    Eigen::MatrixXd new_i_dummy(1, NUM_AGE_CLASSES);
    Eigen::MatrixXd new_d_dummy(1, NUM_AGE_CLASSES);
    Eigen::VectorXd pop(NUM_AGE_CLASSES);
    Eigen::VectorXd cum_c0(NUM_AGE_CLASSES);
    Eigen::VectorXd cum_d0(NUM_AGE_CLASSES);
    Eigen::VectorXd cum_h0(NUM_AGE_CLASSES);
    Eigen::VectorXd cum_i0(NUM_AGE_CLASSES);

    pop << 100, 100, 100, 100;
    new_c.row(0) << 50, 10, 10, 10;
    cum_c0 << 50, 10, 10, 10;
    cum_h0 << 60, 10, 10, 10;
    cum_i0 << 70, 10, 10, 10;
    cum_d0 << 80, 10, 10, 10;

    CalibrationData cd(new_c, new_h_dummy, new_i_dummy, new_d_dummy, pop,
                       cum_c0, cum_d0, cum_h0, cum_i0, NUM_AGE_CLASSES);

    // Define parameters
    double sigma_rate = 1.0 / 5.2;
    double gamma_p_rate = 1.0 / 2.3;
    double gamma_a_rate = 1.0 / 7.0;
    double gamma_i_rate = 1.0 / 7.0;
    Eigen::VectorXd p_asymptomatic = Eigen::VectorXd::Constant(NUM_AGE_CLASSES, 0.5);
    Eigen::VectorXd h_hospitalization = Eigen::VectorXd::Constant(NUM_AGE_CLASSES, 0.05);

    Eigen::VectorXd initial_state = cd.getInitialSEPAIHRDState(
        sigma_rate, gamma_p_rate, gamma_a_rate, gamma_i_rate,
        p_asymptomatic, h_hospitalization
    );

    // Test clamping behavior for age group 0 where constraints are violated
    EXPECT_DOUBLE_EQ(initial_state(8 * NUM_AGE_CLASSES + 0), 80); // D0_0
    EXPECT_LE(initial_state(6 * NUM_AGE_CLASSES + 0), 20);        // ICU0_0 should be clamped
    EXPECT_GE(initial_state(6 * NUM_AGE_CLASSES + 0), 0);         // ICU0_0 should be non-negative

    // Test population conservation
    for (int age = 0; age < NUM_AGE_CLASSES; ++age) {
        double sum_comps = 0;
        for (int comp = 0; comp < 9; ++comp) {
            sum_comps += initial_state(comp * NUM_AGE_CLASSES + age);
        }
        EXPECT_NEAR(sum_comps, pop(age), 1e-6);
    }
}

// Add a new test for parameter validation
TEST_F(CalibrationDataTest, GetInitialSEPAIHRDState_HandlesInvalidRates) {
    Eigen::MatrixXd valid_matrix = Eigen::MatrixXd::Ones(1, NUM_AGE_CLASSES);
    Eigen::VectorXd valid_vector = Eigen::VectorXd::Constant(NUM_AGE_CLASSES, 1000);

    CalibrationData cd(valid_matrix, valid_matrix, valid_matrix, valid_matrix,
                       valid_vector, valid_vector, valid_vector, valid_vector, valid_vector, NUM_AGE_CLASSES);

    // Test with zero/negative rates (should use fallback logic)
    double sigma_rate = 0.0;  // Invalid
    double gamma_p_rate = 1.0 / 2.3;
    double gamma_a_rate = 1.0 / 7.0;
    double gamma_i_rate = 1.0 / 7.0;
    Eigen::VectorXd p_asymptomatic = Eigen::VectorXd::Constant(NUM_AGE_CLASSES, 0.5);
    Eigen::VectorXd h_hospitalization = Eigen::VectorXd::Constant(NUM_AGE_CLASSES, 0.05);

    // Should not throw, but use fallback initialization
    EXPECT_NO_THROW(cd.getInitialSEPAIHRDState(sigma_rate, gamma_p_rate, gamma_a_rate, gamma_i_rate,
                                               p_asymptomatic, h_hospitalization));
}
