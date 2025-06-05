#include "utils/FileUtils.hpp"
#include "model/parameters/SEPAIHRDParameters.hpp"
#include "exceptions/Exceptions.hpp"
#include <gtest/gtest.h>
#include <string>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;

// Test Fixture for tests needing filesystem setup/teardown
class FileUtilsFixture : public ::testing::Test {
protected:
    std::string baseTestDir = "temp_fileutils_test_dir";
    std::string nestedDir = "subdir1/subdir2";
    std::string paramsFilename = "test_params.txt";
    std::string invalidParamsFilename = "invalid_params.txt";
    std::string missingParamsFilename = "missing_params.txt";
    std::string extraParamsFilename = "extra_params.txt";
    std::string nonExistentFile = "does_not_exist.txt";

    // Original working directory to restore later
    fs::path original_cwd;

    void SetUp() override {
        original_cwd = fs::current_path();
        fs::remove_all(baseTestDir);
        fs::create_directories(baseTestDir);

        // Establish a dummy project structure for getProjectRoot/getOutputPath tests.
        fs::create_directory(FileUtils::joinPaths(baseTestDir, "data"));
        fs::create_directory(FileUtils::joinPaths(baseTestDir, "include"));
        fs::create_directory(FileUtils::joinPaths(baseTestDir, "src"));

        // Change CWD into the test directory so getProjectRoot finds it.
        fs::current_path(baseTestDir);

        {
            std::ofstream paramsFile(paramsFilename);
            paramsFile << "# This is a comment\n";
            paramsFile << "beta 0.5\n";
            paramsFile << "theta 0.1\n";
            paramsFile << "sigma 0.2\n";
            paramsFile << "gamma_p 0.3\n";
            paramsFile << "gamma_A 0.4\n";
            paramsFile << "gamma_I 0.5\n";
            paramsFile << "gamma_H 0.6\n";
            paramsFile << "gamma_ICU 0.7\n";
            paramsFile << "contact_matrix_scaling_factor 1.0\n";
            paramsFile << "p 0.1 0.2 # Age-specific\n";
            paramsFile << "h 0.3 0.4\n";
            paramsFile << "icu 0.05 0.1\n";
            paramsFile << "d_H 0.01 0.02\n";
            paramsFile << "d_ICU 0.03 0.04\n";
        }

        {
            std::ofstream invalidParamsFile(invalidParamsFilename);
            invalidParamsFile << "beta not_a_number\n";
        }

        {
            std::ofstream missingParamsFile(missingParamsFilename);
            missingParamsFile << "p 0.1 # Only one value for 2 age classes\n";
        }

        {
            std::ofstream extraParamsFile(extraParamsFilename);
            extraParamsFile << "p 0.1 0.2 0.3 # Three values for 2 age classes\n";
        }
    }

    void TearDown() override {
        fs::current_path(original_cwd);
        fs::remove_all(baseTestDir);
    }

    std::string getTestPath(const std::string& relativePath) {
        return fs::absolute(relativePath).string();
    }

    // Returns the expected project root (absolute path to baseTestDir)
    std::string getExpectedProjectRoot() {
        return (original_cwd / baseTestDir).string();
    }
};

TEST(FileUtilsTest, JoinPaths) {
    EXPECT_EQ(FileUtils::joinPaths("path/to", "file.txt"), "path/to/file.txt");
    EXPECT_EQ(FileUtils::joinPaths("/", "home/user"), "/home/user");
    EXPECT_EQ(FileUtils::joinPaths("path/to/", "file.txt"), "path/to/file.txt");
    EXPECT_EQ(FileUtils::joinPaths("path/to", "/file.txt"), "path/to/file.txt");
    EXPECT_EQ(FileUtils::joinPaths("", "file.txt"), "file.txt");
    EXPECT_EQ(FileUtils::joinPaths("path/to", ""), "path/to");
    EXPECT_EQ(FileUtils::joinPaths("", ""), "");
    EXPECT_EQ(FileUtils::joinPaths("path/./to", "../file.txt"), "path/file.txt");
}

TEST_F(FileUtilsFixture, EnsureDirectoryExists_CreateNew) {
    std::string newDirPath = "new_dir";
    ASSERT_FALSE(fs::exists(newDirPath));
    EXPECT_TRUE(FileUtils::ensureDirectoryExists(newDirPath));
    EXPECT_TRUE(fs::exists(newDirPath));
    EXPECT_TRUE(fs::is_directory(newDirPath));
}

TEST_F(FileUtilsFixture, EnsureDirectoryExists_Existing) {
    std::string existingDirPath = "data";
    ASSERT_TRUE(fs::exists(existingDirPath));
    EXPECT_TRUE(FileUtils::ensureDirectoryExists(existingDirPath));
    EXPECT_TRUE(fs::exists(existingDirPath));
}

TEST_F(FileUtilsFixture, EnsureDirectoryExists_Nested) {
    std::string nestedPath = nestedDir;
    ASSERT_FALSE(fs::exists(nestedPath));
    EXPECT_TRUE(FileUtils::ensureDirectoryExists(nestedPath));
    EXPECT_TRUE(fs::exists(nestedPath));
    EXPECT_TRUE(fs::is_directory(nestedPath));
    EXPECT_TRUE(fs::exists("subdir1"));
}

TEST_F(FileUtilsFixture, GetOutputPath_CreatesDirAndReturnsPath) {
    std::string expectedOutputDir = FileUtils::joinPaths(FileUtils::getProjectRoot(), "data/output");
    ASSERT_FALSE(fs::exists(expectedOutputDir));

    std::string outputPath = FileUtils::getOutputPath();
    EXPECT_EQ(outputPath, expectedOutputDir);
    EXPECT_TRUE(fs::exists(expectedOutputDir));
    EXPECT_TRUE(fs::is_directory(expectedOutputDir));
}

TEST_F(FileUtilsFixture, GetOutputPath_WithFilename) {
    std::string filename = "result.csv";
    std::string expectedOutputDir = FileUtils::joinPaths(FileUtils::getProjectRoot(), "data/output");
    std::string expectedFilePath = FileUtils::joinPaths(expectedOutputDir, filename);

    ASSERT_FALSE(fs::exists(expectedOutputDir));

    std::string outputFilePath = FileUtils::getOutputPath(filename);
    EXPECT_EQ(outputFilePath, expectedFilePath);
    EXPECT_TRUE(fs::exists(expectedOutputDir));
}

TEST_F(FileUtilsFixture, ReadSEPAIHRDParameters_ValidFile) {
    int num_age_classes = 2;
    epidemic::SEPAIHRDParameters params;
    ASSERT_NO_THROW({
        params = FileUtils::readSEPAIHRDParameters(paramsFilename, num_age_classes);
    });

    EXPECT_DOUBLE_EQ(params.beta, 0.5);
    EXPECT_DOUBLE_EQ(params.sigma, 0.2);
    EXPECT_DOUBLE_EQ(params.gamma_ICU, 0.7);
    EXPECT_DOUBLE_EQ(params.contact_matrix_scaling_factor, 1.0);

    ASSERT_EQ(params.p.size(), num_age_classes);
    EXPECT_DOUBLE_EQ(params.p(0), 0.1);
    EXPECT_DOUBLE_EQ(params.p(1), 0.2);

    ASSERT_EQ(params.d_ICU.size(), num_age_classes);
    EXPECT_DOUBLE_EQ(params.d_ICU(0), 0.03);
    EXPECT_DOUBLE_EQ(params.d_ICU(1), 0.04);

    ASSERT_EQ(params.h.size(), num_age_classes);
    ASSERT_EQ(params.icu.size(), num_age_classes);
    ASSERT_EQ(params.d_H.size(), num_age_classes);
}

TEST_F(FileUtilsFixture, ReadSEPAIHRDParameters_FileOpenError) {
    EXPECT_THROW(
        FileUtils::readSEPAIHRDParameters(nonExistentFile, 2),
        epidemic::FileIOException
    );
}

TEST_F(FileUtilsFixture, ReadSEPAIHRDParameters_InvalidNumberFormat) {
    EXPECT_THROW(
        {
            try {
                FileUtils::readSEPAIHRDParameters(invalidParamsFilename, 2);
            } catch (const epidemic::DataFormatException& e) {
                EXPECT_NE(std::string(e.what()).find("Error reading scalar value"), std::string::npos);
                throw;
            }
        },
        epidemic::DataFormatException
    );
}

TEST_F(FileUtilsFixture, ReadSEPAIHRDParameters_NotEnoughAgeValues) {
    EXPECT_THROW(
        {
            try {
                FileUtils::readSEPAIHRDParameters(missingParamsFilename, 2);
            } catch (const epidemic::DataFormatException& e) {
                EXPECT_NE(std::string(e.what()).find("Error reading value for age class 1"), std::string::npos);
                throw;
            }
        },
        epidemic::DataFormatException
    );
}

TEST_F(FileUtilsFixture, ReadSEPAIHRDParameters_TooManyAgeValues) {
    EXPECT_THROW(
        {
            try {
                FileUtils::readSEPAIHRDParameters(extraParamsFilename, 2);
            } catch (const epidemic::DataFormatException& e) {
                EXPECT_NE(std::string(e.what()).find("Too many values provided"), std::string::npos);
                throw;
            }
        },
        epidemic::DataFormatException
    );
}
TEST_F(FileUtilsFixture, ReadSEPAIHRDParameters_ValidKappaVectors) {
    std::string filename = "kappa_params.txt";
    {
        std::ofstream file(filename);
        file << "kappa_end_times 1.0 2.0 3.0\n";
        file << "kappa_values 0.5 0.7 0.9\n";
    }

    epidemic::SEPAIHRDParameters params = FileUtils::readSEPAIHRDParameters(filename, 2);
    
    ASSERT_EQ(params.kappa_end_times.size(), 3);
    EXPECT_DOUBLE_EQ(params.kappa_end_times[0], 1.0);
    EXPECT_DOUBLE_EQ(params.kappa_end_times[1], 2.0);
    EXPECT_DOUBLE_EQ(params.kappa_end_times[2], 3.0);
    
    ASSERT_EQ(params.kappa_values.size(), 3);
    EXPECT_DOUBLE_EQ(params.kappa_values[0], 0.5);
    EXPECT_DOUBLE_EQ(params.kappa_values[1], 0.7);
    EXPECT_DOUBLE_EQ(params.kappa_values[2], 0.9);
}

TEST_F(FileUtilsFixture, ReadSEPAIHRDParameters_KappaMismatch) {
    std::string filename = "mismatched_kappa.txt";
    {
        std::ofstream file(filename);
        file << "kappa_end_times 1.0 2.0 3.0\n";
        file << "kappa_values 0.5 0.7\n";
    }

    EXPECT_THROW({
        FileUtils::readSEPAIHRDParameters(filename, 2);
    }, epidemic::DataFormatException);
}

TEST_F(FileUtilsFixture, ReadSEPAIHRDParameters_InvalidKappaFormat) {
    std::string filename = "invalid_kappa.txt";
    {
        std::ofstream file(filename);
        file << "kappa_end_times 1.0 invalid 3.0\n";
    }

    EXPECT_THROW({
        try {
            FileUtils::readSEPAIHRDParameters(filename, 2);
        } catch (const epidemic::DataFormatException& e) {
            EXPECT_NE(std::string(e.what()).find("Invalid non-numeric data"), std::string::npos);
            throw;
        }
    }, epidemic::DataFormatException);
}

TEST_F(FileUtilsFixture, ReadSEPAIHRDParameters_ExtraScalarValue) {
    std::string filename = "extra_scalar.txt";
    {
        std::ofstream file(filename);
        file << "beta 0.5 0.6\n";
    }

    EXPECT_THROW({
        try {
            FileUtils::readSEPAIHRDParameters(filename, 2);
        } catch (const epidemic::DataFormatException& e) {
            EXPECT_NE(std::string(e.what()).find("Too many values provided"), std::string::npos);
            throw;
        }
    }, epidemic::DataFormatException);
}

TEST_F(FileUtilsFixture, ReadSEPAIHRDParameters_MissingScalarValue) {
    std::string filename = "missing_scalar.txt";
    {
        std::ofstream file(filename);
        file << "beta\n";
    }

    EXPECT_THROW({
        try {
            FileUtils::readSEPAIHRDParameters(filename, 2);
        } catch (const epidemic::DataFormatException& e) {
            EXPECT_NE(std::string(e.what()).find("Missing scalar value"), std::string::npos);
            throw;
        }
    }, epidemic::DataFormatException);
}

TEST_F(FileUtilsFixture, ReadSEPAIHRDParameters_WhitespaceHandling) {
    std::string filename = "whitespace.txt";
    {
        std::ofstream file(filename);
        file << "   beta    0.5   # Comment with spaces   \n";
        file << "\tbeta\t0.6\t#\tComment with tabs\t\n";
        file << "p\t0.1  0.2   # Mixed whitespace\n";
    }

    EXPECT_NO_THROW({
        epidemic::SEPAIHRDParameters params = FileUtils::readSEPAIHRDParameters(filename, 2);
        EXPECT_DOUBLE_EQ(params.beta, 0.6); 
        EXPECT_DOUBLE_EQ(params.p(0), 0.1);
        EXPECT_DOUBLE_EQ(params.p(1), 0.2);
    });
}

TEST_F(FileUtilsFixture, GetProjectRoot_NoProjectStructure) {
    fs::current_path(original_cwd);
    
    fs::path isolated_parent_dir = fs::temp_directory_path() / "mmid_gpr_test_no_structure";
    
    fs::remove_all(isolated_parent_dir); 
    fs::create_directories(isolated_parent_dir);

    fs::path current_test_dir = isolated_parent_dir / "test_subdir";
    fs::create_directory(current_test_dir);
    
    fs::current_path(current_test_dir);

    std::string root = FileUtils::getProjectRoot();
    EXPECT_EQ(root, fs::absolute(current_test_dir).lexically_normal().string());

    fs::current_path(original_cwd); 
    fs::remove_all(isolated_parent_dir);
}

TEST_F(FileUtilsFixture, GetProjectRoot_NestedStructure) {
    // First move back to original working directory outside the test fixture
    fs::current_path(original_cwd);
    
    // Create a nested project structure outside the test fixture directory
    fs::path projectDir = original_cwd / "nested_project";
    fs::create_directories(projectDir / "src/deep/path");
    fs::create_directory(projectDir / "data");
    fs::create_directory(projectDir / "include");
    
    // Change to a deep nested directory
    fs::current_path(projectDir / "src/deep/path");
    
    std::string root = FileUtils::getProjectRoot();
    EXPECT_EQ(root, fs::absolute(projectDir).lexically_normal().string());
    
    fs::current_path(original_cwd);
    fs::remove_all(projectDir);
}