#include "utils/ReadContactMatrix.hpp"
#include "exceptions/Exceptions.hpp"
#include "utils/FileUtils.hpp"
#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>

class ReadContactMatrixTest : public ::testing::Test {
protected:
    std::string testDir = "temp_test_data";
    std::string validCsvPath;
    std::string invalidFormatPath;
    std::string notEnoughRowsPath;
    std::string notEnoughColsPath;
    std::string emptyFilePath;
    std::string nonExistentPath = "non_existent_file.csv";

    void SetUp() override {
        std::filesystem::create_directory(testDir);

        validCsvPath = FileUtils::joinPaths(testDir, "valid_matrix.csv");
        std::ofstream validFile(validCsvPath);
        validFile << "// This is a comment\n";
        validFile << "1.1,2.2,3.3\n";
        validFile << "4.4,5.5,6.6\n";
        validFile.close();

        invalidFormatPath = FileUtils::joinPaths(testDir, "invalid_format.csv");
        std::ofstream invalidFormatFile(invalidFormatPath);
        invalidFormatFile << "1.1,abc,3.3\n";
        invalidFormatFile << "4.4,5.5,6.6\n";
        invalidFormatFile.close();

        notEnoughRowsPath = FileUtils::joinPaths(testDir, "not_enough_rows.csv");
        std::ofstream notEnoughRowsFile(notEnoughRowsPath);
        notEnoughRowsFile << "1.1,2.2,3.3\n";
        notEnoughRowsFile.close();

        notEnoughColsPath = FileUtils::joinPaths(testDir, "not_enough_cols.csv");
        std::ofstream notEnoughColsFile(notEnoughColsPath);
        notEnoughColsFile << "1.1,2.2,3.3\n";
        notEnoughColsFile << "4.4,5.5\n";
        notEnoughColsFile.close();

        emptyFilePath = FileUtils::joinPaths(testDir, "empty_file.csv");
        std::ofstream emptyFile(emptyFilePath);
        emptyFile.close();
    }

    void TearDown() override {
        std::filesystem::remove_all(testDir);
    }
};

TEST_F(ReadContactMatrixTest, ReadValidMatrix) {
    int rows = 2;
    int cols = 3;
    Eigen::MatrixXd expected(rows, cols);
    expected << 1.1, 2.2, 3.3,
                4.4, 5.5, 6.6;

    Eigen::MatrixXd actual = epidemic::readMatrixFromCSV(validCsvPath, rows, cols);
    ASSERT_EQ(actual.rows(), rows);
    ASSERT_EQ(actual.cols(), cols);
    EXPECT_TRUE(actual.isApprox(expected));
}

TEST_F(ReadContactMatrixTest, FileOpenError) {
    EXPECT_THROW(
        epidemic::readMatrixFromCSV(nonExistentPath, 2, 2),
        epidemic::CSVReadException
    );
    try {
         epidemic::readMatrixFromCSV(nonExistentPath, 2, 2);
    } catch (const epidemic::CSVReadException& e) {
        EXPECT_EQ(e.getErrorType(), epidemic::CSVReadException::ErrorType::FileOpenError);
    } catch (...) {
        FAIL() << "Expected CSVReadException::FileOpenError";
    }
}

TEST_F(ReadContactMatrixTest, InvalidNumberFormat) {
     EXPECT_THROW(
        {
            try {
                epidemic::readMatrixFromCSV(invalidFormatPath, 2, 3);
            } catch (const epidemic::CSVReadException& e) {
                EXPECT_EQ(e.getErrorType(), epidemic::CSVReadException::ErrorType::InvalidNumberFormat);
                throw;
            }
        },
        epidemic::CSVReadException
    );
}

TEST_F(ReadContactMatrixTest, NotEnoughRows) {
     EXPECT_THROW(
        {
            try {
                epidemic::readMatrixFromCSV(notEnoughRowsPath, 2, 3);
            } catch (const epidemic::CSVReadException& e) {
                EXPECT_EQ(e.getErrorType(), epidemic::CSVReadException::ErrorType::NotEnoughRows);
                throw;
            }
        },
        epidemic::CSVReadException
    );
}

TEST_F(ReadContactMatrixTest, NotEnoughColumns) {
     EXPECT_THROW(
        {
            try {
                epidemic::readMatrixFromCSV(notEnoughColsPath, 2, 3);
            } catch (const epidemic::CSVReadException& e) {
                EXPECT_EQ(e.getErrorType(), epidemic::CSVReadException::ErrorType::NotEnoughColumns);
                throw;
            }
        },
        epidemic::CSVReadException
    );
}

TEST_F(ReadContactMatrixTest, EmptyFile) {
     EXPECT_THROW(
        {
            try {
                epidemic::readMatrixFromCSV(emptyFilePath, 1, 1);
            } catch (const epidemic::CSVReadException& e) {
                EXPECT_EQ(e.getErrorType(), epidemic::CSVReadException::ErrorType::NotEnoughRows);
                throw;
            }
        },
        epidemic::CSVReadException
    );
}