#ifndef READ_CONTACT_MATRIX_HPP
#define READ_CONTACT_MATRIX_HPP

#include <Eigen/Dense>
#include <string>
#include "exceptions/CSVReadException.hpp"

namespace epidemic { 
/**
 * @brief Reads a contact matrix from a CSV file format
 *
 * @param filename [std::string] The path to the CSV file to read
 * @param rows [int] The expected number of rows in the matrix
 * @param cols [int] The expected number of columns in the matrix
 * 
 * @return Eigen::MatrixXd A matrix containing the data from the CSV file
 * 
 * @throws CSVReadException::FileOpenError If the file cannot be opened
 * @throws CSVReadException::NotEnoughRows If the file has fewer rows than expected
 * @throws CSVReadException::NotEnoughColumns If any row has fewer columns than expected
 * @throws CSVReadException::InvalidNumberFormat If any cell contains invalid numeric data
 * @throws FileIOException If file opening fails for reasons other than permissions/existence (caught by CSVReadException)
 * @throws DataFormatException If parsing fails for reasons other than invalid number format (caught by CSVReadException)
 */
Eigen::MatrixXd readMatrixFromCSV(const std::string& filename, int rows, int cols);
} // namespace epidemic
#endif // READ_CONTACT_MATRIX_HPP