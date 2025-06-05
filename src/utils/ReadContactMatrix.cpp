#include "utils/ReadContactMatrix.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace epidemic { 

Eigen::MatrixXd readMatrixFromCSV(const std::string& filename, int rows, int cols) {
    Eigen::MatrixXd mat(rows, cols);
    std::ifstream file(filename, std::ios::binary);
    const std::string funcName = "epidemic::readMatrixFromCSV";
    if (!file.is_open()) {
        throw epidemic::CSVReadException(epidemic::CSVReadException::ErrorType::FileOpenError, funcName, filename);
    }
    std::string line;
    line.reserve(1024);
    std::string cell;
    cell.reserve(32);

    std::stringstream ss;

    while (std::getline(file, line)) {
        if (line.empty() || line.substr(0, 2) != "//") {
            break;
        }
    }
    if (file.eof() && (line.empty() || line.substr(0, 2) == "//")) {
        throw CSVReadException(CSVReadException::ErrorType::NotEnoughRows, funcName, "No data rows found in file: " + filename);
    }

    ss.str(std::move(line));
    ss.clear();
    for (int j = 0; j < cols; ++j) {
        if (!std::getline(ss, cell, ',')) {
            throw CSVReadException(CSVReadException::ErrorType::NotEnoughColumns, funcName, "row 1 in " + filename);
        }
        try {
            mat(0, j) = std::stod(cell);
        } catch (const std::invalid_argument&) {
            throw epidemic::CSVReadException(epidemic::CSVReadException::ErrorType::InvalidNumberFormat, funcName,
                "row 1, column " + std::to_string(j+1) + ": '" + cell + "' in " + filename);
        } catch (const std::out_of_range&) {
            throw epidemic::CSVReadException(epidemic::CSVReadException::ErrorType::InvalidNumberFormat, funcName,
                "Number out of range at row 1, column " + std::to_string(j+1) + ": '" + cell + "' in " + filename);
        }
    }

    for (int i = 1; i < rows; ++i) {
        if (!std::getline(file, line)) {
            throw epidemic::CSVReadException(epidemic::CSVReadException::ErrorType::NotEnoughRows, funcName,
                "expected " + std::to_string(rows) + " rows, found " + std::to_string(i) + " in " + filename);
        }
        if (line.empty()) {
            i--;
            continue;
        }

        ss.str(std::move(line));
        ss.clear();

        for (int j = 0; j < cols; ++j) {
            if (!std::getline(ss, cell, ',')) {
                throw epidemic::CSVReadException(epidemic::CSVReadException::ErrorType::NotEnoughColumns, funcName,
                    "row " + std::to_string(i+1) + " in " + filename);
            }
            
            try {
                mat(i, j) = std::stod(cell);
            } catch (const std::invalid_argument&) {
                throw epidemic::CSVReadException(epidemic::CSVReadException::ErrorType::InvalidNumberFormat, funcName,
                    "row " + std::to_string(i+1) + ", column " +
                    std::to_string(j+1) + ": '" + cell + "' in " + filename);
            } catch (const std::out_of_range&) {
                throw epidemic::CSVReadException(epidemic::CSVReadException::ErrorType::InvalidNumberFormat, funcName,
                    "Number out of range at row " + std::to_string(i+1) + ", column " +
                    std::to_string(j+1) + ": '" + cell + "' in " + filename);
            }
        }
    }

    return mat;
}

} // namespace epidemic