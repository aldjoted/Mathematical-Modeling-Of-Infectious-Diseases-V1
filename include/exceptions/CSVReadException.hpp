#ifndef CSV_READ_EXCEPTION_HPP
#define CSV_READ_EXCEPTION_HPP

#include "exceptions/Exceptions.hpp"
#include <stdexcept>
#include <string>

namespace epidemic {

/**
 * @brief Exception class for CSV reading errors
 * 
 * Represents various errors that can occur when reading CSV files,
 * including file access issues and data format problems.
 */
class CSVReadException : public DataFormatException {
public:
    /**
     * @brief Types of CSV reading errors that can occur
     */
    enum class ErrorType {
        FileOpenError,      ///< Failed to open the CSV file
        NotEnoughColumns,   ///< Row has fewer columns than expected
        NotEnoughRows,      ///< File has fewer rows than expected
        InvalidNumberFormat ///< Could not parse a value as a number
    };

    /**
     * @brief Constructs a new CSV read exception
     * 
     * @param type The specific type of error that occurred
     * @param details Additional information about the error
     */
    CSVReadException(ErrorType type, const std::string& functionName, const std::string& details);
    
    /**
     * @brief Get the type of error that occurred
     * 
     * @return ErrorType The error type
     */
    ErrorType getErrorType() const noexcept;

private:
    ErrorType errorType; ///< Stores the type of error that occurred
    
    /**
     * @brief Creates an error message based on the error type and details
     * 
     * @param type The specific type of error
     * @param details Additional error information
     * @return std::string The formatted error message
     */
    static std::string createMessage(ErrorType type, const std::string& details);
};

} // namespace epidemic

#endif // CSV_READ_EXCEPTION_HPP