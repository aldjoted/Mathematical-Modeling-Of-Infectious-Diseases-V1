#include "exceptions/CSVReadException.hpp"
#include <string>

namespace epidemic {

    CSVReadException::CSVReadException(ErrorType type, const std::string& functionName, const std::string& details)
    : DataFormatException(functionName, createMessage(type, details)),
      errorType(type) {}

    CSVReadException::ErrorType CSVReadException::getErrorType() const noexcept {
        return errorType;
    }

    std::string CSVReadException::createMessage(ErrorType type, const std::string& details) {
        std::string baseMsg;
        switch (type) {
            case ErrorType::FileOpenError:
                baseMsg = "Could not open file";
                break;
            case ErrorType::NotEnoughColumns:
                baseMsg = "Not enough columns";
                break;
            case ErrorType::NotEnoughRows:
                baseMsg = "Not enough rows";
                break;
            case ErrorType::InvalidNumberFormat:
                baseMsg = "Invalid number format";
                break;
            default:
                 baseMsg = "Unknown CSV error";
                 break;
        }
        return baseMsg + (details.empty() ? "" : ": " + details); 
    }
}