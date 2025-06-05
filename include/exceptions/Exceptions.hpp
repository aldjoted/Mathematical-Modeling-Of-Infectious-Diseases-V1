#ifndef EXCEPTIONS_HPP
#define EXCEPTIONS_HPP

#include <stdexcept>
#include <string>
#include <sstream>

namespace epidemic {

    inline std::string buildErrorMessage(const char* file, int line, const std::string& functionName, const std::string& category, const std::string& message) {
        std::ostringstream oss;
        oss << "[" << file << ":" << line << " (" << functionName << ")] " << category << ": " << message;
        return oss.str();
    }
/**
 * @brief Base exception for epidemic modeling.
 */
class ModelException : public std::runtime_error {
public:
    /**
     * @brief Construct a ModelException.
     * @param functionName Name of the function where the error occurred.
     * @param message Descriptive error message.
     */
    ModelException(const std::string& functionName, const std::string& message)
        : std::runtime_error("[" + functionName + "] " + message),
          functionName_(functionName) {}
    ModelException(const char* file, int line, const std::string& functionName, const std::string& message)
        : std::runtime_error(buildErrorMessage(file, line, functionName, "ModelException", message)),
          functionName_(functionName), file_(file), line_(line) {}

    /**
     * @brief Get the originating function's name.
     * @return const std::string& Function name.
     */
    const std::string& getFunctionName() const noexcept {
        return functionName_;
    }
    const char* getFile() const noexcept { return file_; }
    int getLine() const noexcept { return line_; }

private:
    std::string functionName_;
    const char* file_;
    int line_;
};

/**
 * @brief Exception for invalid method parameters.
 */
class InvalidParameterException : public ModelException {
public:
    /**
     * @brief Construct an InvalidParameterException.
     * @param functionName Name of the function where the error occurred.
     * @param message Details about the invalid parameter.
     */
    public:
    InvalidParameterException(const char* file, int line, const std::string& functionName, const std::string& message)
        : ModelException(functionName, message), file_(file), line_(line) {}
private:
    const char* file_;
    int line_;
};

/**
 * @brief Exception for numerical simulation errors.
 */
class SimulationException : public ModelException {
public:
    /**
     * @brief Construct a SimulationException.
     * @param functionName Name of the function where the error occurred.
     * @param message Details about the simulation error.
     */
    SimulationException(const std::string& functionName, const std::string& message)
        : ModelException(functionName, "Simulation Error: " + message) {}
};

/**
 * @brief Exception for model construction errors.
 */
class ModelConstructionException : public ModelException {
public:
    /**
     * @brief Construct a ModelConstructionException.
     * @param functionName Name of the function where the error occurred.
     * @param message Details about the construction error.
     */
    ModelConstructionException(const std::string& functionName, const std::string& message)
        : ModelException(functionName, "Model Construction Error: " + message) {}
};

/**
 * @brief Exception for intervention application errors.
 */
class InterventionException : public ModelException {
public:
    /**
     * @brief Construct an InterventionException.
     * @param functionName Name of the function where the error occurred.
     * @param message Details about the intervention error.
     */
    InterventionException(const std::string& functionName, const std::string& message)
        : ModelException(functionName, "Intervention Error: " + message) {}
};

/**
 * @brief Exception for file input/output errors.
 */
class FileIOException : public ModelException {
public:
    /**
     * @brief Construct a FileIOException.
     * @param functionName Name of the function where the error occurred.
     * @param message Details about the file I/O error.
     */
    FileIOException(const std::string& functionName, const std::string& message)
        : ModelException(functionName, "File IO Error: " + message) {}
};

/**
 * @brief Exception for data parsing or format errors.
 */
class DataFormatException : public ModelException {
public:
    /**
     * @brief Construct a DataFormatException.
     * @param functionName Name of the function where the error occurred.
     * @param message Details about the formatting error.
     */
    DataFormatException(const std::string& functionName, const std::string& message)
        : ModelException(functionName, "Data Format Error: " + message) {}
};

/**
 * @brief Exception for invalid simulation results.
 */
class InvalidResultException : public ModelException {
public:
    /**
     * @brief Construct an InvalidResultException.
     * @param functionName Name of the function where the error occurred.
     * @param message Details about the invalid result.
     */
    InvalidResultException(const std::string& functionName, const std::string& message)
        : ModelException(functionName, "Invalid Result: " + message) {}
};

/**
 * @brief Exception for out-of-range access.
 */
class OutOfRangeException : public ModelException {
public:
    /**
     * @brief Construct an OutOfRangeException.
     * @param file File where the error occurred.
     * @param line Line number where the error occurred.
     * @param functionName Name of the function where the error occurred.
     * @param message Details about the out-of-range access.
     */
    OutOfRangeException(const char* file, int line, const std::string& functionName, const std::string& message)
        : ModelException(functionName, message), file_(file), line_(line) {}
private:
    const char* file_;
    int line_;
};

} // namespace epidemic

#define THROW_INVALID_PARAM(func, msg) throw epidemic::InvalidParameterException(__FILE__, __LINE__, func, msg)
#define THROW_SIMULATION_ERROR(func, msg) throw epidemic::SimulationException(__FILE__, __LINE__, func, msg)
#define THROW_OUT_OF_RANGE(func, msg) throw epidemic::OutOfRangeException(__FILE__, __LINE__, func, msg)
#define THROW_MODEL_EXCEPTION(func, msg) throw epidemic::ModelException(func, msg)

#endif // EXCEPTIONS_HPP
