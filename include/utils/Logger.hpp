#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <mutex>
#include <sstream>

namespace epidemic {

/**
 * @enum LogLevel
 * @brief Defines severity levels for log messages.
 */
enum class LogLevel {
    DEBUG,    ///< Detailed debugging information.
    INFO,     ///< General informational messages.
    WARNING,  ///< Indicates potential issues.
    ERROR,    ///< Errors hindering specific operations.
    FATAL     ///< Critical errors halting the program.
};

/**
 * @class Logger
 * @brief A thread-safe singleton logger for the application.
 *
 * Provides centralized logging to console and optionally to a file.
 * Messages are timestamped and categorized by severity level and source.
 * Supports filtering messages based on a minimum log level.
 */
class Logger {
public:
    /**
     * @brief Retrieves the singleton instance of the Logger.
     * @return Logger& Reference to the unique logger instance.
     */
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    /**
     * @brief Sets the minimum severity level for messages to be processed.
     *
     * Messages with a level below this setting will be ignored.
     * @param level [in] The minimum LogLevel to output.
     */
    void setLogLevel(LogLevel level) {
        std::lock_guard<std::mutex> lock(mutex_);
        logLevel_ = level;
    }

    /**
     * @brief Configures file logging.
     *
     * Enables or disables logging to a specified file. If enabling and the file
     * is already open, it will be closed and reopened (potentially truncating or appending
     * based on default behavior, which is appending here).
     *
     * @param enable   [in] True to enable file logging, false to disable.
     * @param filename [in] The path to the log file (used only if enable is true). Defaults to "epidemic_model.log".
     * @return bool True if the requested state (enabled/disabled with file open/closed) was achieved, false on failure (e.g., cannot open file).
     */
    bool enableFileLogging(bool enable, const std::string& filename = "epidemic_model.log") {
        std::lock_guard<std::mutex> lock(mutex_);
        if (enable) {
            if (logFile_.is_open()) {
                logFile_.close(); // Close existing file first
            }
            // Open in append mode
            logFile_.open(filename, std::ios::app);
            if (!logFile_.is_open()) {
                // Optionally log an error to console even if file logging failed
                std::cerr << formatLogMessage(LogLevel::ERROR, "Logger", "Failed to open log file: " + filename) << std::endl;
                return false;
            }
            log(LogLevel::INFO, "Logger", "File logging enabled to: " + filename); // Log enabling action
            return true;
        } else {
            if (logFile_.is_open()) {
                log(LogLevel::INFO, "Logger", "File logging disabled."); // Log disabling action before closing
                logFile_.close();
            }
            return true; // Disabling is always considered successful
        }
    }

    /**
     * @brief Logs a message if its level meets the minimum threshold.
     *
     * This is the core logging method. It formats the message and outputs
     * it to the console and/or file based on current settings. Thread-safe.
     *
     * @param level   [in] The severity level of the message.
     * @param source  [in] Identifier for the source of the message (e.g., class name, function name).
     * @param message [in] The content of the log message.
     */
    void log(LogLevel level, const std::string& source, const std::string& message) {
        // Check level early before locking
        if (level < logLevel_) return;

        // Format message once
        std::string formattedMessage = formatLogMessage(level, source, message);

        // Lock only during output operations
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << formattedMessage << std::endl;
        if (logFile_.is_open()) {
            logFile_ << formattedMessage << std::endl;
            // Optionally flush the file buffer
            // logFile_.flush();
        }
    }

    /** @brief Logs a message with DEBUG level. @param source Source identifier. @param message Message content. */
    void debug(const std::string& source, const std::string& message)   { log(LogLevel::DEBUG, source, message); }
    /** @brief Logs a message with INFO level. @param source Source identifier. @param message Message content. */
    void info(const std::string& source, const std::string& message)    { log(LogLevel::INFO, source, message); }
    /** @brief Logs a message with WARNING level. @param source Source identifier. @param message Message content. */
    void warning(const std::string& source, const std::string& message) { log(LogLevel::WARNING, source, message); }
    /** @brief Logs a message with ERROR level. @param source Source identifier. @param message Message content. */
    void error(const std::string& source, const std::string& message)   { log(LogLevel::ERROR, source, message); }
    /** @brief Logs a message with FATAL level. @param source Source identifier. @param message Message content. */
    void fatal(const std::string& source, const std::string& message)   { log(LogLevel::FATAL, source, message); }

private:
    // Private constructor to enforce singleton pattern.
    Logger() : logLevel_(LogLevel::INFO) {}

    // Prevent copying and assignment.
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    /**
     * @brief Formats a log entry with timestamp, level, source, and message.
     * @param level   [in] The severity level.
     * @param source  [in] The source identifier.
     * @param message [in] The message content.
     * @return std::string The fully formatted log string.
     */
    std::string formatLogMessage(LogLevel level, const std::string& source, const std::string& message) {
        std::ostringstream oss;
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        oss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S") << " ";

        switch (level) {
            case LogLevel::DEBUG:   oss << "[DEBUG]  "; break;
            case LogLevel::INFO:    oss << "[INFO]   "; break;
            case LogLevel::WARNING: oss << "[WARNING]"; break;
            case LogLevel::ERROR:   oss << "[ERROR]  "; break;
            case LogLevel::FATAL:   oss << "[FATAL]  "; break;
        }

        oss << " [" << source << "] " << message;
        return oss.str();
    }

    LogLevel logLevel_;         ///< Minimum level for messages to be processed.
    std::ofstream logFile_;     ///< Output file stream (if file logging is enabled).
    std::mutex mutex_;          ///< Ensures thread safety for log operations.
};

} // namespace epidemic

#endif // LOGGER_H