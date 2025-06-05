#ifndef FILE_UTILS_HPP
#define FILE_UTILS_HPP

#include <string>
#include <Eigen/Dense>
#include "model/parameters/SEPAIHRDParameters.hpp"

using namespace Eigen;

/**
 * @namespace FileUtils
 * @brief Contains utilities for file and directory operations in the epidemic simulation system.
 */
namespace FileUtils {
    /**
     * @brief Ensures the specified directory exists, creating it if necessary.
     * @param path [in] Directory path to check/create
     * @return true if the directory exists or was successfully created, false otherwise
     */
    bool ensureDirectoryExists(const std::string& path);
    
    /**
     * @brief Locates and returns the project root directory.
     * @details Searches for data, include, and src directories to identify the root.
     * @return Path to the project root directory as a string
     */
    std::string getProjectRoot();
    
    /**
     * @brief Constructs a path to the output directory with optional filename.
     * @details Creates the output directory if it doesn't exist.
     * @param filename [in] Optional filename to append to the output path (empty by default)
     * @return Full path to the output directory or file as a string
     */
    std::string getOutputPath(const std::string& filename = "");

    /**
     * @brief Joins two path segments using the proper path separator.
     * @param path1 [in] First path segment
     * @param path2 [in] Second path segment
     * @return Combined path as a string
     */
    std::string joinPaths(const std::string& path1, const std::string& path2);

    /**
     * @brief Reads SEPAIHRD model parameters from a specified file.
     * @details The file should contain parameter names followed by their values.
     * Age-specific parameters should have values for each age class.
     * Lines starting with '#' are treated as comments and ignored.
     * 
     * @param filename [in] Path to the parameter file
     * @param num_age_classes [in] The number of age classes expected for age-specific parameters
     * @return A SEPAIHRDParameters struct populated with values from the file
     * @throws epidemic::FileIOException If the file cannot be opened
     * @throws epidemic::DataFormatException If the file format is invalid or data is missing/incorrect
     */
    epidemic::SEPAIHRDParameters readSEPAIHRDParameters(const std::string& filename, int num_age_classes);
}

#endif