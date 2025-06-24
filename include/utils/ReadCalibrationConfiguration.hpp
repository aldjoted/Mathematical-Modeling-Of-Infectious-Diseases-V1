#ifndef READCALIBRATIONCONFIGURATION_HPP
#define READCALIBRATIONCONFIGURATION_HPP

#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <utility>
#include <iomanip>
#include "model/parameters/SEPAIHRDParameters.hpp"

/**
 * @brief Reads initial SEPAIHRD parameters from a configuration file.
 *
 * The configuration file contains scalar parameters as well as age-specific vector parameters
 * (p, h, icu, d_H, d_ICU) and extended fields for NPI strategy (kappa_end_times, kappa_values).
 *
 * @param filename Path to the initial guess configuration file.
 * @param num_age_classes Number of age groups.
 * @return epidemic::SEPAIHRDParameters Structure with model parameters.
 *
 * @throws std::runtime_error if the file cannot be opened or if a line is formatted incorrectly.
 */
epidemic::SEPAIHRDParameters readSEPAIHRDParameters(const std::string &filename, int num_age_classes);

/**
 * @brief Reads parameter bounds from a text file.
 *
 * Each non-empty line in the file should contain:
 * <param_name> <lower_bound> <upper_bound>
 * Lines starting with '#' are ignored.
 *
 * @param filename Path to the parameter bounds file.
 * @return std::map<std::string, std::pair<double, double>> Map of parameter names to (lower_bound, upper_bound) pairs.
 *
 * @throws std::runtime_error if the file cannot be opened or if a line is formatted incorrectly.
 */
std::map<std::string, std::pair<double, double>> readParamBounds(const std::string &filename);

/**
 * @brief Reads proposal sigmas from a text file.
 *
 * Each non-empty line in the file should contain:
 * <param_name> <sigma>
 * Lines starting with '#' are ignored.
 *
 * @param filename Path to the proposal sigmas file.
 * @return std::map<std::string, double> Map of parameter names to sigma values.
 *
 * @throws std::runtime_error if the file cannot be opened or if a line is formatted incorrectly.
 */
std::map<std::string, double> readProposalSigmas(const std::string &filename);

/**
 * @brief Reads a list of parameters to calibrate from a text file.
 *
 * Each non-empty line should contain a single parameter name.
 * Lines starting with '#' are ignored.
 *
 * @param filename Path to the parameters-to-calibrate file.
 * @return std::vector<std::string> Vector of parameter names.
 *
 * @throws std::runtime_error if the file cannot be opened.
 */
std::vector<std::string> readParamsToCalibrate(const std::string &filename);

/**
 * @brief Saves the best calibration parameters to a file.
 *
 * Writes the best parameter values after calibration in a format compatible with
 * readSEPAIHRDParameters. Includes metadata about the calibration process.
 *
 * @param filename Path to save the calibrated parameters.
 * @param parameters The complete set of model parameters.
 * @param calibrated_param_values_map Map of calibrated parameter names to their final values.
 * @param actual_calibrated_param_names Vector of names for parameters that were actually part of the calibration set.
 * @param obj_value The best objective function value achieved.
 * @param timestamp Optional timestamp for when calibration was completed.
 *
 * @throws std::runtime_error if the file cannot be opened or written.
 */
void saveCalibrationResults(const std::string &filename, 
                          const epidemic::SEPAIHRDParameters &parameters,
                          const std::vector<std::string>& actual_calibrated_param_names, // Vector of names for parameters that were actually part of the calibration set.
                          double obj_value,
                          const std::string &timestamp = "");

/**
 * @brief Reads Metropolis-Hastings sampler settings from a text file.
 *
 * Each non-empty line in the file should contain:
 * <setting_name> <value>
 * Lines starting with '#' are ignored.
 *
 * @param filename Path to the MCMC settings file.
 * @return std::map<std::string, double> Map of setting names to values.
 *
 * @throws std::runtime_error if the file cannot be opened or if a line is formatted incorrectly.
 */
std::map<std::string, double> readMetropolisHastingsSettings(const std::string &filename);

/**
 * @brief Reads Hill Climbing optimizer settings from a text file.
 *
 * Each non-empty line in the file should contain:
 * <setting_name> <value>
 * Lines starting with '#' are ignored.
 *
 * @param filename Path to the Hill Climbing settings file.
 * @return std::map<std::string, double> Map of setting names to values.
 *
 * @throws std::runtime_error if the file cannot be opened or if a line is formatted incorrectly.
 */
std::map<std::string, double> readHillClimbingSettings(const std::string &filename);

/**
 * @brief Reads Particle Swarm Optimizer settings from a text file.
 *
 * Each non-empty line in the file should contain:
 * <setting_name> <value>
 * Lines starting with '#' are ignored.
 *
 * @param filename Path to the PSO settings file.
 * @return std::map<std::string, double> Map of setting names to values.
 *
 * @throws std::runtime_error if the file cannot be opened or if a line is formatted incorrectly.
 */
std::map<std::string, double> readParticleSwarmSettings(const std::string &filename);

#endif // READCALIBRATIONCONFIGURATION_HPP