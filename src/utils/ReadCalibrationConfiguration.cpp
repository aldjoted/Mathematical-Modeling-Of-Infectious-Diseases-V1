#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <map>
#include <ctime>
#include <algorithm>
#include "utils/ReadCalibrationConfiguration.hpp"
#include "utils/Logger.hpp"
#include "exceptions/Exceptions.hpp"
#include "model/parameters/SEPAIHRDParameters.hpp"

static void parseValue(std::istringstream& iss, double& scalar_val, std::vector<double>& vector_val, bool& is_vector) {
    vector_val.clear();
    double value;
    is_vector = false;
    if (!(iss >> value)) return;
    vector_val.push_back(value);
    while (iss >> value) {
        is_vector = true;
        vector_val.push_back(value);
    }
    if (!is_vector) {
        scalar_val = vector_val[0];
    }
}

static void assignAgeVector(const std::string& name, const std::vector<double>& values, epidemic::SEPAIHRDParameters& params, int num_age_classes) {
    if (static_cast<int>(values.size()) != num_age_classes) {
        throw epidemic::DataFormatException("readSEPAIHRDParameters", "Incorrect number of values for " + name + ". Expected " + std::to_string(num_age_classes) + ", got " + std::to_string(values.size()));
    }
    
    Eigen::VectorXd vec = Eigen::Map<const Eigen::Vector<double, Eigen::Dynamic>>(values.data(), values.size());

    if      (name == "a") params.a = vec;
    else if (name == "h_infec") params.h_infec = vec;
    else if (name == "p") params.p = vec;
    else if (name == "h") params.h = vec;
    else if (name == "icu") params.icu = vec;
    else if (name == "d_H") params.d_H = vec;
    else if (name == "d_ICU") params.d_ICU = vec;
}


// --- Main Functions ---

void saveCalibrationResults(const std::string &filename,
                          const epidemic::SEPAIHRDParameters& meters,
                          const std::vector<std::string>& actual_calibrated_param_names,
                          double obj_value,
                          const std::string& tamp_str) {

    std::ofstream file(filename);
    if (!file.is_open()) {
        epidemic::Logger::getInstance().error("saveCalibrationResults", "Unable to open file for writing: " + filename);
        throw epidemic::FileIOException("saveCalibrationResults", "Unable to open file for writing: " + filename);
    }

    std::string ts = tamp_str;
    if (ts.empty()) {
        std::time_t now = std::time(nullptr);
        char mbstr[100];
        if (std::strftime(mbstr, sizeof(mbstr), "%Y-%m-%d %H:%M:%S", std::localtime(&now))) {
            ts = mbstr;
        } else {
            ts = "TIMESTAMP_ERROR";
        }
    }

    file << "# Calibrated SEPAIHRD Model Parameters" << std::endl;
    file << "# Calibration completed: " << ts << std::endl;
    file << "# Best objective function value: " << std::scientific << std::setprecision(8) << obj_value << std::endl;
    file << "# Calibrated parameters are marked with [C] if they were part of the calibration set." << std::endl;
    file << std::endl;

    file << "# --- Transmission Parameters ---" << std::endl;
    auto writeScalarParam = [&](const std::string& name, double value) {
        bool calibrated = std::find(actual_calibrated_param_names.begin(), actual_calibrated_param_names.end(), name) != actual_calibrated_param_names.end();
        file << name << " " << std::scientific << std::setprecision(8) << value;
        if (calibrated) file << " # [C]";
        file << std::endl;
    };
    
    // Write out the time-varying beta schedule
    file << "beta_end_times";
    for (double t : meters.beta_end_times) file << " " << std::fixed << std::setprecision(1) << t;
    file << std::endl;

    for (size_t i = 0; i < meters.beta_values.size(); ++i) {
        std::string beta_name = "beta_" + std::to_string(i + 1);
        writeScalarParam(beta_name, meters.beta_values[i]);
    }
    
    writeScalarParam("beta", meters.beta); // Also write the constant beta for reference
    writeScalarParam("theta", meters.theta);

    file << std::endl << "# --- Disease Progression Rates ---" << std::endl;
    writeScalarParam("sigma", meters.sigma);
    writeScalarParam("gamma_p", meters.gamma_p);
    writeScalarParam("gamma_A", meters.gamma_A);
    writeScalarParam("gamma_I", meters.gamma_I);
    writeScalarParam("gamma_H", meters.gamma_H);
    writeScalarParam("gamma_ICU", meters.gamma_ICU);
    
    file << std::endl << "# --- Age-specific Parameters ---" << std::endl;
    auto writeAgeVectorParam = [&](const std::string& base_name, const Eigen::VectorXd& values) {
        file << base_name;
        for (int i = 0; i < values.size(); ++i) {
            file << " " << std::scientific << std::setprecision(8) << values(i);
        }
        // Check if any of the vector elements were calibrated
        bool any_calibrated = false;
        for (int i = 0; i < values.size(); ++i) {
             if (std::find(actual_calibrated_param_names.begin(), actual_calibrated_param_names.end(), base_name + "_" + std::to_string(i)) != actual_calibrated_param_names.end()) {
                any_calibrated = true;
                break;
            }
        }
        if (any_calibrated) file << " # [C]";
        file << std::endl;
    };

    writeAgeVectorParam("p", meters.p);
    writeAgeVectorParam("a", meters.a);
    writeAgeVectorParam("h_infec", meters.h_infec);
    writeAgeVectorParam("h", meters.h);
    writeAgeVectorParam("icu", meters.icu);
    writeAgeVectorParam("d_H", meters.d_H);
    writeAgeVectorParam("d_ICU", meters.d_ICU);

    file << std::endl << "# --- Initial State Multipliers ---" << std::endl;
    writeScalarParam("E0_multiplier", meters.E0_multiplier);
    writeScalarParam("P0_multiplier", meters.P0_multiplier);
    writeScalarParam("A0_multiplier", meters.A0_multiplier);
    writeScalarParam("I0_multiplier", meters.I0_multiplier);
    writeScalarParam("H0_multiplier", meters.H0_multiplier);
    writeScalarParam("ICU0_multiplier", meters.ICU0_multiplier);
    writeScalarParam("R0_multiplier", meters.R0_multiplier);
    writeScalarParam("D0_multiplier", meters.D0_multiplier);

    file << std::endl << "# --- NPI Strategy Parameters ---" << std::endl;
    file << "kappa_end_times";
    for (double t : meters.kappa_end_times) file << " " << std::fixed << std::setprecision(1) << t;
    file << std::endl;

    for (size_t i = 0; i < meters.kappa_values.size(); ++i) {
        std::string kappa_name = "kappa_" + std::to_string(i + 1);
        writeScalarParam(kappa_name, meters.kappa_values[i]);
    }

    file.close();
    epidemic::Logger::getInstance().info("saveCalibrationResults", "Calibration results saved to: " + filename);
}

epidemic::SEPAIHRDParameters readSEPAIHRDParameters(const std::string& filename, int num_age_classes) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        epidemic::Logger::getInstance().error("readSEPAIHRDParameters", "Unable to open parameters file: " + filename);
        throw epidemic::FileIOException("readSEPAIHRDParameters", "Unable to open parameters file: " + filename);
    }

    epidemic::SEPAIHRDParameters params;
    // Pre-size age-specific vectors to avoid errors if not present in the file
    params.a = Eigen::VectorXd::Zero(num_age_classes);
    params.h_infec = Eigen::VectorXd::Zero(num_age_classes);
    params.p = Eigen::VectorXd::Zero(num_age_classes);
    params.h = Eigen::VectorXd::Zero(num_age_classes);
    params.icu = Eigen::VectorXd::Zero(num_age_classes);
    params.d_H = Eigen::VectorXd::Zero(num_age_classes);
    params.d_ICU = Eigen::VectorXd::Zero(num_age_classes);

    std::string line;
    int line_number = 0;

    std::map<int, double> temp_beta_map, temp_kappa_map;

    while (std::getline(file, line)) {
        line_number++;

        line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
        line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string param_name;
        if (!(iss >> param_name)) {
            epidemic::Logger::getInstance().warning("readSEPAIHRDParameters", "Could not read parameter name on line " + std::to_string(line_number));
            continue;
        }

        double scalar_val = 0.0;
        std::vector<double> vector_val;
        bool is_vector = false;
        parseValue(iss, scalar_val, vector_val, is_vector);

        if (vector_val.empty()) {
             epidemic::Logger::getInstance().warning("readSEPAIHRDParameters", "No value found for parameter '" + param_name + "' on line " + std::to_string(line_number));
             continue;
        }

        if (param_name.rfind("beta_", 0) == 0 && param_name != "beta_end_times") {
            try {
                int idx = std::stoi(param_name.substr(5));
                temp_beta_map[idx] = scalar_val;
            } catch (const std::exception& e) {
                epidemic::Logger::getInstance().warning("readSEPAIHRDParameters", 
                    "Could not parse beta index from parameter: " + param_name + " on line " + std::to_string(line_number));
            }
        } else if (param_name.rfind("kappa_", 0) == 0 && param_name != "kappa_end_times") {
            try {
                int idx = std::stoi(param_name.substr(6));
                temp_kappa_map[idx] = scalar_val;
            } catch (const std::exception& e) {
                epidemic::Logger::getInstance().warning("readSEPAIHRDParameters", 
                    "Could not parse kappa index from parameter: " + param_name + " on line " + std::to_string(line_number));
            }
        } else if (param_name == "beta") params.beta = scalar_val;
        else if (param_name == "theta") params.theta = scalar_val;
        else if (param_name == "sigma") params.sigma = scalar_val;
        else if (param_name == "gamma_p") params.gamma_p = scalar_val;
        else if (param_name == "gamma_A") params.gamma_A = scalar_val;
        else if (param_name == "gamma_I") params.gamma_I = scalar_val;
        else if (param_name == "gamma_H") params.gamma_H = scalar_val;
        else if (param_name == "gamma_ICU") params.gamma_ICU = scalar_val;
        else if (param_name == "E0_multiplier") params.E0_multiplier = scalar_val;
        else if (param_name == "P0_multiplier") params.P0_multiplier = scalar_val;
        else if (param_name == "A0_multiplier") params.A0_multiplier = scalar_val;
        else if (param_name == "I0_multiplier") params.I0_multiplier = scalar_val;
        else if (param_name == "H0_multiplier") params.H0_multiplier = scalar_val;
        else if (param_name == "ICU0_multiplier") params.ICU0_multiplier = scalar_val;
        else if (param_name == "R0_multiplier") params.R0_multiplier = scalar_val;
        else if (param_name == "D0_multiplier") params.D0_multiplier = scalar_val;
        else if (param_name == "beta_end_times") params.beta_end_times = vector_val;
        else if (param_name == "kappa_end_times") params.kappa_end_times = vector_val;
        else if (param_name == "a" || param_name == "h_infec" || param_name == "p" || param_name == "h" || param_name == "icu" || param_name == "d_H" || param_name == "d_ICU") {
            assignAgeVector(param_name, vector_val, params, num_age_classes);
        } else {
            epidemic::Logger::getInstance().warning("readSEPAIHRDParameters", "Unrecognized parameter '" + param_name + "' on line " + std::to_string(line_number));
        }
    }

    // Assemble beta_values from the map
    if (!temp_beta_map.empty()) {
        params.beta_values.resize(temp_beta_map.rbegin()->first);
        for(const auto& pair : temp_beta_map) {
            params.beta_values[pair.first - 1] = pair.second;
        }
    }
    // Assemble kappa_values from the map
    if (!temp_kappa_map.empty()) {
        params.kappa_values.resize(temp_kappa_map.rbegin()->first);
        for(const auto& pair : temp_kappa_map) {
            params.kappa_values[pair.first - 1] = pair.second;
        }
    }

    file.close();
    return params;
}

std::map<std::string, std::pair<double, double>> readParamBounds(const std::string &filename) {
    std::map<std::string, std::pair<double, double>> bounds;
    std::ifstream file(filename);
    if (!file.is_open()) {
        epidemic::Logger::getInstance().error("ReadCalibrationConfiguration::readParamBounds", "Error opening param bounds file: " + filename);
        throw epidemic::FileIOException("readParamBounds", "Error opening param bounds file: " + filename);
    }
    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        line_number++;
        line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
        line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);
        if (line.empty() || line[0]=='#') continue;

        std::istringstream iss(line);
        std::string param;
        double low, high;
        if (!(iss >> param >> low >> high)) {
            epidemic::Logger::getInstance().error("ReadCalibrationConfiguration::readParamBounds", "Invalid line in bounds file (line " + std::to_string(line_number) + "): " + line);
            throw epidemic::DataFormatException("readParamBounds", "Invalid line in bounds file: " + line);
        }
        std::string extra;
        if (iss >> extra) {
            epidemic::Logger::getInstance().error("ReadCalibrationConfiguration::readParamBounds", "Too many values on line in bounds file (line " + std::to_string(line_number) + "): " + line);
            throw epidemic::DataFormatException("readParamBounds", "Too many values on line in bounds file: " + line);
        }
        bounds[param] = std::make_pair(low, high);
    }
    file.close();
    return bounds;
}


std::map<std::string, double> readProposalSigmas(const std::string &filename) {
    std::map<std::string, double> sigmas;
    std::ifstream file(filename);
    if (!file.is_open()) {
        epidemic::Logger::getInstance().error("ReadCalibrationConfiguration::readProposalSigmas", "Error opening proposal sigmas file: " + filename);
        throw epidemic::FileIOException("readProposalSigmas","Error opening proposal sigmas file: " + filename);
    }
    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        line_number++;
        line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
        line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);
        if (line.empty() || line[0]=='#') continue;

        std::istringstream iss(line);
        std::string param;
        double sigma_val;
        if (!(iss >> param >> sigma_val)) {
            epidemic::Logger::getInstance().error("ReadCalibrationConfiguration::readProposalSigmas", "Invalid line in proposal sigmas file (line " + std::to_string(line_number) + "): " + line);
            throw epidemic::DataFormatException("readProposalSigmas", "Invalid line in proposal sigmas file: " + line);
        }
        std::string extra;
        if (iss >> extra) {
             epidemic::Logger::getInstance().error("ReadCalibrationConfiguration::readProposalSigmas", "Too many values on line in sigmas file (line " + std::to_string(line_number) + "): " + line);
            throw epidemic::DataFormatException("readProposalSigmas", "Too many values on line in sigmas file: " + line);
        }
        sigmas[param] = sigma_val;
    }
    file.close();
    return sigmas;
}


std::vector<std::string> readParamsToCalibrate(const std::string &filename) {
    std::vector<std::string> params;
    std::ifstream file(filename);
    if (!file.is_open()) {
        epidemic::Logger::getInstance().error("ReadCalibrationConfiguration::readParamsToCalibrate", "Error opening params_to_calibrate file: " + filename);
        throw epidemic::FileIOException("readParamsToCalibrate","Error opening params_to_calibrate file: " + filename);
    }
    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        line_number++;
        line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
        line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);
        if (line.empty() || line[0]=='#') continue;

        std::string param_name_read;
        std::istringstream iss(line);
        if (!(iss >> param_name_read)) {
             epidemic::Logger::getInstance().warning("ReadCalibrationConfiguration::readParamsToCalibrate", "Empty or unreadable parameter name on line " + std::to_string(line_number) + " in params_to_calibrate file.");
            continue;
        }
        std::string extra;
        if (iss >> extra) {
             epidemic::Logger::getInstance().warning("ReadCalibrationConfiguration::readParamsToCalibrate", "Extra content found after parameter name '" + param_name_read + "' on line " + std::to_string(line_number) + " in params_to_calibrate file. Using only first word.");
        }
        params.push_back(param_name_read);
    }
    file.close();
    return params;
}

// Generic helper function to read settings map
static std::map<std::string, double> readSettingsFile(const std::string& filename, const std::string& calling_function_name) {
    std::map<std::string, double> settings;
    std::ifstream file(filename);
    if (!file.is_open()) {
        epidemic::Logger::getInstance().error("ReadCalibrationConfiguration::" + calling_function_name, "Error opening settings file: " + filename);
        throw epidemic::FileIOException(calling_function_name, "Error opening settings file: " + filename);
    }
    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        line_number++;
        line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
        line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string setting_name;
        double value;
        if (!(iss >> setting_name >> value)) {
            epidemic::Logger::getInstance().error("ReadCalibrationConfiguration::" + calling_function_name, "Invalid line in settings file (line " + std::to_string(line_number) + "): " + line);
            throw epidemic::DataFormatException(calling_function_name, "Invalid line in settings file: " + line);
        }
        std::string extra;
        if (iss >> extra) {
            epidemic::Logger::getInstance().error("ReadCalibrationConfiguration::" + calling_function_name, "Too many values on line in settings file (line " + std::to_string(line_number) + "): " + line);
            throw epidemic::DataFormatException(calling_function_name, "Too many values on line in settings file: " + line);
        }
        settings[setting_name] = value;
    }
    file.close();
    epidemic::Logger::getInstance().info("ReadCalibrationConfiguration::" + calling_function_name, "Successfully read " + std::to_string(settings.size()) + " settings from " + filename);
    return settings;
}

std::map<std::string, double> readMetropolisHastingsSettings(const std::string &filename) {
    return readSettingsFile(filename, "readMetropolisHastingsSettings");
}

std::map<std::string, double> readHillClimbingSettings(const std::string &filename) {
    return readSettingsFile(filename, "readHillClimbingSettings");
}

std::map<std::string, double> readParticleSwarmSettings(const std::string &filename) {
    return readSettingsFile(filename, "readParticleSwarmSettings");
}

std::map<std::string, double> readNUTSSettings(const std::string &filename) {
    return readSettingsFile(filename, "readNUTSSettings");
}