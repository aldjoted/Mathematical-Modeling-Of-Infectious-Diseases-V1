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

void saveCalibrationResults(const std::string &filename,
                          const epidemic::SEPAIHRDParameters &parameters,
                          const std::map<std::string, double> &calibrated_param_values_map,
                          const std::vector<std::string>& actual_calibrated_param_names,
                          double obj_value,
                          const std::string &timestamp_str) {
    (void)calibrated_param_values_map;

    std::ofstream file(filename);
    if (!file.is_open()) {
        epidemic::Logger::getInstance().error("saveCalibrationResults", "Unable to open file for writing: " + filename);
        throw epidemic::FileIOException("saveCalibrationResults", "Unable to open file for writing: " + filename);
    }

    std::string ts = timestamp_str;
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

    file << "# Scalar parameters" << std::endl;

    auto writeScalarParam = [&](const std::string& name, double value) {
        bool calibrated = std::find(actual_calibrated_param_names.begin(), actual_calibrated_param_names.end(), name) != actual_calibrated_param_names.end();
        file << name << " " << std::scientific << std::setprecision(6) << value;
        if (calibrated) file << " # [C]";
        file << std::endl;
    };

    writeScalarParam("beta", parameters.beta);
    writeScalarParam("theta", parameters.theta);
    writeScalarParam("sigma", parameters.sigma);
    writeScalarParam("gamma_p", parameters.gamma_p);
    writeScalarParam("gamma_A", parameters.gamma_A);
    writeScalarParam("gamma_I", parameters.gamma_I);
    writeScalarParam("gamma_H", parameters.gamma_H);
    writeScalarParam("gamma_ICU", parameters.gamma_ICU);
    writeScalarParam("contact_matrix_scaling_factor", parameters.contact_matrix_scaling_factor);

    file << std::endl << "# Age-specific parameters" << std::endl;

    auto writeAgeVectorParam = [&](const std::string& base_name, const Eigen::VectorXd& values) {
        for (int i = 0; i < values.size(); ++i) {
            std::string full_param_name = base_name + "_" + std::to_string(i);
            bool calibrated = std::find(actual_calibrated_param_names.begin(), actual_calibrated_param_names.end(), full_param_name) != actual_calibrated_param_names.end();
            file << full_param_name << " " << std::scientific << std::setprecision(6) << values(i);
            if (calibrated) file << " # [C]";
            file << std::endl;
        }
    };

    writeAgeVectorParam("p", parameters.p);
    writeAgeVectorParam("h", parameters.h);
    writeAgeVectorParam("icu", parameters.icu);
    writeAgeVectorParam("d_H", parameters.d_H);
    writeAgeVectorParam("d_ICU", parameters.d_ICU);

    file << std::endl << "# NPI strategy parameters" << std::endl;

    file << "kappa_end_times";
    for (size_t i = 0; i < parameters.kappa_end_times.size(); ++i) {
        file << " " << std::fixed << std::setprecision(1) << parameters.kappa_end_times[i];
    }
    file << std::endl;

    if (parameters.kappa_values.size() >= 1) {
        for (size_t i = 0; i < parameters.kappa_values.size(); ++i) {
            std::string kappa_name = "kappa_" + std::to_string(i + 1);
            bool calibrated = std::find(actual_calibrated_param_names.begin(), actual_calibrated_param_names.end(), kappa_name) != actual_calibrated_param_names.end();
            file << kappa_name << " " << std::scientific << std::setprecision(6) << parameters.kappa_values[i];
            if (calibrated) file << " # [C]";
            file << std::endl;
        }
    }

    file.close();
    epidemic::Logger::getInstance().info("saveCalibrationResults", "Calibration results saved to: " + filename);
}

epidemic::SEPAIHRDParameters readSEPAIHRDParameters(const std::string& filename, int num_age_classes) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        epidemic::Logger::getInstance().error("readSEPAIHRDParameters", "Unable to open parameters file: " + filename);
        throw std::runtime_error("Unable to open parameters file: " + filename);
    }

    epidemic::SEPAIHRDParameters params;

    params.p = Eigen::VectorXd(num_age_classes);
    params.h = Eigen::VectorXd(num_age_classes);
    params.icu = Eigen::VectorXd(num_age_classes);
    params.d_H = Eigen::VectorXd(num_age_classes);
    params.d_ICU = Eigen::VectorXd(num_age_classes);

    params.N = Eigen::VectorXd::Zero(num_age_classes);
    params.M_baseline = Eigen::MatrixXd::Zero(num_age_classes, num_age_classes);

    std::string line;
    int line_number = 0;
    std::vector<std::string> age_vector_params = {"p", "h", "icu", "d_H", "d_ICU"};

    while (std::getline(file, line)) {
        line_number++;

        line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
        line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);

        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string param_name;
        if (!(iss >> param_name)) {
            epidemic::Logger::getInstance().error("readSEPAIHRDParameters", "Could not read parameter name on line " + std::to_string(line_number) + ": " + line);
            continue;
        }

        bool is_age_vector = false;
        for(const auto& name : age_vector_params) {
            if (param_name == name) {
                is_age_vector = true;
                break;
            }
        }
        if (is_age_vector) {
            Eigen::VectorXd* vec_ptr = nullptr;
            if (param_name == "p") vec_ptr = &params.p;
            else if (param_name == "h") vec_ptr = &params.h;
            else if (param_name == "icu") vec_ptr = &params.icu;
            else if (param_name == "d_H") vec_ptr = &params.d_H;
            else if (param_name == "d_ICU") vec_ptr = &params.d_ICU;
            else {
                throw std::runtime_error("Internal error reading age vector on line " + std::to_string(line_number));
            }
            for (int i = 0; i < num_age_classes; ++i) {
                double value;
                if (!(iss >> value)) {
                    throw std::runtime_error("Error reading value for age class " + std::to_string(i) +
                                               " of parameter '" + param_name + "' on line " + std::to_string(line_number));
                }
                (*vec_ptr)(i) = value;
            }
            double extra;
            if (iss >> extra) {
                throw std::runtime_error("Too many values provided for age-specific parameter '" + param_name +
                                           "' on line " + std::to_string(line_number) + ". Expected " + std::to_string(num_age_classes) + " values.");
            }
        } else if (param_name == "kappa_end_times") {
            double value;
            while (iss >> value) {
                params.kappa_end_times.push_back(value);
            }
            if (params.kappa_end_times.empty()) {
                throw std::runtime_error("No values provided for kappa_end_times on line " + std::to_string(line_number));
            }
        } else if (param_name == "kappa_values") {
            double value;
            while (iss >> value) {
                params.kappa_values.push_back(value);
            }
            if (params.kappa_values.empty()) {
                throw std::runtime_error("No values provided for kappa_values on line " + std::to_string(line_number));
            }
        } else {

            double value;
            if (!(iss >> value)) {
                throw std::runtime_error("Error reading scalar value for parameter '" + param_name +
                                           "' on line " + std::to_string(line_number));
            }
            if (param_name == "beta") params.beta = value;
            else if (param_name == "theta") params.theta = value;
            else if (param_name == "sigma") params.sigma = value;
            else if (param_name == "gamma_p") params.gamma_p = value;
            else if (param_name == "gamma_A") params.gamma_A = value;
            else if (param_name == "gamma_I") params.gamma_I = value;
            else if (param_name == "gamma_H") params.gamma_H = value;
            else if (param_name == "gamma_ICU") params.gamma_ICU = value;
            else if (param_name == "contact_matrix_scaling_factor") params.contact_matrix_scaling_factor = value;
            else {
                epidemic::Logger::getInstance().warning("readSEPAIHRDParameters", "Unrecognized parameter name '" + param_name +
                                                       "' on line " + std::to_string(line_number) + ". Ignoring.");
            }
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