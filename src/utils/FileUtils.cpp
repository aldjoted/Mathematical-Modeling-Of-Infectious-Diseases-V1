#include "utils/FileUtils.hpp"
#include "exceptions/Exceptions.hpp"
#include <iostream>
#include <filesystem>
#include <vector>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace FileUtils {

bool ensureDirectoryExists(const std::string& path) {
    try {
        if (!fs::exists(path)) {
            return fs::create_directories(path);
        }
        return true;
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error creating directory: " << e.what() << std::endl;
        return false;
    }
}

std::string getProjectRoot() {
    std::string currentDir = fs::current_path().string();
    std::vector<std::string> possibleRoots = { currentDir };

    fs::path current(currentDir);
    for (int i = 0; i < 5; i++) {
        current = current.parent_path();
        if (!current.empty()) {
            possibleRoots.push_back(current.string());
        }
    }

    for (const auto& root : possibleRoots) {
        if (fs::exists(root + "/data") &&
            fs::exists(root + "/include") &&
            fs::exists(root + "/src")) {
            return fs::absolute(fs::path(root)).lexically_normal().string();
        }
    }

    return fs::absolute(fs::path(currentDir)).lexically_normal().string();
}

std::string getOutputPath(const std::string& filename) {
    std::string projectRoot = getProjectRoot();
    fs::path outputDir = fs::path(projectRoot) / "data" / "output";

    if (!ensureDirectoryExists(outputDir.string())) {
        std::cerr << "Warning: Could not create output directory: " << outputDir << std::endl;
    }

    if (filename.empty()) {
        return outputDir.lexically_normal().string();
    } else {
        return (outputDir / filename).lexically_normal().string();
    }
}

std::string joinPaths(const std::string& path1, const std::string& path2) {
    if(path2.empty()){
        return path1;
    }
    std::string rel = path2;
    if (!rel.empty() && rel[0] == '/') {
        rel = rel.substr(1);
    }
    fs::path p = fs::path(path1) / fs::path(rel);
    return p.lexically_normal().string();
}

epidemic::SEPAIHRDParameters readSEPAIHRDParameters(const std::string& filename, int num_age_classes) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw epidemic::FileIOException("FileUtils::readSEPAIHRDParameters", "Unable to open parameters file: " + filename);
    }

    epidemic::SEPAIHRDParameters params;
    params.N = VectorXd::Zero(num_age_classes);
    params.M_baseline = Eigen::MatrixXd::Zero(num_age_classes, num_age_classes);
    params.p = VectorXd::Zero(num_age_classes);
    params.h = VectorXd::Zero(num_age_classes);
    params.icu = VectorXd::Zero(num_age_classes);
    params.d_H = VectorXd::Zero(num_age_classes);
    params.d_ICU = VectorXd::Zero(num_age_classes);

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
             std::cerr << "[Warning] FileUtils::readSEPAIHRDParameters: Could not read parameter name on line " << line_number << ": " << line << std::endl;
             continue;
        }

        bool is_age_vector = false;
        for (const auto& p_name : age_vector_params) {
            if (param_name == p_name) {
                is_age_vector = true;
                break;
            }
        }

        if (is_age_vector) {
            VectorXd* vec_ptr = nullptr;
            if (param_name == "p") vec_ptr = &params.p;
            else if (param_name == "h") vec_ptr = &params.h;
            else if (param_name == "icu") vec_ptr = &params.icu;
            else if (param_name == "d_H") vec_ptr = &params.d_H;
            else if (param_name == "d_ICU") vec_ptr = &params.d_ICU;
            else {
                 throw epidemic::DataFormatException("FileUtils::readSEPAIHRDParameters", "Internal error reading age vector on line " + std::to_string(line_number));
            }

            for (int i = 0; i < num_age_classes; ++i) {
                double value;
                if (!(iss >> value)) {
                    throw epidemic::DataFormatException("FileUtils::readSEPAIHRDParameters", "Error reading value for age class " + std::to_string(i) + " of parameter '" + param_name + "' on line " + std::to_string(line_number));
                }
                (*vec_ptr)(i) = value;
            }
            double extra_val;
            if (iss >> extra_val) {
                 throw epidemic::DataFormatException("FileUtils::readSEPAIHRDParameters", "Too many values provided for age-specific parameter '" + param_name + "' on line " + std::to_string(line_number) + ". Expected " + std::to_string(num_age_classes) + " values.");
            }
        } else if (param_name == "kappa_end_times") {
            params.kappa_end_times.clear();
            double value;
            while (iss >> value) {
                params.kappa_end_times.push_back(value);
            }
            if (iss.fail() && !iss.eof()) {
                throw epidemic::DataFormatException("FileUtils::readSEPAIHRDParameters", "Invalid non-numeric data found for 'kappa_end_times' on line " + std::to_string(line_number));
            }
        } else if (param_name == "kappa_values") {
            params.kappa_values.clear();
            double value;
            while (iss >> value) {
                params.kappa_values.push_back(value);
            }
            if (iss.fail() && !iss.eof()) {
                 throw epidemic::DataFormatException("FileUtils::readSEPAIHRDParameters", "Invalid non-numeric data found for 'kappa_values' on line " + std::to_string(line_number));
            }
        } else {
            double value;
            if (!(iss >> value)) {
                std::string remaining_in_line;
                iss.clear();
                iss >> remaining_in_line;
                if (remaining_in_line.empty() && iss.eof()){
                     throw epidemic::DataFormatException("FileUtils::readSEPAIHRDParameters", "Missing scalar value for parameter '" + param_name + "' on line " + std::to_string(line_number));
                } else {
                     throw epidemic::DataFormatException("FileUtils::readSEPAIHRDParameters", "Error reading scalar value for parameter '" + param_name + "' on line " + std::to_string(line_number) + ". Found: '" + remaining_in_line + "'");
                }
            }

            double extra_val_check;
            if (iss >> extra_val_check) {
                 throw epidemic::DataFormatException("FileUtils::readSEPAIHRDParameters", "Too many values provided for scalar parameter '" + param_name + "' on line " + std::to_string(line_number) + ". Expected 1 value.");
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
                 std::cerr << "Warning: Unrecognized parameter name '" << param_name << "' on line " << line_number << ". Ignoring." << std::endl;
            }
        }
    }

    file.close();
    if (!params.kappa_end_times.empty() && !params.kappa_values.empty() &&
        params.kappa_end_times.size() != params.kappa_values.size()) {
        throw epidemic::DataFormatException("FileUtils::readSEPAIHRDParameters",
            "Mismatch between number of kappa_end_times (" + std::to_string(params.kappa_end_times.size()) +
            ") and kappa_values (" + std::to_string(params.kappa_values.size()) + ") read from file: " + filename);
    }
    return params;
}

} // namespace FileUtils