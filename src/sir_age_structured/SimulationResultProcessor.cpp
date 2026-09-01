#include "sir_age_structured/SimulationResultProcessor.hpp"
#include "sir_age_structured/AgeSIRModel.hpp"
#include "exceptions/Exceptions.hpp"
#include <fstream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cctype>
#include <string>

namespace epidemic {

    Eigen::MatrixXd SimulationResultProcessor::getCompartmentData(const SimulationResult& result,
                                                                  const IEpidemicModel& model,
                                                                  const std::string& compartment_in,
                                                                  int num_compartments_per_age)
    {
        if (!result.isValid()) {
            throw InvalidResultException("SimulationResultProcessor::getCompartmentData", "Simulation result object is invalid or empty.");
        }
    
        int state_size = model.getStateSize();
        if (state_size <= 0) {
             throw ModelException("SimulationResultProcessor::getCompartmentData", "Model returned non-positive state size.");
        }
        if (num_compartments_per_age <= 0) {
             THROW_INVALID_PARAM("SimulationResultProcessor::getCompartmentData", "Number of compartments per age group must be positive.");
        }
    
        int num_age_classes = 0;
        if (state_size % num_compartments_per_age != 0) {
             THROW_INVALID_PARAM("SimulationResultProcessor::getCompartmentData",
                                           "Model state size (" + std::to_string(state_size) +
                                           ") is not divisible by the provided number of compartments per age group (" +
                                           std::to_string(num_compartments_per_age) + ").");
        }
        num_age_classes = state_size / num_compartments_per_age;
    
        std::string compartment_lower = compartment_in;
        std::transform(compartment_lower.begin(), compartment_lower.end(), compartment_lower.begin(), ::tolower);
    
        const auto state_names = model.getStateNames();
        if (state_names.size() != static_cast<size_t>(state_size)) {
             throw ModelException("SimulationResultProcessor::getCompartmentData", "Model returned inconsistent number of state names.");
        }
    
        int offset = -1;
        std::string base_compartment_name; 
    
        for (size_t i = 0; i < state_names.size(); ++i) {
            std::string current_name_lower = state_names[i];
            std::transform(current_name_lower.begin(), current_name_lower.end(), current_name_lower.begin(), ::tolower);
            
            if (current_name_lower.rfind(compartment_lower, 0) == 0) {
                if (num_age_classes > 1) {
                    std::string suffix = current_name_lower.substr(compartment_lower.length());
                    if (!suffix.empty() && std::all_of(suffix.begin(), suffix.end(), ::isdigit)) {
                        offset = static_cast<int>(i);
                        break;
                    }
                } else { // num_age_classes == 1
                    if (current_name_lower == compartment_lower) {
                        offset = static_cast<int>(i);
                        break;
                    }
                }
            }
        }
    
        if (offset == -1) {
            THROW_INVALID_PARAM("SimulationResultProcessor::getCompartmentData",
                                            "Could not find compartment '" + compartment_in +
                                            "' in the model's state names with the expected structure.");
        }
    
        // The compartment block [offset, offset + num_age_classes) is loop-invariant,
        // so bounds are validated once here rather than per time point.
        if (offset < 0 || offset + num_age_classes > state_size) {
            throw SimulationException("SimulationResultProcessor::getCompartmentData",
                                      "Internal error: compartment block [" + std::to_string(offset) + ", " +
                                      std::to_string(offset + num_age_classes) + ") out of bounds [0, " +
                                      std::to_string(state_size) + ").");
        }

        Eigen::MatrixXd compartment_result(result.solution.size(), num_age_classes);
        for (size_t t = 0; t < result.solution.size(); ++t) {
            const std::vector<double>& state = result.solution[t];
            if (state.size() != static_cast<size_t>(state_size)) {
                throw SimulationException("SimulationResultProcessor::getCompartmentData",
                                          "Internal error: Solution vector size mismatch at time index " + std::to_string(t) +
                                          ". Expected " + std::to_string(state_size) + ", got " +
                                          std::to_string(state.size()) + ".");
            }
            const double* src = state.data() + offset;
            for (int a = 0; a < num_age_classes; ++a) {
                compartment_result(t, a) = src[a];
            }
        }
        return compartment_result;
    }
    
    void SimulationResultProcessor::saveResultsToCSV(const SimulationResult& result,
                                                     const IEpidemicModel& model,
                                                     const std::string& filename)
    {
        if (!result.isValid()) {
            throw InvalidResultException("SimulationResultProcessor::saveResultsToCSV", "Simulation result object is invalid or empty. Cannot save results.");
        }
    
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw FileIOException("SimulationResultProcessor::saveResultsToCSV", "Could not open file for writing: " + filename);
        }
        std::cout << "Saving simulation results to: " << filename << std::endl;
    
        std::vector<std::string> state_names = model.getStateNames();
        if (state_names.empty() || state_names.size() != result.solution[0].size()) {
             std::cerr << "Warning: Could not retrieve consistent state names from model. Using generic headers." << std::endl;
             file << "Time";
             for (size_t j = 0; j < result.solution[0].size(); ++j) {
                 file << ",State_" << j;
             }
        } else {
            file << "Time";
            for (const auto& name : state_names) {
                file << "," << name;
            }
        }
        file << std::endl;
    
        for (size_t i = 0; i < result.time_points.size(); ++i) {
            file << result.time_points[i];
            for (size_t j = 0; j < result.solution[i].size(); ++j) {
                file << "," << result.solution[i][j];
            }
            file << std::endl;
        }
    
        file.close();
        std::cout << "Results saved successfully." << std::endl;
    }

    Eigen::MatrixXd SimulationResultProcessor::getIncidenceData(
        const SimulationResult& result,
        const IEpidemicModel& model,
        int num_compartments_per_age) 
    {
        (void)num_compartments_per_age;
        if (!result.isValid()) {
            throw InvalidResultException("SimulationResultProcessor::getIncidenceData", 
                                    "Simulation result object is invalid or empty.");
        }

        int n_ages = model.getNumAgeClasses();
        int n_times = result.time_points.size();
        
        // The model downcast and its population / contact-matrix accessors do not depend on
        // t, so they are resolved once instead of once per time point (the accessors return
        // by value, so this also drops a vector and a matrix copy per time point).
        const auto* concrete_model = dynamic_cast<const AgeSIRModel*>(&model);
        if (!concrete_model) {
            throw ModelException("SimulationResultProcessor::getIncidenceData",
                                "Model does not provide required methods for incidence calculation");
        }

        const Eigen::VectorXd N = concrete_model->getPopulationSizes();
        const double q = concrete_model->getTransmissibility();
        const Eigen::MatrixXd C = concrete_model->getCurrentContactMatrix();

        Eigen::MatrixXd incidence(n_times, n_ages);
        Eigen::VectorXd I_over_N(n_ages);
        Eigen::VectorXd lambda(n_ages);

        for (int t = 0; t < n_times; t++) {
            const std::vector<double>& state = result.solution[t];

            Eigen::Map<const Eigen::VectorXd> S(&state[0], n_ages);
            Eigen::Map<const Eigen::VectorXd> I(&state[n_ages], n_ages);

            for(int j=0; j<n_ages; ++j) {
                I_over_N(j) = (N(j) > 1e-9) ? (I(j) / N(j)) : 0.0;
            }
            lambda.noalias() = q * (C * I_over_N);
            lambda = lambda.cwiseMax(0.0);

            incidence.row(t) = (lambda.array() * S.array()).transpose();
        }
        
        return incidence;
    }
    
} // namespace epidemic