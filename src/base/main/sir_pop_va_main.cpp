#include "base/ModelParameters.hpp"
#include "base/SIR_population_variable.hpp"
#include "utils/FileUtils.hpp"
#include <iostream>
#include <chrono>
#include <string>

using namespace std;
using namespace chrono;

int main(){
    ModelParameters params;
    // Get project root and construct path to parameters file
    std::string projectRoot = FileUtils::getProjectRoot();
    std::string parameterPath = FileUtils::joinPaths(projectRoot, "src/base/main/input_parameters.txt");
    // load the parameters
    if(!loadModelParameters(parameterPath, params)) {
        cerr << "Error: Could not load model parameters from: " << parameterPath << endl;
        return 1;
    }
    // Example overrides if needed
    // params.N = 1000.0;
    // params.B = 10.0; // Example birth rate
    // params.mu = 0.01; // Example death rate
    cout << "Running SIR model with population variable..." << endl;
    auto popstart = high_resolution_clock::now();

    try {
        SIRModel_population_variable model_pop(params.N, params.beta, params.gamma,
            params.B, params.mu, params.S0, params.I0, params.R0, params.t_start,
            params.t_end, params.h, params.eps);
        model_pop.solve();
    } catch (const std::exception& e) {
        cerr << "Error during SIR model (pop var) execution: " << e.what() << endl;
        return 1;
    }

    auto popend = high_resolution_clock::now();
    auto popduration = duration_cast<duration<double>>(popend - popstart);
    cout << "Population variable model finished." << endl;
    cout << "Execution time: " << popduration.count() << " seconds" << endl;
    return 0;
}