#include "base/SIR_stochastic.hpp"
#include "base/ModelParameters.hpp"
#include "utils/FileUtils.hpp"
#include <iostream>
#include <chrono>
#include <string>

using namespace std;
using namespace chrono;

int main (){
    ModelParameters params;
    // Get project root and construct path to parameters file
    std::string projectRoot = FileUtils::getProjectRoot();
    std::string parameterPath = FileUtils::joinPaths(projectRoot, "src/base/main/input_parameters.txt");
    // load the parameters
    if(!loadModelParameters(parameterPath, params)) {
        cerr << "Error: Could not load model parameters from: " << parameterPath << endl;
        return 1;
    }

    params.h = 1.0 / 24.0;
    cout << "Note: Using time step h = " << params.h << " for stochastic model." << endl;

    cout << "Running stochastic SIR model (" << params.numSimulations << " simulations)..." << endl;
    auto stochStart = high_resolution_clock::now();

    try {
        StochasticSIRModel stochastic_model(params.N, params.beta, params.gamma, params.S0,
            params.I0, params.R0, params.t_start, params.t_end, params.h,
            params.numSimulations);
        stochastic_model.runSimulations();
    } catch (const std::exception& e) {
         cerr << "Error during Stochastic SIR model execution: " << e.what() << endl;
         return 1;
    }

    auto stochEnd = high_resolution_clock::now();
    auto stochDuration = duration_cast<duration<double>>(stochEnd - stochStart);
    cout << "Stochastic model finished." << endl;
    cout << "Execution time: " << stochDuration.count() << " seconds" << endl;
    return 0;
}