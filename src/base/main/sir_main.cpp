#include "base/SIRModel.hpp"
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
    params.t_start = 0.0;
    params.t_end = 365.0;

    cout << "Running deterministic SIR model..." << endl;
    auto detStart = high_resolution_clock::now();

    try {
        SIRModel model(params.N, params.beta, params.gamma, params.S0,
            params.I0, params.R0, params.t_start, params.t_end, params.h,
            params.eps);
        model.solve();
    } catch (const std::exception& e) {
        cerr << "Error during SIR model execution: " << e.what() << endl;
        return 1;
    }

    auto detEnd = high_resolution_clock::now();
    auto detDuration = duration_cast<duration<double>>(detEnd - detStart);
    cout << "Deterministic model finished." << endl;
    cout << "Execution time: " << detDuration.count() << " seconds" << endl;
    return 0;
}