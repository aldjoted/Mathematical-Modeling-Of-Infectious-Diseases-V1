#include "base/ModelParameters.hpp"
#include <fstream>
#include <sstream>

bool loadModelParameters(const std::string& filename, ModelParameters& params) {
    std::ifstream file(filename);
    if (!file)
        return false;
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip blank lines or comments
        if(line.empty() || line[0]=='#' || line.substr(0,2)=="//")
            continue;
        std::istringstream iss(line);
        std::string key;
        std::string valueStr;
        if (!(iss >> key >> valueStr))
            continue;
        
        // Check and parse each parameter:
        if (key == "N") params.N = std::stod(valueStr);
        else if (key == "beta") params.beta = std::stod(valueStr);
        else if (key == "gamma") params.gamma = std::stod(valueStr);
        else if (key == "S0") params.S0 = std::stod(valueStr);
        else if (key == "I0") params.I0 = std::stod(valueStr);
        else if (key == "R0") params.R0 = std::stod(valueStr);
        else if (key == "t_start") params.t_start = std::stod(valueStr);
        else if (key == "t_end") params.t_end = std::stod(valueStr);
        else if (key == "h") params.h = std::stod(valueStr);
        else if (key == "eps") params.eps = std::stod(valueStr);
        else if (key == "numSimulations") params.numSimulations = std::stoul(valueStr);
        else if (key == "B") params.B = std::stod(valueStr);
        else if (key == "mu") params.mu = std::stod(valueStr);
    }
    return true;
}