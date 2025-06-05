#ifndef MODEL_PARAMETERS_HPP
#define MODEL_PARAMETERS_HPP

#include <string>
/**
 * @brief Container for epidemiological model parameters
 */
struct ModelParameters {
    // Common parameters for all models
    double N;
    double beta;
    double gamma;
    double S0;
    double I0;
    double R0;
    double t_start;
    double t_end;
    double h;
    double eps;
    
    unsigned int numSimulations; ///>- Additional parameter for the stochastic SIR model
    
    double B;  ///>- Birth rate
    double mu; ///>- Natural death rate
};
/**
 * @brief Load model parameters from a configuration file
 * @param filename Path to the parameter file
 * @param params Reference to ModelParameters struct to populate
 * @return true if parameters loaded successfully, false otherwise
 */
bool loadModelParameters(const std::string& filename, ModelParameters& params);

#endif // MODEL_PARAMETERS_HPP