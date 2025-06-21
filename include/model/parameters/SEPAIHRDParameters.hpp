#ifndef MODEL_PARAMETERS_H
#define MODEL_PARAMETERS_H

#include <Eigen/Dense>
#include <vector>

/**
 * @namespace epidemic
 * @brief Contains classes and data structures for epidemic modeling
 */
namespace epidemic {

/**
 * @struct SEPAIHRDParameters
 * @brief Parameters for the age-structured SEPAIHRD model
 * @details Contains all parameters needed for the Susceptible-Exposed-Presymptomatic-Asymptomatic-
 *          Infectious-Hospitalized-Recovered-Deceased compartmental model with age structure.
 */
struct SEPAIHRDParameters {
    /** @brief Population sizes for each age class */
    Eigen::VectorXd N;

    /** @brief Contact matrix between age classes */
    Eigen::MatrixXd M_baseline;

    /** @brief Scaling factor for the contact matrix (Default to 1.0, no scaling) */
    double contact_matrix_scaling_factor = 1.0;

    /** @brief Transmission rate (infectiousness parameter) */
    double beta;

    /** @brief Reduced transmissibility of symptomatic individuals */
    double theta;

    /** @brief Rate of progression from exposed to presymptomatic (1/latent period) */
    double sigma;

    /** @brief Rate of progression from presymptomatic (1/presymptomatic period) */
    double gamma_p;

    /** @brief Recovery rate for asymptomatic individuals (1/infectious period) */
    double gamma_A;

    /** @brief Recovery rate for symptomatic individuals (1/infectious period) */
    double gamma_I;

    /** @brief Recovery rate for hospitalized individuals (1/hospital stay duration) */
    double gamma_H;

    /** @brief Recovery rate for ICU individuals (1/ICU stay duration) */
    double gamma_ICU;

    /** @brief Age-specific fraction of asymptomatic cases */
    Eigen::VectorXd p;

    /** @brief Age-specific hospitalization rate among symptomatic cases */
    Eigen::VectorXd h;

    /** @brief Age-specific ICU admission rate among hospitalized cases */
    Eigen::VectorXd icu;

    /** @brief Age-specific mortality rate in hospitals */
    Eigen::VectorXd d_H;

    /** @brief Age-specific mortality rate in ICU */
    Eigen::VectorXd d_ICU;
    
    /** @brief End times for kappa adjustments (NPI intervention times) */
    std::vector<double> kappa_end_times;

    /** @brief Scaling factors corresponding to each kappa period */
    std::vector<double> kappa_values;

    /** @brief Multiplier for the initial number of exposed individuals (E0) */
    double E0_multiplier = 1.0;

    /** @brief Multiplier for the initial number of presymptomatic individuals (P0) */
    double P0_multiplier = 1.0;

    /** @brief Multiplier for the initial number of asymptomatic individuals (A0) */
    double A0_multiplier = 1.0;

    /** @brief Multiplier for the initial number of symptomatic individuals (I0) */
    double I0_multiplier = 1.0;

    /**
     * @brief Validates that all parameter dimensions are consistent
     * @details Checks that all vector parameters have the same length as the population vector
     *          and that the contact matrix dimensions match the number of age classes.
     * @return true if parameters are valid, false otherwise
     */
    bool validate() const {
        int n = N.size();
        if (n <= 0) return false;

        bool vector_ok = (p.size() == n) &&
                         (h.size() == n) &&
                         (icu.size() == n) &&
                         (d_H.size() == n) &&
                         (d_ICU.size() == n);

        bool matrix_ok = (M_baseline.rows() == n && M_baseline.cols() == n);

        bool multipliers_ok = (E0_multiplier >= 0.0) &&
                              (P0_multiplier >= 0.0) &&
                              (A0_multiplier >= 0.0) &&
                              (I0_multiplier >= 0.0);

        return vector_ok && matrix_ok && multipliers_ok;
    }
};

}

#endif // MODEL_PARAMETERS_H