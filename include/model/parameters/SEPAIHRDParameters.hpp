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

    /** @brief Age-specific relative susceptibility vector */
    Eigen::VectorXd a; 

    /** @brief Age-specific relative infectiousness vector */
    Eigen::VectorXd h_infec;

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
    double E0_multiplier;

    /** @brief Multiplier for the initial number of presymptomatic individuals (P0) */
    double P0_multiplier;

    /** @brief Multiplier for the initial number of asymptomatic individuals (A0) */
    double A0_multiplier;

    /** @brief Multiplier for the initial number of symptomatic individuals (I0) */
    double I0_multiplier;

    /** @brief Multiplier for the initial number of hospitalized individuals (H0) */
    double H0_multiplier;

    /** @brief Multiplier for the initial number of ICU individuals (ICU0) */
    double ICU0_multiplier;

    /** @brief Multiplier for the initial number of recovered individuals (R0) */
    double R0_multiplier;

    /** @brief Multiplier for the initial number of deceased individuals (D0) */
    double D0_multiplier;

    /**
     * @brief Validates that all parameter dimensions are consistent
     * @details Checks that all vector parameters have the same length as the population vector
     *          and that the contact matrix dimensions match the number of age classes.
     * @return true if parameters are valid, false otherwise
     */
    bool validate() const {
        int num_age_classes = N.size();
        if (num_age_classes == 0) return false;
        
        if (M_baseline.rows() != num_age_classes || M_baseline.cols() != num_age_classes ||
            p.size() != num_age_classes || h.size() != num_age_classes ||
            icu.size() != num_age_classes || d_H.size() != num_age_classes ||
            d_ICU.size() != num_age_classes || a.size() != num_age_classes ||
            h_infec.size() != num_age_classes) {
            return false;
        }

        if (kappa_end_times.size() != kappa_values.size()) {
            return false;
        }

        if (beta < 0 || theta < 0 || sigma < 0 || gamma_p < 0 || gamma_A < 0 || 
            gamma_I < 0 || gamma_H < 0 || gamma_ICU < 0) {
            return false;
        }
        
        if ((p.array() < 0).any() || (p.array() > 1).any() ||
            (h_infec.array() < 0).any() || (icu.array() < 0).any() ||
            (d_H.array() < 0).any() || (d_ICU.array() < 0).any() ||
            (a.array() < 0).any() || (h.array() < 0).any()) {
            return false;
        }
        
        return true;
    }
};

}

#endif // MODEL_PARAMETERS_H