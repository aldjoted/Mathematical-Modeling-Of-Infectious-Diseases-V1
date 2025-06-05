#ifndef AGE_SEPAIHRD_MODEL_H
#define AGE_SEPAIHRD_MODEL_H

#include "sir_age_structured/EpidemicModel.hpp"
#include "model/parameters/SEPAIHRDParameters.hpp"
#include "model/interfaces/INpiStrategy.hpp"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <mutex>
#include <memory>

namespace epidemic {

    /**
     * @brief Age-structured SEPAIHRD epidemic model
     * 
     * Implements a COVID-19 like model with age structure for:
     * Susceptible (S), Exposed (E), Presymptomatic (P), Asymptomatic (A),
     * Symptomatic (I), Hospitalized (H), intensive care unit (ICU), Recovered (R), and Deceased (D).
     * Non-pharmaceutical interventions are modeled via an injected INpiStrategy
     * that provides time-varying reduction factors applied to the baseline contact matrix.
     */
    class AgeSEPAIHRDModel : public EpidemicModel {
        private:
        /** @brief Number of age classes */
        int num_age_classes;
        
        /** @brief Population sizes for each age class */
        Eigen::VectorXd N;
        
        /** @brief Baseline contact matrix between age classes */
        Eigen::MatrixXd M_baseline;
        
        /** @brief Current effective transmission probability per contact. */
        double beta;
        
        /** @brief Current relative transmissibility of symptomatic individuals. */
        double theta;
        
        /** @brief Rate of progression from exposed to presymptomatic */
        double sigma;
        
        /** @brief Rate of progression from presymptomatic to symptomatic/asymptomatic */
        double gamma_p;
        
        /** @brief Rate of recovery for asymptomatic individuals */
        double gamma_A;
        
        /** @brief Rate of recovery for symptomatic individuals */
        double gamma_I;
        
        /** @brief Rate of recovery for hospitalized individuals */
        double gamma_H;
        
        /** @brief Rate of recovery for ICU individuals */
        double gamma_ICU;
        
        /** @brief Age-specific fraction of asymptomatic cases */
        Eigen::VectorXd p;
        
        /** @brief Age-specific hospitalization rate */
        Eigen::VectorXd h;
        
        /** @brief Age-specific ICU admission rate */
        Eigen::VectorXd icu;
        
        /** @brief Age-specific mortality rate in hospitals */
        Eigen::VectorXd d_H;
        
        /** @brief Age-specific mortality rate in ICU */
        Eigen::VectorXd d_ICU;
        
        /** @brief Strategy defining NPI effects on contact rates */
        std::shared_ptr<INpiStrategy> npi_strategy;
    
        /** @brief Original transmission rate before interventions */
        double baseline_beta;
        
        /** @brief Original reduced transmissibility before interventions */
        double baseline_theta;
    
        /** @brief Mutex for thread safety on getters/setters and internal state modification. */
        mutable std::mutex mutex_; 
    
    public:
        /**
         * @brief Constructs the age-structured SEPAIHRD model
         * @param params Model parameters struct containing all necessary parameters (excluding NPI schedule)
         * @param npi_strategy_ptr Shared pointer to an initialized NPI strategy object.
         * @throws InvalidParameterException if parameters or strategy are invalid.
        */
        AgeSEPAIHRDModel(const SEPAIHRDParameters& params, std::shared_ptr<INpiStrategy> npi_strategy_ptr);
        
        /**
         * @brief Computes the derivatives of the state variables using the appropriate kappa.
         * @param state Current state variables
         * @param derivatives Computed derivatives of the state variables
         * @param time Current time
         */
        void computeDerivatives(const std::vector<double>& state, 
                               std::vector<double>& derivatives, 
                               double time) override;
        
        /**
         * @brief Applies an intervention (potentially modifying beta or theta, but not kappa schedule)
         * @param name Name of the intervention
         * @param time Time at which the intervention is applied
         * @param params Parameters of the intervention
         */
        void applyIntervention(const std::string& name, double time, const Eigen::VectorXd& params) override;
        
        /**
         * @brief Resets the model parameters to their baseline values
         */
        void reset() override;
        
        /**
         * @brief Returns the number of state variables in the model
         * @return Number of state variables
         */
        int getStateSize() const override;
        
        /**
         * @brief Returns the names of the state variables
         * @return Vector of state variable names
         */
        std::vector<std::string> getStateNames() const override;
        
        /**
         * @brief Returns the number of age classes in the model
         * @return Number of age classes
         */
        int getNumAgeClasses() const override;
        
        /**
         * @brief Returns the population sizes by age class (const reference)
         * @return Const reference to the vector of population sizes
         */
        const Eigen::VectorXd& getPopulationSizes() const;
        
        /**
         * @brief Returns the baseline contact matrix (const reference)
         * @return Const reference to the baseline contact matrix
         */
        const Eigen::MatrixXd& getContactMatrix() const;
        
        /**
         * @brief Returns the current transmission rate (beta)
         * @return Transmission rate
         */
        double getTransmissionRate() const;
        
        /**
         * @brief Returns the current reduced transmissibility of symptomatic individuals (theta)
         * @return Reduced transmissibility
         */
        double getReducedTransmissibility() const;
    
        // --- Getters for other parameters (needed for computeObjective in calibration) ---
        /** @brief Returns the rate of progression from exposed to presymptomatic (sigma) */
        double getSigma() const;
        /** @brief Returns the rate of progression from presymptomatic to (a)symptomatic (gamma_p) */
        double getGammaP() const;
        /** @brief Returns the rate of recovery for asymptomatic individuals (gamma_A) */
        double getGammaA() const;
        /** @brief Returns the rate of recovery for symptomatic individuals (gamma_I) */
        double getGammaI() const;
        /** @brief Returns the rate of recovery for hospitalized individuals (gamma_H) */
        double getGammaH() const;
        /** @brief Returns the rate of recovery for ICU individuals (gamma_ICU) */
        double getGammaICU() const;
        /** @brief Returns the age-specific fraction of asymptomatic cases (p) */
        const Eigen::VectorXd& getProbAsymptomatic() const; 
        /** @brief Returns the age-specific hospitalization rate (h) */
        const Eigen::VectorXd& getHospRate() const;         
        /** @brief Returns the age-specific ICU admission rate (icu) */
        const Eigen::VectorXd& getIcuRate() const;          
        /** @brief Returns the age-specific mortality rate in hospitals (d_H) */
        const Eigen::VectorXd& getMortalityRateH() const;   
        /** @brief Returns the age-specific mortality rate in ICU (d_ICU) */
        const Eigen::VectorXd& getMortalityRateICU() const; 
    
        /**
         * @brief Sets a new transmission rate (beta). Thread-safe.
         * Intended for controlled updates (e.g., during calibration). Bypasses intervention logic.
         * @param new_beta New transmission rate
         * @throws InvalidParameterException if new_beta is negative.
         */
        void setTransmissionRate(double new_beta); 
        
        /**
         * @brief Sets a new reduced transmissibility factor (theta). Thread-safe.
         * Intended for controlled updates (e.g., during calibration). Bypasses intervention logic.
         * @param new_theta New reduced transmissibility factor
         * @throws InvalidParameterException if new_theta is negative.
         */
        void setReducedTransmissibility(double new_theta);
    
        /**
         * @brief Get a shared pointer to the NPI strategy object.
         * Allows external access (e.g., for calibration of NPI parameters).
         * @return std::shared_ptr<INpiStrategy>
         */
        std::shared_ptr<INpiStrategy> getNpiStrategy() const;

        /**
         * @brief Gets the current model parameters as a SEPAIHRDParameters struct.
         * @return SEPAIHRDParameters struct populated with current model values.
         * @note The contact_matrix_scaling_factor in the returned struct will be 1.0,
         *       as the model internally stores the (potentially pre-scaled) M_baseline.
         */
        SEPAIHRDParameters getModelParameters() const;

        /**
         * @brief Sets the model parameters from a SEPAIHRDParameters struct.
         * @param params The SEPAIHRDParameters struct containing the new parameters.
         * @throws InvalidParameterException if parameter validation fails (e.g., negative rates,
         *         inconsistent dimensions with existing num_age_classes).
         * @note This will update the current working parameters and the baseline_beta and baseline_theta.
         *       It does not change num_age_classes post-construction; vector dimensions in params
         *       must match the existing num_age_classes.
         */
        void setModelParameters(const SEPAIHRDParameters& params);

        /**
         * @brief Checks if the initial deaths accounted for in the model setup are zero.
         * 
         * @return true if initial deaths are considered zero, false otherwise.
         */
        bool areInitialDeathsZero() const;
    
    };
    
    }

#endif // AGE_SEPAIHRD_MODEL_H