#ifndef AGE_SEPAIHRD_MODEL_H
#define AGE_SEPAIHRD_MODEL_H

#include "sir_age_structured/EpidemicModel.hpp"
#include "model/parameters/SEPAIHRDParameters.hpp"
#include "model/interfaces/INpiStrategy.hpp"
#include "model/PiecewiseConstantParameterStrategy.hpp"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>

namespace epidemic {

    /**
     * @brief High-Performance Age-structured SEPAIHRD epidemic model.
     * 
     * Implements a COVID-19 like model with age structure for:
     * Susceptible (S), Exposed (E), Presymptomatic (P), Asymptomatic (A),
     * Symptomatic (I), Hospitalized (H), intensive care unit (ICU), Recovered (R), and Deceased (D).
     * Transmission is governed by a global rate beta, an age-specific susceptibility vector a,
     * and an age-specific infectiousness vector h_infec.
     * 
     * Optimization Notes:
     * - Uses a single contiguous memory block for scratch space (cache locality).
     * - Precomputes reciprocals for division-free hot paths.
     * - final specifier to enable devirtualization.
     */
    class AgeSEPAIHRDModel final : public EpidemicModel {
    public:
        using VectorRef = Eigen::Ref<Eigen::VectorXd>;
        using ConstVectorRef = Eigen::Ref<const Eigen::VectorXd>;
    private:
        /** @brief Number of age classes */
        int num_age_classes;
        
        // Model Parameters (SoA Layout for SIMD efficiency)
        /** @brief Population sizes for each age class */
        Eigen::VectorXd N;
        
        /** @brief Optimization: Precomputed 1.0 / N[i] to turn division into multiplication */
        Eigen::VectorXd inv_N;
        
        /** @brief Baseline contact matrix between age classes */
        Eigen::MatrixXd M_baseline;
        
        /** @brief Current effective transmission probability per contact. */
        double beta;

        /** @brief End times for the piecewise-constant beta values */
        std::vector<double> beta_end_times_;

        /** @brief The sequence of beta values */
        std::vector<double> beta_values_;

        /** @brief Age-specific relative susceptibility vector */
        Eigen::VectorXd a; 

        /** @brief Age-specific relative infectiousness vector */
        Eigen::VectorXd h_infec;
        
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
        
        /** @brief Age-specific community/nursing home mortality rate (direct I->D) */
        Eigen::VectorXd d_community;
        
        /** @brief Strategy defining NPI effects on contact rates */
        std::shared_ptr<INpiStrategy> npi_strategy;

        /** @brief Strategy for piecewise constant parameters */
        std::unique_ptr<PiecewiseConstantParameterStrategy> beta_strategy_;
    
        /** @brief Original transmission rate before interventions */
        double baseline_beta;
        
        /** @brief Original reduced transmissibility before interventions */
        double baseline_theta;

        /**
         * @brief Optimization: Single contiguous workspace for all intermediate calculations.
         * Layout: [inf_pressure | lambda]
         * This avoids multiple small allocations and keeps hot data spatially local.
         */
        mutable std::vector<double> workspace_;
        
        /** @brief Pointers into the contiguous workspace (Zero-cost abstraction) */
        mutable double* cached_inf_pressure_ptr_ = nullptr;
        mutable double* cached_lambda_ptr_ = nullptr;

        /**
         * @brief Resizes the working vectors to match the number of age classes.
         */
        void resizeWorkingVectors();

        // Add these private member variables:
        double E0_multiplier = 1.0;
        double P0_multiplier = 1.0;
        double A0_multiplier = 1.0;
        double I0_multiplier = 1.0;
        double H0_multiplier = 1.0;
        double ICU0_multiplier = 1.0;
        double R0_multiplier = 1.0;
        double D0_multiplier = 1.0;
        
        /** @brief Number of days before t=0 to start the simulation (run-up period) */
        double runup_days = 30.0;
        
        /** @brief Total number of exposed individuals to seed at t=-runup_days */
        double seed_exposed = 10.0;

        // --- Private helper methods ---
    public:

        /**
         * @brief Computes the current beta value based on the schedule and time.
         * @param time Current simulation time
         * @return Current beta value
         */
        double computeBeta(double time) const;
        
        /**
         * @brief Constructs the age-structured SEPAIHRD model
         * @param params Model parameters struct containing all necessary parameters (excluding NPI schedule)
         * @param npi_strategy_ptr Shared pointer to an initialized NPI strategy object.
         * @throws InvalidParameterException if parameters or strategy are invalid.
        */
        AgeSEPAIHRDModel(const SEPAIHRDParameters& params, std::shared_ptr<INpiStrategy> npi_strategy_ptr);
        
        /**
         * @brief Copy Constructor (Deep copy required for strategies)
         */
        AgeSEPAIHRDModel(const AgeSEPAIHRDModel& other);

        /**
         * @brief Clone method for Prototype Pattern
         */
        std::shared_ptr<AgeSEPAIHRDModel> clone() const;

        /**
         * @brief Hot path derivative computation.
         * Marked 'final' (via class) to allow compiler devirtualization.
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
         * @brief Returns the current age-specific susceptibility vector (a)
         * @return Age-specific susceptibility vector
         */
        const Eigen::VectorXd& getSusceptibility() const;

        /**
         * @brief Returns the current age-specific infectiousness vector (h_infec)
         * @return Age-specific infectiousness vector
         */
        const Eigen::VectorXd& getInfectiousness() const;
        
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
        /** @brief Returns the age-specific community mortality rate (d_community) */
        const Eigen::VectorXd& getCommunityMortalityRate() const; 
    
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
         * @brief Sets a new age-specific susceptibility vector (a). Thread-safe.
         * Intended for controlled updates (e.g., during calibration). Bypasses intervention logic.
         * @param new_a New age-specific susceptibility vector
         * @throws InvalidParameterException if new_a has incorrect dimensions.
         */
        void setSusceptibility(const Eigen::VectorXd& new_a);

        /**
         * @brief Sets a new age-specific infectiousness vector (h_infec). Thread-safe.
         * Intended for controlled updates (e.g., during calibration). Bypasses intervention logic.
         * @param new_h_infec New age-specific infectiousness vector
         * @throws InvalidParameterException if new_h_infec has incorrect dimensions.
         */
        void setInfectiousness(const Eigen::VectorXd& new_h_infec);
    
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

        // --- Inline accessors for performance (noexcept for optimizer hints) ---
        [[nodiscard]] double getRunupDays() const noexcept { return runup_days; }
        [[nodiscard]] double getSeedExposed() const noexcept { return seed_exposed; }

        [[nodiscard]] double getE0Multiplier() const noexcept { return E0_multiplier; }
        [[nodiscard]] double getP0Multiplier() const noexcept { return P0_multiplier; }
        [[nodiscard]] double getA0Multiplier() const noexcept { return A0_multiplier; }
        [[nodiscard]] double getI0Multiplier() const noexcept { return I0_multiplier; }
        [[nodiscard]] double getH0Multiplier() const noexcept { return H0_multiplier; }
        [[nodiscard]] double getICU0Multiplier() const noexcept { return ICU0_multiplier; }
        [[nodiscard]] double getR0Multiplier() const noexcept { return R0_multiplier; }
        [[nodiscard]] double getD0Multiplier() const noexcept { return D0_multiplier; }
    };
    
} // namespace epidemic

#endif // AGE_SEPAIHRD_MODEL_H