#ifndef AGE_SIR_MODEL_HPP
#define AGE_SIR_MODEL_HPP

#include "EpidemicModel.hpp"
#include "exceptions/Exceptions.hpp"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <stdexcept>
#include <mutex>
#include <memory>

namespace epidemic {

    class ModelFactory;

    /**
     * @class AgeSIRModel
     * @brief Thread-safe age-structured SIR epidemic model.
     *
     * This class implements an age-structured Susceptible-Infected-Recovered (SIR)
     * epidemic model with:
     * - Age-specific contact patterns
     * - Age-specific recovery rates
     * - Support for dynamic interventions (e.g., contact reductions, transmissibility changes)
     *
     * The model is thread-safe with mutex protection for all state modifications,
     * making it suitable for multi-threaded applications like parallel calibration.
     */
    class AgeSIRModel : public EpidemicModel {
    public:
        /**
         * @brief Deleted default constructor.
         * Use the static factory method `create()` instead.
         */
        AgeSIRModel() = delete;

        /**
         * @brief Compute derivatives for the ODE system of the age-structured SIR model. Thread-safe.
         *
         * Calculates dS/dt, dI/dt, and dR/dt for each age group based on the current state,
         * transmissibility, recovery rates, and effective contact matrix.
         *
         * @param[in] state Current state vector [S_0,...,S_{n-1}, I_0...,I_{n-1}, R_0,...,R_{n-1}], where n is the number of age classes.
         * @param[out] derivatives Output vector where calculated derivatives will be stored. Must have the same size as `state`.
         * @param[in] time Current simulation time (currently unused in the calculation but required by the interface).
         *
         * @throws epidemic::InvalidParameterException If `state` or `derivatives` vectors have incorrect size (not 3 * num_age_classes).
         * @note Potential numerical issues are typically handled by the ODE solver, not by throwing exceptions here.
         */
        void computeDerivatives(const std::vector<double>& state,
                              std::vector<double>& derivatives,
                              double time) override;

        /**
         * @brief Apply an intervention to modify model parameters dynamically. Thread-safe.
         *
         * Supported interventions:
         * - `"contact_reduction"`, `"social_distancing"`, `"lockdown"`: Multiplies the current `scale_C_total` by `params[0]`. `params[0]` must be >= 0.
         * - `"mask_mandate"`, `"transmission_reduction"`: Multiplies the current transmissibility `q` by `(1.0 - params[0])`. `params[0]` must be between 0 and 1.
         *
         * @param[in] name Intervention type name (case-sensitive).
         * @param[in] time Time at which intervention is applied (currently unused by model logic but kept for interface consistency).
         * @param[in] params Parameters specific to the intervention type. Must have size 1 for supported interventions.
         *
         * @throws epidemic::InterventionException If intervention `name` is unknown, `params` has the wrong size, or parameter values are out of the allowed range.
         */
        void applyIntervention(const std::string& name,
                             double time,
                             const Eigen::VectorXd& params) override;

        /**
         * @brief Reset model parameters (q, scale_C_total) to their initial values provided at construction. Thread-safe.
         * Recalculates the current contact matrix based on the reset scale factor.
         */
        void reset() override;

        /**
         * @brief Get the total number of state variables in the model. Thread-safe (const).
         *
         * @return int Total number of state variables (always 3 * num_age_classes).
         */
        int getStateSize() const override;

        /**
         * @brief Get the names of all state variables in the order they appear in the state vector. Thread-safe (const).
         * Returns ["S0", "S1", ..., "Sn-1", "I0", ..., "In-1", "R0", ..., "Rn-1"].
         *
         * @return std::vector<std::string> Vector of state variable names.
         */
        std::vector<std::string> getStateNames() const override;

        /**
         * @struct SIRParams
         * @brief Structure containing calibratable SIR model parameters.
         * @deprecated Use individual setters/getters or the `create` factory method for parameter management.
         *             Kept temporarily for reference if needed during transition.
         */
        struct SIRParams {
            double q_val;                ///< Transmissibility parameter (q)
            double scale_C_val;          ///< Contact matrix scaling factor (scale_C_total)
            Eigen::VectorXd gamma_vals;  ///< Recovery rates by age group (gamma)
        };

        /**
         * @brief Get the number of age classes in the model. Thread-safe (const).
         *
         * @return int Number of age groups.
         */
        int getNumAgeClasses() const;

        /**
         * @brief Get the population sizes by age group. Thread-safe (const).
         *
         * @return Eigen::VectorXd Vector of population sizes (N).
         */
        Eigen::VectorXd getPopulationSizes() const;

        /**
         * @brief Get the current effective contact matrix (C_current = scale_C_total * C_baseline). Thread-safe (const).
         *
         * Returns the contact matrix currently used in derivative calculations,
         * reflecting applied scaling factors from interventions or direct setters.
         *
         * @return Eigen::MatrixXd Current effective contact matrix.
         */
        Eigen::MatrixXd getCurrentContactMatrix() const;

        /**
         * @brief Get the baseline (unmodified) contact matrix provided at construction. Thread-safe (const).
         *
         * @return Eigen::MatrixXd Original baseline contact matrix (C_baseline).
         */
        Eigen::MatrixXd getBaselineContactMatrix() const;

        /**
         * @brief Get the recovery rates by age group. Thread-safe (const).
         *
         * @return Eigen::VectorXd Vector of recovery rates (gamma values).
         */
        Eigen::VectorXd getRecoveryRate() const;

        /**
         * @brief Get the current transmissibility parameter. Thread-safe (const).
         *
         * @return double Current transmissibility parameter (q).
         */
        double getTransmissibility() const;

        /**
         * @brief Get the current overall contact matrix scaling factor. Thread-safe (const).
         *
         * @return double Current scaling factor for the contact matrix (scale_C_total).
         */
        double getContactScaleFactor() const;

        /**
         * @brief Set the recovery rates by age group. Thread-safe.
         *
         * @param[in] new_gamma Vector of new recovery rates. Must have size `num_age_classes`.
         *
         * @throws epidemic::InvalidParameterException If `new_gamma` vector has the wrong size or contains negative values.
         */
        void setRecoveryRate(const Eigen::VectorXd& new_gamma);

        /**
         * @brief Set the transmissibility parameter. Thread-safe.
         *
         * @param[in] new_q New transmissibility value. Must be non-negative.
         *
         * @throws epidemic::InvalidParameterException If `new_q` is negative.
         */
        void setTransmissibility(double new_q);

        /**
         * @brief Set the overall contact matrix scaling factor. Thread-safe.
         * This directly sets `scale_C_total` and updates `C_current`.
         *
         * @param[in] new_scale New scaling factor. Must be non-negative.
         *
         * @throws epidemic::InvalidParameterException If `new_scale` is negative.
         */
        void setContactScaleFactor(double new_scale);

        /**
         * @brief Factory method to create an AgeSIRModel instance.
         *
         * Validates input parameters before creating the model object. Ensures dimensional consistency
         * and non-negativity of parameters where required.
         *
         * @param[in] N Vector of population sizes by age group. Must have positive size and non-negative entries.
         * @param[in] C Baseline contact matrix between age groups. Must be square with dimensions matching `N.size()` and have non-negative entries.
         * @param[in] gamma Vector of recovery rates by age group. Must have size matching `N.size()` and non-negative entries.
         * @param[in] q Transmissibility parameter. Must be non-negative.
         * @param[in] scale_C Initial overall scaling factor for the contact matrix. Must be non-negative. Defaults to 1.0.
         *
         * @return std::shared_ptr<AgeSIRModel> Shared pointer to the created model instance.
         *
         * @throws epidemic::ModelConstructionException If parameters are invalid (e.g., dimension mismatch, negative rates/population/scale_C, negative contact entries, zero age classes).
         */
        static std::shared_ptr<AgeSIRModel> create(
            const Eigen::VectorXd& N,
            const Eigen::MatrixXd& C,
            const Eigen::VectorXd& gamma,
            double q,
            double scale_C = 1.0);

    private:
        /**
         * @brief Private constructor, only accessible to the `create` static method via `MakeSharedEnabler` trick and `ModelFactory`.
         * Initializes the model state and baseline parameters. Calls internal validation.
         *
         * @param[in] N Vector of population sizes by age group.
         * @param[in] C Baseline contact matrix between age groups.
         * @param[in] gamma Vector of recovery rates by age group.
         * @param[in] q Transmissibility parameter.
         * @param[in] scale_C Initial scaling factor for contact matrix.
         *
         * @throws epidemic::ModelConstructionException If internal validation (`validate_parameters_nolock`) fails during construction.
         */
        AgeSIRModel(const Eigen::VectorXd& N,
                    const Eigen::MatrixXd& C,
                    const Eigen::VectorXd& gamma,
                    double q,
                    double scale_C);
        /**
         * @brief Friend declaration to allow ModelFactory access to the private constructor.
         */
        friend class epidemic::ModelFactory;
        /**
         * @brief Update the effective contact matrix `C_current` based on `C_baseline` and `scale_C_total`. (Internal, assumes lock is held).
         * Clamps `scale_C_total` to 1.0 if it becomes negative (with a warning).
         */
        void update_C_current_nolock();
        /**
         * @brief Validate all core model parameters for consistency and valid ranges. (Internal, assumes lock is held).
         * Checks dimensions of N, gamma, C_baseline against num_age_classes.
         * Checks for non-negativity of N, gamma, q, scale_C_total, C_baseline entries.
         * Checks if num_age_classes is positive.
         *
         * @throws epidemic::InvalidParameterException If any parameters are invalid.
         */
        void validate_parameters_nolock() const;
        /**
         * @brief Set recovery rate without acquiring the mutex lock. (Internal, assumes lock is held).
         * @param[in] new_gamma Vector of new recovery rates.
         * @throws epidemic::InvalidParameterException If `new_gamma` has wrong size or negative values.
         */
        void setRecoveryRate_nolock(const Eigen::VectorXd& new_gamma);
        /**
         * @brief Set transmissibility without acquiring the mutex lock. (Internal, assumes lock is held).
         * @param[in] new_q New transmissibility value.
         * @throws epidemic::InvalidParameterException If `new_q` is negative.
         */
        void setTransmissibility_nolock(double new_q);
        /**
         * @brief Set contact scale factor without acquiring the mutex lock. (Internal, assumes lock is held).
         * Note: This only sets `scale_C_total`. `update_C_current_nolock` must be called separately afterwards.
         * @param[in] new_scale New scaling factor.
         * @throws epidemic::InvalidParameterException If `new_scale` is negative.
         */
        void setContactScaleFactor_nolock(double new_scale);


        mutable std::mutex mutex_;     ///< Mutex for ensuring thread safety of state modifications.
        int num_age_classes;           ///< Number of age groups in the model.
        Eigen::VectorXd N;             ///< Population sizes per age class.
        Eigen::VectorXd gamma;         ///< Age-specific recovery rates (I -> R).
        double q;                      ///< Transmissibility parameter.
        Eigen::MatrixXd C_baseline;    ///< Baseline (unmodified) contact matrix.
        Eigen::MatrixXd C_current;     ///< Effective contact matrix after scaling (C_baseline * scale_C_total).
        double scale_C_total;          ///< Overall scaling factor applied to C_baseline.

        // Baseline parameters stored for reset() functionality
        double baseline_q;             ///< Initial transmissibility value for reset.
        double baseline_scale_C_total; ///< Initial contact scaling factor for reset.
    };

} // namespace epidemic

#endif // AGE_SIR_MODEL_HPP