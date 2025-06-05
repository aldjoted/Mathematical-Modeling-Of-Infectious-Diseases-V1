#ifndef MODEL_FACTORY
#define MODEL_FACTORY

#include "sir_age_structured/AgeSIRModel.hpp"
#include "model/AgeSEPAIHRDModel.hpp"
#include "model/parameters/SEPAIHRDParameters.hpp"
#include "model/interfaces/INpiStrategy.hpp"
#include "exceptions/Exceptions.hpp"
#include <memory>
#include <vector>
#include <Eigen/Dense>

namespace epidemic {
    /**
     * @class ModelFactory
     * @brief Factory class for creating epidemic model instances.
     *
     * Provides static methods to create and initialize different types of
     * age-structured epidemic models and their corresponding initial state vectors.
     * Includes basic validation checks before model creation.
     */
    class ModelFactory {
        public:
            /**
             * @brief Creates an age-structured SIR model instance.
             *
             * Delegates creation and detailed validation to AgeSIRModel::create.
             * Handles and transforms exceptions to provide clear error messages.
             *
             * @param N Vector of population sizes for each age group.
             * @param C Matrix of contact rates between age groups.
             * @param gamma Vector of recovery rates for each age group.
             * @param q Global transmission parameter.
             * @param scale_C Optional scaling factor for the contact matrix (default: 1.0).
             * @return std::shared_ptr<AgeSIRModel> Shared pointer to the created model.
             * 
             * @throws ModelConstructionException If model creation fails due to invalid parameters
             *         or if any unexpected errors occur during model creation.
             */
            static std::shared_ptr<AgeSIRModel> createAgeSIRModel(
                const Eigen::VectorXd& N,
                const Eigen::MatrixXd& C,
                const Eigen::VectorXd& gamma,
                double q,
                double scale_C = 1.0);
    
            /**
             * @brief Creates an age-structured SEPAIHRD model instance.
             *
             * Performs comprehensive validation of parameters before model construction:
             * - Checks for positive number of age classes
             * - Validates NPI strategy pointer
             * - Ensures dimensional consistency across all parameters
             * - Verifies non-negativity of all rate parameters
             * - Confirms probability parameters are within [0,1] bounds
             *
             * @param params Struct containing all model parameters (populations, rates, matrices, fractions).
             * @param npi_strategy_ptr Shared pointer to the NPI strategy implementation.
             * @return std::shared_ptr<AgeSEPAIHRDModel> Shared pointer to the created model.
             *
             * @throws ModelConstructionException If parameters fail validation or if model creation encounters errors.
             * @throws InvalidParameterException If npi_strategy_ptr is null or invalid parameters are detected.
             */
            static std::shared_ptr<AgeSEPAIHRDModel> createAgeSEPAIHRDModel(
                const SEPAIHRDParameters& params,
                std::shared_ptr<INpiStrategy> npi_strategy_ptr
            );
    
            /**
             * @brief Creates the initial state vector for an age-structured SIR model.
             *
             * Concatenates S0, I0, and R0 vectors into a single state vector [S0, I0, R0].
             * Performs validation to ensure:
             * - All vectors have identical, non-zero dimensions
             * - All values are non-negative
             *
             * @param S0 Initial susceptible populations by age group.
             * @param I0 Initial infected populations by age group.
             * @param R0 Initial recovered populations by age group.
             * @return Eigen::VectorXd Combined state vector [S0, I0, R0] suitable for simulation.
             *
             * @throws InvalidParameterException If vectors have mismatched dimensions, are empty,
             *         or contain negative values.
             */
            static Eigen::VectorXd createInitialSIRState(
                const Eigen::VectorXd& S0,
                const Eigen::VectorXd& I0,
                const Eigen::VectorXd& R0);
    
            /**
             * @brief Creates the initial state vector for an age-structured SEPAIHRD model.
             *
             * Concatenates all compartment vectors (S0, E0, P0, A0, I0, H0, ICU0, R0, D0)
             * into a single state vector. Performs validation to ensure:
             * - All vectors have identical, non-zero dimensions
             * - All values are non-negative
             *
             * @param S0 Initial susceptible populations by age group.
             * @param E0 Initial exposed populations by age group.
             * @param P0 Initial presymptomatic populations by age group.
             * @param A0 Initial asymptomatic populations by age group.
             * @param I0 Initial symptomatic populations by age group.
             * @param H0 Initial hospitalized populations by age group.
             * @param ICU0 Initial ICU populations by age group.
             * @param R0 Initial recovered populations by age group.
             * @param D0 Initial deceased populations by age group.
             * @return Eigen::VectorXd Combined state vector [S0, E0, P0, A0, I0, H0, ICU0, R0, D0] suitable for simulation.
             *
             * @throws InvalidParameterException If vectors have mismatched dimensions, are empty,
             *         or contain negative values.
             */
            static Eigen::VectorXd createInitialSEPAIHRDState(
                const Eigen::VectorXd& S0,
                const Eigen::VectorXd& E0,
                const Eigen::VectorXd& P0,
                const Eigen::VectorXd& A0,
                const Eigen::VectorXd& I0,
                const Eigen::VectorXd& H0,
                const Eigen::VectorXd& ICU0,
                const Eigen::VectorXd& R0,
                const Eigen::VectorXd& D0);
    };
} // namespace epidemic
#endif // MODEL_FACTORY