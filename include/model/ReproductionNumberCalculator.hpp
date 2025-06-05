#ifndef REPRODUCTION_NUMBER_CALCULATOR_HPP
#define REPRODUCTION_NUMBER_CALCULATOR_HPP

#include "model/AgeSEPAIHRDModel.hpp"
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <string>

namespace epidemic {
    /**
     * @brief Calculates R0 (basic reproduction number) and Rt (effective reproduction number)
     * for an age-structured SEPAIHRD model using the next generation matrix method.
     
     * The NGM method considers the epidemiological states involved in transmission and early progression:
     * Exposed (E), Presymptomatic (P), Asymptomatic (A), and Symptomatic-Infectious (I) for each age group.
     * The reproduction number is the spectral radius (dominant eigenvalue) of the NGM, K = F * V^-1, where:
     * - F is the matrix representing the rate of new infections generated.
     * - V is the matrix representing the rates of transfer/transition between the E, P, A, I states
     *   and removal from these states (excluding new infection generation).
     */
    class ReproductionNumberCalculator {
    public:
    /**
     * @brief Constructor.
     * @param model A shared pointer to a configured AgeSEPAIHRDModel instance.
     *              The model should have its parameters (beta, theta, gammas, NPI strategy, etc.) set
     *              to the values for which R0 or Rt is to be calculated.
     * @throws std::invalid_argument if the model pointer is null.
     */
    explicit ReproductionNumberCalculator(std::shared_ptr<AgeSEPAIHRDModel> model);
    /**
     * @brief Calculates the Basic Reproduction Number (R0).
     * Assumes a fully susceptible population (S_a = N_a for all age groups a) and
     * uses the baseline NPI scaling factor (kappa at time 0 from the model's NPI strategy if available).
     * @return The calculated R0 value.
     * @throws ModelException if an available NPI strategy provides an invalid (negative) kappa value.
     */
    double calculateR0();

    /**
     * @brief Calculates the Effective Reproduction Number (Rt) at a specific time.
     * Uses the provided current susceptible population vector S_current(t) for each age group
     * and the NPI scaling factor kappa(t) obtained from the model's NPI strategy at the given time.
     * @param S_current An Eigen::VectorXd representing the number of susceptible individuals
     *                  in each age group at the specified time t. Its size must match num_age_classes.
     * @param time The simulation time at which to calculate Rt (used to get kappa(t)).
     * @return The calculated Rt value.
     * @throws std::invalid_argument if S_current size is incorrect.
     * @throws ModelException if NPI strategy is needed but not available or provides invalid kappa.
     */
    double calculateRt(const Eigen::VectorXd& S_current, double time);

    private:
    /** @brief A shared pointer to the age-structured SEPAIHRD model. */
    std::shared_ptr<AgeSEPAIHRDModel> model_;
    /** @brief The number of age classes in the model. */
    int num_age_classes_;
    /** @brief The number of epidemiological states considered in the NGM system (E, P, A, I = 4). */
    int num_epi_states_in_ngm_system_; 

    /**
     * @brief Builds the F matrix (new infection generation) for R0 calculation.
     * @return The F matrix.
     */
    Eigen::MatrixXd buildFMatrixForR0() const;

    /**
     * @brief Builds the F matrix (new infection generation) for Rt calculation.
     * @param S_current Current susceptible population vector.
     * @param kappa_t Current NPI scaling factor.
     * @return The F matrix.
     */
    Eigen::MatrixXd buildFMatrixForRt(const Eigen::VectorXd& S_current, double kappa_t) const;

    /**
     * @brief Builds the V matrix (transitions/removals from E, P, A, I states).
     * This matrix is constant for a given set of model progression parameters.
     * @return The V matrix.
     */
    Eigen::MatrixXd buildVMatrix() const;
    };
}



#endif // REPRODUCTION_NUMBER_CALCULATOR_HPP