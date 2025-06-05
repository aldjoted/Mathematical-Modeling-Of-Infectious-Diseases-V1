#ifndef SIRMODEL_H
#define SIRMODEL_H

#include "utils/FileUtils.hpp"

/**
 * @brief Implements the standard deterministic SIR (Susceptible-Infected-Recovered) epidemiological model.
 *
 * Solves the classic SIR ordinary differential equations (ODEs) using the GSL library.
 * Assumes a closed population with constant size N.
 */
class SIRModel {
public:
    /**
     * @brief Constructs a new standard SIR Model instance.
     *
     * @param N         [in] Total population size (constant).
     * @param beta      [in] Transmission rate parameter (contacts * probability of transmission).
     * @param gamma     [in] Recovery rate parameter (1 / average infectious period).
     * @param S0        [in] Initial number of susceptible individuals.
     * @param I0        [in] Initial number of infected individuals.
     * @param R0        [in] Initial number of recovered individuals.
     * @param t_start   [in] Start time of the simulation.
     * @param t_end     [in] End time of the simulation.
     * @param h         [in] Initial step size for the GSL ODE solver.
     * @param eps       [in] Error tolerance (absolute) for the GSL ODE solver step control.
     */
    SIRModel(double N, double beta, double gamma, double S0, double I0, double R0,
             double t_start, double t_end, double h, double eps);

    /**
     * @brief Solves the SIR model ODEs and writes the time series results to a CSV file.
     *
     * Integrates the system from t_start to t_end using the GSL ODE solver (RKF45).
     * The output CSV file will be named "sir_results.csv".
     * @return void
     */
    void solve();

private:
    double N;       ///< Total population size
    double beta;    ///< Transmission rate
    double gamma;   ///< Recovery rate
    double S0;      ///< Initial susceptible population
    double I0;      ///< Initial infected population
    double R0;      ///< Initial recovered population
    double t_start; ///< Simulation start time
    double t_end;   ///< Simulation end time
    double h;       ///< Initial ODE solver step size
    double eps;     ///< ODE solver error tolerance (absolute)

    /**
     * @brief The GSL system function defining the standard SIR differential equations.
     *
     * This static function computes the derivatives dS/dt, dI/dt, dR/dt for the classic SIR model,
     * compatible with the GSL ODE solver interface.
     *
     * @param t       [in] Current time point.
     * @param y       [in] Current state vector [S, I, R].
     * @param f       [out] Output array to store computed derivatives [dS/dt, dI/dt, dR/dt].
     * @param params  [in] Pointer to the SIRModel instance (used to access model parameters N, beta, gamma).
     * @return int GSL_SUCCESS if computation is successful, otherwise an error code.
     */
    static int sys_function(double t, const double y[], double f[], void *params);
};

#endif // SIRMODEL_H