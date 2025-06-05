#ifndef SIRMODEL_POPULATION_VARIABLE_H
#define SIRMODEL_POPULATION_VARIABLE_H

#include "utils/FileUtils.hpp"

/**
 * @brief Simulates an SIR (Susceptible-Infected-Recovered) model with population dynamics (births and deaths).
 *
 * This class solves the SIR model differential equations incorporating constant birth (B)
 * and natural death (mu) rates using the GSL ODE solver. It can also calculate
 * the equilibria (disease-free and endemic) of the system.
 */
class SIRModel_population_variable {
public:
    /**
     * @brief Constructs a new SIR Model with population variation.
     *
     * @param N         [in] Initial total population size. Note: may change if B != mu*N.
     * @param beta      [in] Transmission rate parameter (contacts * probability of transmission).
     * @param gamma     [in] Recovery rate parameter (1 / average infectious period).
     * @param B         [in] Crude birth rate (new individuals per unit time, assumed susceptible).
     * @param mu        [in] Natural death rate parameter (per capita death rate, independent of disease).
     * @param S0        [in] Initial number of susceptible individuals.
     * @param I0        [in] Initial number of infected individuals.
     * @param R0        [in] Initial number of recovered individuals.
     * @param t_start   [in] Start time of the simulation.
     * @param t_end     [in] End time of the simulation.
     * @param h         [in] Initial step size for the GSL ODE solver.
     * @param eps       [in] Error tolerance (absolute) for the GSL ODE solver step control.
     */
    SIRModel_population_variable(double N, double beta, double gamma, double B, double mu,
                                 double S0, double I0, double R0,
                                 double t_start, double t_end, double h, double eps);

    /**
     * @brief Solves the SIR model ODEs with population dynamics and writes the results to a CSV file.
     *
     * Integrates the system from t_start to t_end using the GSL ODE solver (RKF45).
     * The output CSV file will be named "sir_population_variable_results.csv".
     * @return void
     */
    void solve();

    /**
     * @brief Calculates and prints the equilibria of the SIR model with population dynamics.
     *
     * This method displays the disease-free equilibrium (DFE) and the
     * endemic equilibrium (EE) if it exists (i.e., if R0 > 1).
     * Note: Assumes B = mu*N for equilibria calculation to ensure constant population.
     * @return void
     */
    void calculateEquilibria();

private:
    double N;       ///< Initial total population size
    double beta;    ///< Transmission rate
    double gamma;   ///< Recovery rate
    double B;       ///< Birth rate (absolute number per time unit)
    double mu;      ///< Natural death rate (per capita)
    double S0;      ///< Initial susceptible population
    double I0;      ///< Initial infected population
    double R0;      ///< Initial recovered population
    double t_start; ///< Simulation start time
    double t_end;   ///< Simulation end time
    double h;       ///< Initial ODE solver step size
    double eps;     ///< ODE solver error tolerance (absolute)

    /**
     * @brief The GSL system function defining the SIR model ODEs with population dynamics.
     *
     * This static function computes the derivatives dS/dt, dI/dt, dR/dt for the SIR model
     * with births and deaths, compatible with the GSL ODE solver interface.
     *
     * @param t       [in] Current time.
     * @param y       [in] Current state vector [S, I, R].
     * @param f       [out] Output derivative vector [dS/dt, dI/dt, dR/dt].
     * @param params  [in] Pointer to the SIRModel_population_variable instance (provides model parameters).
     * @return int Status code (GSL_SUCCESS on success).
     */
    static int sys_function(double t, const double y[], double f[], void *params);
};

#endif // SIRMODEL_POPULATION_VARIABLE_H