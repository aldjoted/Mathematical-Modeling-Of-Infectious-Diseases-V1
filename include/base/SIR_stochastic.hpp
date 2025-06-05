#ifndef STOCHASTIC_SIR_MODEL_H
#define STOCHASTIC_SIR_MODEL_H

#include "utils/FileUtils.hpp"
#include <vector>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

/**
 * @brief Implements a discrete-time stochastic SIR (Susceptible-Infected-Recovered) model.
 *
 * Uses the binomial chain approach described by Bailey (1975).
 * Transitions between compartments (new infections, recoveries) at each time step
 * are modeled as binomial random variables. Allows running multiple simulations.
 */
class StochasticSIRModel {
public:
    /**
     * @brief Initializes the stochastic SIR model parameters and simulation settings.
     *
     * @param N              [in] Total population size (constant).
     * @param beta           [in] Transmission rate parameter.
     * @param gamma          [in] Recovery rate parameter.
     * @param S0             [in] Initial number of susceptible individuals.
     * @param I0             [in] Initial number of infected individuals.
     * @param R0             [in] Initial number of recovered individuals.
     * @param t_start        [in] Start time for the simulation.
     * @param t_end          [in] End time for the simulation.
     * @param h              [in] Time step duration (e.g., 1 day, 1/8 day). Should be small enough
     *                          such that probabilities p_inf = 1-exp(-beta*I*h/N) and p_rec = 1-exp(-gamma*h) are < 1.
     * @param numSimulations [in] The number of independent stochastic simulations to perform. Default is 1.
     */
    StochasticSIRModel(double N, double beta, double gamma,
                       double S0, double I0, double R0,
                       double t_start, double t_end, double h,
                       unsigned int numSimulations = 1);

    /**
     * @brief Destructor. Frees the allocated GSL random number generator resources.
     */
    ~StochasticSIRModel();

    /**
     * @brief Runs all the configured stochastic simulations.
     *
     * Populates the internal results vector. Writes simulation results to "stochastic_sir_results.csv"
     * and summary statistics to "stochastic_sir_summary.csv".
     * @return void
     */
    void runSimulations();

    /**
     * @brief Retrieves the raw results from all simulations.
     *
     * @return A 3D vector containing the trajectories for each simulation.
     *         Structure: [simulation_index][compartment_index (0=S, 1=I, 2=R)][time_index].
     */
    std::vector<std::vector<std::vector<double>>> getResults() const;

    /**
     * @brief Calculates and retrieves summary statistics across all simulations.
     *
     * Calculates mean, median, 5th, and 95th percentiles at each time point for S, I, and R compartments.
     *
     * @return A 3D vector containing the calculated statistics.
     *         Structure: [statistic_index (0=mean, 1=median, 2=p5, 3=p95)][compartment_index (0=S, 1=I, 2=R)][time_index].
     */
    std::vector<std::vector<std::vector<double>>> getStatistics() const;

private:
    double N;       ///< Total population size
    double beta;    ///< Infection rate parameter (per capita rate for an S-I contact resulting in infection)
    double gamma;   ///< Recovery rate parameter (per capita rate of recovery for an I individual)
    double S0;      ///< Initial number of susceptible individuals
    double I0;      ///< Initial number of infected individuals
    double R0;      ///< Initial number of recovered individuals
    double t_start; ///< Simulation start time
    double t_end;   ///< Simulation end time
    double h;       ///< Simulation time step

    unsigned int numSimulations; ///< Number of simulations to run

    gsl_rng* rng;   ///< GSL random number generator instance

    ///< Stores simulation results: results[sim_idx][comp_idx][time_idx] (comp_idx: 0=S, 1=I, 2=R)
    std::vector<std::vector<std::vector<double>>> results;

    /**
     * @brief Runs a single stochastic simulation trajectory.
     *
     * Simulates the transitions using binomial distributions for new infections and recoveries
     * based on the current state and parameters over the time interval [t_start, t_end].
     * Stores the result in the `results` member variable at the given index.
     *
     * @param simulationIndex [in] The index (0 to numSimulations-1) for storing this run's results.
     * @return void
     */
    void runSingleSimulation(unsigned int simulationIndex);
};

#endif // STOCHASTIC_SIR_MODEL_H