#include "base/SIR_stochastic.hpp"
#include "utils/FileUtils.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <limits>
#include <ctime>
#include <stdexcept>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics.h>
#include <unistd.h>


StochasticSIRModel::StochasticSIRModel(double N, double beta, double gamma,
                                     double S0, double I0, double R0,
                                     double t_start, double t_end, double h,
                                     unsigned int numSimulations)
    : N(N), beta(beta), gamma(gamma),
      S0(S0), I0(I0), R0(R0),
      t_start(t_start), t_end(t_end), h(h),
      numSimulations(numSimulations), rng(nullptr)
{
    if (N <= 0 || beta < 0 || gamma < 0 || S0 < 0 || I0 < 0 || R0 < 0 || h <= 0 || t_end <= t_start || numSimulations == 0) {
         throw std::invalid_argument("Invalid parameters for StochasticSIRModel constructor.");
    }
     if (std::abs((S0 + I0 + R0) - N) > 1e-6 * N) { // Allow for minor float inaccuracies
         throw std::invalid_argument("Initial compartments S0+I0+R0 must sum to N.");
     }

    // Initialize GSL random number generator
    rng = gsl_rng_alloc(gsl_rng_mt19937);
    if (!rng) {
         throw std::runtime_error("Failed to allocate GSL RNG.");
    }
    // Seed using a combination of time and potentially other sources for better randomness
    gsl_rng_set(rng, static_cast<unsigned long>(time(NULL)) ^ (getpid() << 16));


    // Calculate number of time steps (+1 for the initial state)
    int numSteps = static_cast<int>(std::round((t_end - t_start) / h)) + 1;
     if (numSteps <= 0) {
          throw std::invalid_argument("t_start, t_end, and h result in zero or negative simulation steps.");
     }
    // Resize the results container
    try {
        results.resize(numSimulations);
        for (unsigned int i = 0; i < numSimulations; ++i) {
            results[i].resize(3); // S, I, R compartments
            for (int j = 0; j < 3; ++j) {
                results[i][j].resize(numSteps, 0.0);
            }
            // Set initial values for each simulation
            results[i][0][0] = this->S0;
            results[i][1][0] = this->I0;
            results[i][2][0] = this->R0;
        }
    } catch (const std::bad_alloc& e) {
         gsl_rng_free(rng);
         throw std::runtime_error("Failed to allocate memory for simulation results: " + std::string(e.what()));
    }
}


StochasticSIRModel::~StochasticSIRModel() {
    if (rng) gsl_rng_free(rng);
}


void StochasticSIRModel::runSimulations() {
    if (!rng) {
         std::cerr << "Error: RNG not initialized in runSimulations." << std::endl;
         return;
    }
    for (unsigned int i = 0; i < numSimulations; ++i) {
        runSingleSimulation(i);
    }

    // Write summary statistics if multiple simulations were run
    if (numSimulations > 1) {
        std::string statsFilePath = FileUtils::getOutputPath("stochastic_sir_stats.csv");
        std::ofstream statsFile(statsFilePath);

        if(!statsFile.is_open()) {
            std::cerr << "Error: Could not open file for writing: " << statsFilePath << std::endl;
            // Continue to write individual sims if possible
        } else {
            statsFile << "t,S_mean,S_median,S_p05,S_p95,I_mean,I_median,I_p05,I_p95,R_mean,R_median,R_p05,R_p95\n";

            try {
                auto stats = getStatistics(); // Calculate stats
                int numSteps = results[0][0].size(); // Get actual number of steps from results

                for (int step = 0; step < numSteps; ++step) {
                    double t = t_start + step * h;
                    statsFile << t;
                    for (int comp = 0; comp < 3; ++comp) { // For S, I, R
                        statsFile << "," << stats[0][comp][step] // mean
                                  << "," << stats[1][comp][step] // median
                                  << "," << stats[2][comp][step] // 5th percentile
                                  << "," << stats[3][comp][step]; // 95th percentile
                    }
                    statsFile << "\n";
                }
            } catch (const std::exception& e) {
                 std::cerr << "Error calculating or writing statistics: " << e.what() << std::endl;
            }
            statsFile.close();
        }
    }

    unsigned int max_sims_to_write = 100; // Example limit
    for (unsigned int sim = 0; sim < std::min(numSimulations, max_sims_to_write); ++sim) {
        std::string simFilePath = FileUtils::getOutputPath("stochastic_sir_sim_" + std::to_string(sim) + ".csv");
        std::ofstream dataFile(simFilePath);

        if(!dataFile.is_open()) {
            std::cerr << "Error: Could not open file for writing: " << simFilePath << std::endl;
            continue; // Skip this file if it can't be opened
        }
        dataFile << "t,S,I,R\n";

        int numSteps = results[sim][0].size();
        for (int step = 0; step < numSteps; ++step) {
            double t = t_start + step * h;
            dataFile << t << ","
                     << results[sim][0][step] << ","
                     << results[sim][1][step] << ","
                     << results[sim][2][step] << "\n";
        }
        dataFile.close();
    }
     if (numSimulations > max_sims_to_write) {
          std::cout << "Note: Only writing the first " << max_sims_to_write << " simulation trajectories to CSV." << std::endl;
     }
}


void StochasticSIRModel::runSingleSimulation(unsigned int simulationIndex) {
    if (!rng || simulationIndex >= results.size()) {
        std::cerr << "Error: Invalid state or index in runSingleSimulation." << std::endl;
        return;
    }

    int numSteps = results[simulationIndex][0].size(); // Use actual size

    for (int step = 0; step < numSteps - 1; ++step) {
        double S_current = results[simulationIndex][0][step];
        double I_current = results[simulationIndex][1][step];
        double R_current = results[simulationIndex][2][step];

        // Ensure compartments are non-negative integers for binomial draws
        int S_int = static_cast<int>(std::round(S_current));
        int I_int = static_cast<int>(std::round(I_current));
        int N_int = S_int + I_int + static_cast<int>(std::round(R_current));

        if (S_int < 0 || I_int < 0 || N_int < 0) {
             // This shouldn't happen with proper updates, but as a safeguard
             S_int = std::max(0, S_int);
             I_int = std::max(0, I_int);
             N_int = std::max(0, N_int);
             std::cerr << "Warning: Negative compartment detected and corrected at step " << step << std::endl;
        }

        // Handle extinction or no susceptibles/infectives
        if (I_int <= 0 || S_int <= 0) {
             // No new infections or recoveries possible/needed
             results[simulationIndex][0][step + 1] = S_current;
             results[simulationIndex][1][step + 1] = I_current;
             results[simulationIndex][2][step + 1] = R_current;
             continue;
        }

        // Calculate probabilities for this step
        double infection_rate_term = (N > 0) ? beta * I_current * h / N : 0.0;
        double pI = 1.0 - std::exp(-infection_rate_term);
        double pR = 1.0 - std::exp(-gamma * h);          

        // Clamp probabilities to [0, 1]
        pI = std::max(0.0, std::min(1.0, pI));
        pR = std::max(0.0, std::min(1.0, pR));

        // Generate binomial random numbers for transitions
        unsigned int I_new = gsl_ran_binomial(rng, pI, S_int);
        unsigned int R_new = gsl_ran_binomial(rng, pR, I_int);

        // Update compartments using integer transitions for consistency
        double S_next = static_cast<double>(S_int - I_new);
        double I_next = static_cast<double>(I_int + I_new - R_new);
        double R_next = R_current + R_new;

         // Enforce non-negativity strictly after updates
         S_next = std::max(0.0, S_next);
         I_next = std::max(0.0, I_next);
         R_next = std::max(0.0, R_next); // R should naturally increase or stay non-negative


        // Store results for the *next* time step
        results[simulationIndex][0][step + 1] = S_next;
        results[simulationIndex][1][step + 1] = I_next;
        results[simulationIndex][2][step + 1] = R_next;
    }
}


std::vector<std::vector<std::vector<double>>> StochasticSIRModel::getStatistics() const {
    if (numSimulations <= 0 || results.empty() || results[0].empty() || results[0][0].empty()) {
         // Return empty stats if no simulations or results
         return {};
    }

    size_t numSteps = results[0][0].size();
    std::vector<std::vector<std::vector<double>>> stats(4); // 0:mean, 1:median, 2:p05, 3:p95

    // Allocate stats structure
    for (auto& stat : stats) {
        stat.resize(3); // S, I, R
        for (auto& compartment : stat) {
            compartment.resize(numSteps, 0.0);
        }
    }

    // Temporary vector for calculations (reusable)
    std::vector<double> data(numSimulations);

    // Calculate statistics for each time step and compartment
    for (size_t step = 0; step < numSteps; ++step) {
        for (int comp = 0; comp < 3; ++comp) {
            for (unsigned int sim = 0; sim < numSimulations; ++sim) {
                 if (step < results[sim][comp].size()) {
                    data[sim] = results[sim][comp][step];
                 } else {
                     // Handle inconsistent data sizes if necessary (e.g., fill with NaN or last value)
                     data[sim] = std::numeric_limits<double>::quiet_NaN();
                 }
            }

            // Sort data for median and percentile calculations
            std::sort(data.begin(), data.end());

            // Calculate statistics using GSL
            stats[0][comp][step] = gsl_stats_mean(data.data(), 1, numSimulations);
            stats[1][comp][step] = gsl_stats_median_from_sorted_data(data.data(), 1, numSimulations);
            stats[2][comp][step] = gsl_stats_quantile_from_sorted_data(data.data(), 1, numSimulations, 0.05); // 5th percentile
            stats[3][comp][step] = gsl_stats_quantile_from_sorted_data(data.data(), 1, numSimulations, 0.95); // 95th percentile
        }
    }

    return stats;
}

std::vector<std::vector<std::vector<double>>> StochasticSIRModel::getResults() const {
    return results;
}