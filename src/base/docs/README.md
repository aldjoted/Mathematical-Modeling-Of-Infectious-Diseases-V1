# Base SIR Models in C++

This folder contains C++ implementations of fundamental SIR (Susceptible-Infected-Recovered) epidemiological models, serving as functional examples discussed in [SMATM128 (UNamur) a course taught by Nicolas Franco](https://www.unamur.be/fr/modelisation-mathematique-des-maladies-infectieuses-33). The models utilize the GNU Scientific Library (GSL) for numerical integration (ODE solvers) and random number generation.

## Models Implemented

### 1. Standard Deterministic SIR Model ([`SIRModel`](include/base/SIRModel.hpp))

- **Description:** THe classic Kermack-McKendrick SIR model assuming a closed population with a constant size `N`. It models the flow between Susceptible, Infected, and Recovered compartments using ordinary differential equations (ODEs).
- **Method:** Solved numerically using the GSL ODE solver (e.g., RKF45).
- **Core Parameters:**
  - `N`: Total population size (constant).
  - `beta` (β): Transmission rate parameter. Represents the effective contact rate multiplied by the probability of transmission given contact between S and I.
  - `gamma` (γ): Recovery rate parameter. The inverse (`1/γ`) represents the average duration of the infectious period.
- **Model Equations:**

    ```math
    dS/dt = - (β / N) * S * I

    dI/dt = (β / N) * S * I - γ * I
    
    dR/dt = γ * I
    ```

- **Executable:** `sir_model`
- **Output:** `sir_result.csv` containing the time series of S, I, R compartments.

### 2. Deterministic SIR Model with Population Dynamics ([`SIRModel_population_variable`](include/base/SIR_population_variable.hpp))

- **Description:** An extension of the standard SIR model that incorporates population changes through constant crude birth rate `B` and per-capita natural death rate `mu` (μ). The total population `N` may vary over time if `B != μ * N`.
- **Method:** Solved numerically using the GSL ODE solver (e.g., RKF45).
- **Additional Parameters:**
  - `B`: Crude birth rate (number of new individuals, assumed susceptible, entering the population per unit time).
  - `mu` (μ): Natural death rate (per capita rate of death from causes unrelated to the infection, applied to all compartments).
- **Model Equations:**

    ```math
    dS/dt = B - (β / N) * S * I - μ * S
    dI/dt = (β / N) * S * I - γ * I - μ * I
    dR/dt = γ * I - μ * R
    ```

    *(Note: N in the equations represents the current total population S+I+R)*
- **Executable:** `sir_pop_var`
- **Output:** `sir_variable_population_result.csv` containing the time series of S, I, R, and total population N. Also prints calculated equilibria (DFE, EE) to the console.

### 3. Stochastic SIR Model (Chain Binomial) ([`StochasticSIRModel`](include/base/SIR_stochastic.hpp))

- **Description:** A discrete-time, stochastic implementation of the SIR model based on the chain binomial approach (often attributed to Bailey, 1975). Transitions between compartments within a small time step `h` are modeled as random draws from binomial distributions. This captures demographic stochasticity.
- **Method:** Simulation proceeds in discrete steps `h`. Uses GSL for random number generation (binomial draws). Can run multiple independent simulations to capture variability.
- **Key Parameter:**
  - `h`: Time step duration. Must be chosen appropriately small for the probabilities to be valid (e.g., `h = 1/24.0` for hourly steps).
- **Stochastic Transitions (per step `h`):**
  - Probability of one susceptible becoming infected: `p_inf = 1 - exp(-β * I(t) * h / N)`
  - Probability of one infected recovering: `p_rec = 1 - exp(-γ * h)`
  - Number of new infections: `I_new ~ Binomial(S(t), p_inf)`
  - Number of new recoveries: `R_new ~ Binomial(I(t), p_rec)`
  - Updates:

    ```math
        S(t+h) = S(t) - I_new
        I(t+h) = I(t) + I_new - R_new
        R(t+h) = R(t) + R_new
    ```
- **Executable:** `sir_stochastic`
- **Outputs:**
  - `stochastic_sir_sim_<i>.csv`: Trajectory for each individual simulation `i` (up to a limit).
  - `stochastic_sir_stats.csv`: Summary statistics (mean, median, 5th/95th percentiles) across all simulations if more than one is run.