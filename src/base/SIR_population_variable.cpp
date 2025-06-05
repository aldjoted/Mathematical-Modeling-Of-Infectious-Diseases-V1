#include "base/SIR_population_variable.hpp"
#include "utils/FileUtils.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>

using namespace std;

SIRModel_population_variable::SIRModel_population_variable(double N, double beta, double gamma, double B, double mu,
                   double S0, double I0, double R0,
                   double t_start, double t_end, double h, double eps)
    : N(N), beta(beta), gamma(gamma), B(B), mu(mu),
      S0(S0), I0(I0), R0(R0),
      t_start(t_start), t_end(t_end), h(h), eps(eps)
{}

int SIRModel_population_variable::sys_function([[maybe_unused]] double t, const double y[], double f[], void *params) {
    SIRModel_population_variable* model = static_cast<SIRModel_population_variable*>(params);
    double beta = model->beta;
    double gamma = model->gamma;
    double B = model->B;
    double mu = model->mu;

    // Calculate N dynamically as sum of compartments
    double currentN = y[0] + y[1] + y[2];
    if (currentN <= 0) {
        f[0] = B - mu * y[0];
        f[1] = - gamma * y[1] - mu * y[1];
        f[2] = gamma * y[1] - mu * y[2];
        return GSL_SUCCESS;
    }

    // y[0]=S, y[1]=I, y[2]=R
    f[0] = B - beta * y[0] * y[1] / currentN - mu * y[0];       // dS/dt
    f[1] = beta * y[0] * y[1] / currentN - gamma * y[1] - mu * y[1];  // dI/dt
    f[2] = gamma * y[1] - mu * y[2];                             // dR/dt

    return GSL_SUCCESS;
}

void SIRModel_population_variable::calculateEquilibria() {
    cout << "Equilibria for SIR model with population variation (assuming B=mu*N for constant pop. equilibrium):" << endl;

    // Disease-Free Equilibrium (DFE): (N, 0, 0) where N = B/mu
    double N_dfe = (mu > 0) ? B / mu : N;
    cout << "Disease-Free Equilibrium (DFE): S=" << N_dfe << ", I=0, R=0" << endl;

    // Calculate R0 for this model
    double R0_calc = (gamma + mu > 0) ? beta / (gamma + mu) : std::numeric_limits<double>::infinity();
    cout << "Basic Reproduction Number (R0) = " << R0_calc << endl;


    // Endemic Equilibrium (EE)
    if (R0_calc > 1.0 && beta > 0) {
        double S_star = N_dfe / R0_calc;
        double I_star = (B - mu * S_star) / (gamma + mu);
        I_star = std::max(0.0, I_star);
        double R_star = std::max(0.0, N_dfe - S_star - I_star); // R* must be non-negative

        cout << "Endemic Equilibrium (EE) exists:" << endl;
        cout << "  S* = " << S_star << endl;
        cout << "  I* = " << I_star << endl;
        cout << "  R* = " << R_star << endl;
    } else {
        cout << "Endemic Equilibrium (EE) does not exist (R0 <= 1)" << endl;
    }
}


void SIRModel_population_variable::solve() {
    gsl_odeiv2_system sys = { sys_function, nullptr, 3, this };
    gsl_odeiv2_driver *driver = gsl_odeiv2_driver_alloc_y_new(
                                    &sys, gsl_odeiv2_step_rkf45, h, eps, 0.0); // Use 0.0 for rel error

    if (!driver) {
        cerr << "Error: Failed to allocate GSL driver." << endl;
        return;
    }

    double t = t_start;
    double y[3] = { S0, I0, R0 };
    double report_dt = (t_end > t_start) ? min(1.0, (t_end - t_start) / 1000.0) : 1.0; // Report up to ~1000 points or every 1 time unit
    double next_report_time = t;

    std::string filepath = FileUtils::getOutputPath("sir_variable_population_result.csv");
    std::ofstream dataFile(filepath);

    if (!dataFile.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
        gsl_odeiv2_driver_free(driver);
        return;
    }

    dataFile << "t,S,I,R,N_total\n";
    double currentN_write = y[0] + y[1] + y[2];
    dataFile << t << "," << y[0] << "," << y[1] << "," << y[2] << "," << currentN_write << "\n";
    next_report_time += report_dt;

    while (t < t_end) {
        double t_step_target = min(next_report_time, t_end);
        int status = gsl_odeiv2_driver_apply(driver, &t, t_step_target, y);

        if (status != GSL_SUCCESS) {
            cout << "Error: GSL solver failed with code " << status << " at t = " << t << endl;
            break;
        }

        if (t >= next_report_time) {
             // Ensure non-negativity
             y[0] = max(0.0, y[0]);
             y[1] = max(0.0, y[1]);
             y[2] = max(0.0, y[2]);
             currentN_write = y[0] + y[1] + y[2];
             dataFile << t << "," << y[0] << "," << y[1] << "," << y[2] << "," << currentN_write << "\n";
             next_report_time += report_dt;
             if (next_report_time > t_end) next_report_time = t_end;
        }
    }
    // Ensure final state at t_end is captured
    if (t < t_end || abs(t-t_end) > 1e-9) {
         int status = gsl_odeiv2_driver_apply(driver, &t, t_end, y);
          if (status != GSL_SUCCESS) {
             cerr << "Error: GSL solver failed during final step to t_end." << endl;
         } else {
             y[0] = max(0.0, y[0]);
             y[1] = max(0.0, y[1]);
             y[2] = max(0.0, y[2]);
             currentN_write = y[0] + y[1] + y[2];
             dataFile << t << "," << y[0] << "," << y[1] << "," << y[2] << "," << currentN_write << "\n";
         }
    }


    dataFile.close();
    gsl_odeiv2_driver_free(driver);

    // Calculate and display equilibria after simulation completes
    calculateEquilibria();

}