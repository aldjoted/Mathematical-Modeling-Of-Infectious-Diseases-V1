#include "base/SIRModel.hpp"
#include "utils/FileUtils.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>

using namespace std;

SIRModel::SIRModel(double N, double beta, double gamma, double S0, double I0, double R0,
                   double t_start, double t_end, double h, double eps)
    : N(N), beta(beta), gamma(gamma),
      S0(S0), I0(I0), R0(R0),
      t_start(t_start), t_end(t_end), h(h), eps(eps)
{}

int SIRModel::sys_function([[maybe_unused]] double t, const double y[], double f[], void *params) {
    SIRModel* model = static_cast<SIRModel*>(params);
    double N = model->N;
    double beta = model->beta;
    double gamma = model->gamma;
    if (N <= 0) { // Prevent division by zero
         f[0] = 0;
         f[1] = - gamma * y[1];
         f[2] = gamma * y[1];
         return GSL_SUCCESS;
    }
    f[0] = -beta * y[0] * y[1] / N;                // dS/dt
    f[1] = beta * y[0] * y[1] / N - gamma * y[1];  // dI/dt
    f[2] = gamma * y[1];                           // dR/dt

    return GSL_SUCCESS;
}

void SIRModel::solve() {
    gsl_odeiv2_system sys = { sys_function, nullptr, 3, this };
    gsl_odeiv2_driver *driver = gsl_odeiv2_driver_alloc_y_new(
                                    &sys, gsl_odeiv2_step_rkf45, h, eps, 0.0); // Use 0.0 for rel error
    if (!driver) {
        cerr << "Error: Failed to allocate GSL driver." << endl;
        return;
    }
    double t = t_start;
    double y[3] = { S0, I0, R0 };
    double report_dt = (t_end > t_start) ? min(1.0, (t_end - t_start) / 1000.0) : 1.0;
    double next_report_time = t;


    std::string filepath = FileUtils::getOutputPath("sir_result.csv");
    std::ofstream dataFile(filepath);

    if (!dataFile.is_open()) {
        std::cerr << "Error: Could not open file for writting: " << filepath << std::endl;
        gsl_odeiv2_driver_free(driver);
        return;
    }

    dataFile << "t,S,I,R\n";
    dataFile << t << "," << y[0] << "," << y[1] << "," << y[2] << "\n";
    next_report_time += report_dt;


    while (t < t_end) {
        double t_step_target = min(next_report_time, t_end);

        int status = gsl_odeiv2_driver_apply(driver, &t, t_step_target, y);

        if (status != GSL_SUCCESS) {
            cerr << "Error: GSL solver failed with code " << status << " at t = " << t << endl;
            break;
        }
        if (t >= next_report_time) {
             y[0] = max(0.0, y[0]);
             y[1] = max(0.0, y[1]);
             y[2] = max(0.0, y[2]);
             dataFile << t << "," << y[0] << "," << y[1] << "," << y[2] << "\n";
             next_report_time += report_dt;
             if (next_report_time > t_end) next_report_time = t_end;
        }
    }
    if (t < t_end || abs(t-t_end) > 1e-9) {
         int status = gsl_odeiv2_driver_apply(driver, &t, t_end, y);
         if (status != GSL_SUCCESS) {
             cerr << "Error: GSL solver failed during final step to t_end." << endl;
         } else {
             y[0] = max(0.0, y[0]);
             y[1] = max(0.0, y[1]);
             y[2] = max(0.0, y[2]);
             dataFile << t << "," << y[0] << "," << y[1] << "," << y[2] << "\n";
         }
    }
    dataFile.close();
    gsl_odeiv2_driver_free(driver);
}