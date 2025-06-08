#include "sir_age_structured/Simulator.hpp"
#include "exceptions/Exceptions.hpp"
#include "utils/Logger.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <functional>

namespace epidemic {

    using state_type = std::vector<double>;
    
    Simulator::Simulator(std::shared_ptr<EpidemicModel> model_,
                         std::shared_ptr<IOdeSolverStrategy> solver_strategy_,
                         double start_time_,
                         double end_time_,
                         double time_step_,
                         double abs_error_,
                         double rel_error_)
        : model(model_),
          solver_strategy(solver_strategy_),
          start_time(start_time_),
          end_time(end_time_),
          time_step(time_step_),
          abs_error(abs_error_),
          rel_error(rel_error_)
    {
        if (!model) {
            THROW_INVALID_PARAM("Simulator::Simulator", "Model pointer cannot be null.");
        }
        if (!solver_strategy) {
            THROW_INVALID_PARAM("Simulator::Simulator", "Solver strategy pointer cannot be null.");
        }
        if (end_time <= start_time) {
            THROW_INVALID_PARAM("Simulator::Simulator", "End time must be greater than start time.");
        }
        if (time_step <= 0) {
            THROW_INVALID_PARAM("Simulator::Simulator", "Time step hint must be positive.");
        }
        setErrorTolerance(abs_error_, rel_error_);
        //Logger::getInstance().info("Simulator::Simulator", "Simulator initialized.");
    }
    
    void Simulator::setErrorTolerance(double abs_error_, double rel_error_) {
        if (abs_error_ < 0 || rel_error_ < 0) {
            THROW_INVALID_PARAM("Simulator::setErrorTolerance",
                                            "Error tolerances cannot be negative. Received: abs_error=" +
                                            std::to_string(abs_error_) + ", rel_error=" +
                                            std::to_string(rel_error_));
        }
        abs_error = abs_error_;
        rel_error = rel_error_;
        /*Logger::getInstance().debug("Simulator::setErrorTolerance",
                                   "Error tolerances set to abs=" + std::to_string(abs_error) +
                                   ", rel=" + std::to_string(rel_error));*/
    }
    
    SimulationResult Simulator::run(const Eigen::VectorXd& initial_state,
                                    const std::vector<double>& output_time_points) {
        //Logger::getInstance().info("Simulator::run", "Starting simulation run.");
    
        if (initial_state.size() != model->getStateSize()) {
            THROW_INVALID_PARAM("Simulator::run",
                                            "Initial state size (" + std::to_string(initial_state.size()) +
                                            ") does not match model state size (" +
                                            std::to_string(model->getStateSize()) + ").");
        }
        if (output_time_points.empty()) {
            THROW_INVALID_PARAM("Simulator::run", "Output time points vector cannot be empty.");
        }
        if (output_time_points.front() < start_time || output_time_points.back() > end_time) {
            THROW_INVALID_PARAM("Simulator::run",
                                            "Output time points must be within [" +
                                            std::to_string(start_time) + ", " +
                                            std::to_string(end_time) + "]. Received: [" +
                                            std::to_string(output_time_points.front()) + ", " +
                                            std::to_string(output_time_points.back()) + "].");
        }
        for (size_t i = 1; i < output_time_points.size(); ++i) {
            if (output_time_points[i] <= output_time_points[i - 1]) {
                THROW_INVALID_PARAM("Simulator::run",
                                                "Output time points must be strictly increasing. Found " +
                                                std::to_string(output_time_points[i]) + " after " +
                                                std::to_string(output_time_points[i-1]) + ".");
            }
        }
        //Logger::getInstance().debug("Simulator::run", "Input validation passed.");
    
        SimulationResult result;
    
        state_type std_initial_state(initial_state.data(),
                                      initial_state.data() + initial_state.size());
    
        auto system_function = [this](const state_type& x, state_type& dxdt, double t) {
            (*model)(x, dxdt, t);
        };
    
        std::vector<state_type> raw_solution;
        raw_solution.reserve(output_time_points.size());
        auto observer = [&result, &raw_solution](const state_type& x, double t) {
            result.time_points.push_back(t);
            raw_solution.push_back(x);
            //Logger::getInstance().debug("Simulator::run observer", "t = " + std::to_string(t) + ", S_age0 = " + std::to_string(x[0]));
        };
    
        //Logger::getInstance().info("Simulator::run", "Calling integrate.");
        try {
            solver_strategy->integrate(
                system_function,
                std_initial_state,
                output_time_points,
                time_step,
                observer,
                abs_error,
                rel_error
            );
        } catch (const ModelException& e) {
            Logger::getInstance().error("Simulator::run", e.what());
            throw;
        } catch (const std::exception& e) {
            std::string msg = "Integration failed: " + std::string(e.what());
            Logger::getInstance().error("Simulator::run", msg);
            throw SimulationException("Simulator::run", msg);
        } catch (...) {
            std::string msg = "Unknown error during integration.";
            Logger::getInstance().error("Simulator::run", msg);
            throw SimulationException("Simulator::run", msg);
        }
    
        if (result.time_points.front() != output_time_points.front()) {
            throw SimulationException("Simulator::run",
                                      "Missing initial timepoint " + std::to_string(output_time_points.front()));
        }
    
        result.solution.reserve(raw_solution.size());
        for (const auto& state_vec : raw_solution) {
            result.solution.push_back(state_vec);
        }
    
        if (result.solution.size() != result.time_points.size()) {
            throw SimulationException("Simulator::run", "Solution/timepoint count mismatch.");
        }
    
       /*Logger::getInstance().info("Simulator::run",
                                   "Simulation completed: " + std::to_string(result.time_points.size()) +
                                   " points.");*/
        return result;
    }
    
    std::shared_ptr<EpidemicModel> Simulator::getModel() const {
        return model;
    }
    
} // namespace epidemic