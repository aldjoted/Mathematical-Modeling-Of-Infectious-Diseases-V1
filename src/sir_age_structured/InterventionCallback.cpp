#include "sir_age_structured/InterventionCallBack.hpp"
#include "exceptions/Exceptions.hpp"
#include "utils/Logger.hpp"
#include <iostream>
#include <limits>

using namespace epidemic;


InterventionCallback::InterventionCallback(std::shared_ptr<AgeSIRModel> model)
    : model_(model), last_applied_time_(-1.0) {
    if (!model_) {
        THROW_INVALID_PARAM("InterventionCallback::InterventionCallback",
                                        "Model pointer cannot be null.");
    }
    Logger::getInstance().debug("InterventionCallback::InterventionCallback",
                               "Intervention callback initialized.");
}

void InterventionCallback::validateParameters(const std::string& name, const Eigen::VectorXd& params) {
    if (name == "contact_reduction" || name == "social_distancing" || name == "lockdown") {
        if (params.size() != 1) {
            THROW_INVALID_PARAM("InterventionCallback::validateParameters",
                                            "Intervention '" + name + "' requires exactly 1 parameter (scale factor). Got " + std::to_string(params.size()) + ".");
        }
        if (params(0) < 0.0) {
            THROW_INVALID_PARAM("InterventionCallback::validateParameters",
                                            "Contact scale factor for '" + name + "' cannot be negative. Got " + std::to_string(params(0)) + ".");
        }
    } else if (name == "mask_mandate" || name == "transmission_reduction") {
        if (params.size() != 1) {
            THROW_INVALID_PARAM("InterventionCallback::validateParameters",
                                            "Intervention '" + name + "' requires exactly 1 parameter (reduction factor [0,1]). Got " + std::to_string(params.size()) + ".");
        }
        if (params(0) < 0.0 || params(0) > 1.0) {
            THROW_INVALID_PARAM("InterventionCallback::validateParameters",
                                            "Transmission reduction factor for '" + name + "' must be between 0 and 1. Got " + std::to_string(params(0)) + ".");
        }
    } else {
        Logger::getInstance().info("InterventionCallback::validateParameters",
                                   "Warning: Unknown intervention type '" + name + "'. Skipping callback-level validation; model will validate.");
    }
    Logger::getInstance().debug("InterventionCallback::validateParameters",
                               "Parameters for intervention '" + name + "' validated at callback level.");
}


void InterventionCallback::addIntervention(double time, const std::string& name, const Eigen::VectorXd& params) {
    try {
        if (time < 0) {
            THROW_INVALID_PARAM("InterventionCallback::addIntervention",
                                           "Intervention time cannot be negative. Got: " + std::to_string(time));
        }

        validateParameters(name, params);

        intervention_schedule_.insert({time, {name, params}});
        last_applied_time_ = time;

        Logger::getInstance().info("InterventionCallback::addIntervention",
                                  "Added intervention '" + name + "' scheduled for time " +
                                  std::to_string(time));
    }
    catch (const InvalidParameterException& e) {
        throw;
    }
    catch (const std::exception& e) {
        throw InterventionException("InterventionCallback::addIntervention",
                                   "Failed to add intervention '" + name + "' due to unexpected error: " + std::string(e.what()));
    }
    catch (...) {
        throw InterventionException("InterventionCallback::addIntervention",
                                   "Failed to add intervention '" + name + "' due to an unknown error.");
    }
}

void InterventionCallback::applyScheduledInterventions(double t) {
    for (auto it = intervention_schedule_.upper_bound(last_applied_time_);
         it != intervention_schedule_.upper_bound(t);
         ++it)
    {
        double intervention_time = it->first;

        const std::string& name = it->second.first;
        const Eigen::VectorXd& params = it->second.second;

        try {
            Logger::getInstance().info("InterventionCallback::applyScheduledInterventions",
                                        "Applying intervention '" + name +
                                        "' scheduled for t=" + std::to_string(intervention_time) +
                                        " (current t=" + std::to_string(t) + ")");

            model_->applyIntervention(name, intervention_time, params);

        }
        catch (const InterventionException& e) {
             Logger::getInstance().error("InterventionCallback::applyScheduledInterventions",
                                       "Model rejected intervention '" + name +
                                       "' at time " + std::to_string(intervention_time) +
                                       ": " + e.what());
        }
        catch (const InvalidParameterException& e) {
             Logger::getInstance().error("InterventionCallback::applyScheduledInterventions",
                                       "Model reported invalid parameters for intervention '" + name +
                                       "' at time " + std::to_string(intervention_time) +
                                       ": " + e.what());
        }
        catch (const ModelException& e) {
            Logger::getInstance().error("InterventionCallback::applyScheduledInterventions",
                                      "Model error applying intervention '" + name +
                                      "' at time " + std::to_string(intervention_time) +
                                      ": " + e.what());
        }
        catch (const std::exception& e) {
            Logger::getInstance().error("InterventionCallback::applyScheduledInterventions",
                                      "Unexpected standard error applying intervention '" + name +
                                      "' at time " + std::to_string(intervention_time) +
                                      ": " + e.what());
        }
        catch (...) {
            Logger::getInstance().error("InterventionCallback::applyScheduledInterventions",
                                      "Unknown error applying intervention '" + name +
                                      "' at time " + std::to_string(intervention_time));
        }
    }

    last_applied_time_ = t;
}

void InterventionCallback::reset() {
    intervention_schedule_.clear();
    last_applied_time_ = -1.0;
    Logger::getInstance().info("InterventionCallback::reset", "Intervention schedule cleared and callback state reset.");
}

