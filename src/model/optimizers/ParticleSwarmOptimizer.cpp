#include "model/optimizers/ParticleSwarmOptimizer.hpp"
#include "utils/Logger.hpp"
#include <cmath>
#include <algorithm>
#include <random>           
#include <sstream>          

namespace epidemic {

ParticleSwarmOptimizer::~ParticleSwarmOptimizer() = default;

std::string ParticleSwarmOptimizer::vector_to_string(const Eigen::VectorXd& vec) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < vec.size(); ++i) {
        oss << vec[i] << (i == vec.size() - 1 ? "" : ", ");
    }
    oss << "]";
    return oss.str();
}

void ParticleSwarmOptimizer::configure(const std::map<std::string, double>& settings) {
    Logger& logger = Logger::getInstance();

    for (const auto& pair : settings) {
        try {
            if (pair.first == "iterations") {
                if (pair.second <= 0) throw std::invalid_argument("iterations must be positive integer.");
                iterations_ = static_cast<int>(pair.second);
            } else if (pair.first == "swarm_size") {
                if (pair.second <= 0) throw std::invalid_argument("swarm_size must be positive integer.");
                swarm_size_ = static_cast<int>(pair.second);
            } else if (pair.first == "omega_start") {
                if (pair.second < 0) throw std::invalid_argument("omega_start must be non-negative.");
                omega_start_ = pair.second;
            } else if (pair.first == "omega_end") {
                if (pair.second < 0) throw std::invalid_argument("omega_end must be non-negative.");
                omega_end_ = pair.second;
            } else if (pair.first == "c1_initial") {
                if (pair.second < 0) throw std::invalid_argument("c1_initial must be non-negative.");
                c1_initial_ = pair.second;
            } else if (pair.first == "c1_final") {
                if (pair.second < 0) throw std::invalid_argument("c1_final must be non-negative.");
                c1_final_ = pair.second;
            } else if (pair.first == "c2_initial") {
                if (pair.second < 0) throw std::invalid_argument("c2_initial must be non-negative.");
                c2_initial_ = pair.second;
            } else if (pair.first == "c2_final") {
                if (pair.second < 0) throw std::invalid_argument("c2_final must be non-negative.");
                c2_final_ = pair.second;
            } else if (pair.first == "report_interval") {
                if (pair.second <= 0) throw std::invalid_argument("report_interval must be positive integer.");
                report_interval_ = static_cast<int>(pair.second);
            } else if (pair.first == "use_lbest_topology") {
                use_lbest_topology_ = (pair.second != 0.0);
            } else if (pair.first == "lbest_neighborhood_k") {
                if (pair.second < 0) throw std::invalid_argument("lbest_neighborhood_k must be non-negative integer.");
                lbest_neighborhood_k_ = static_cast<int>(pair.second);
            } else if (pair.first == "use_constriction_factor") {
                use_constriction_factor_ = (pair.second != 0.0);
            } else if (pair.first == "log_new_gbest") {
                log_new_gbest_ = (pair.second != 0.0);
            } else if (pair.first == "log_particle_details") {
                log_particle_details_ = (pair.second != 0.0);
            } else if (pair.first == "particle_log_interval") {
                if (pair.second <= 0) throw std::invalid_argument("particle_log_interval must be positive integer.");
                particle_log_interval_ = static_cast<int>(pair.second);
            } else if (pair.first == "particles_to_log_count") {
                if (pair.second < 0) throw std::invalid_argument("particles_to_log_count must be non-negative integer.");
                particles_to_log_count_ = static_cast<int>(pair.second);
            }
            else if (pair.first == "omega") { // Deprecated
                logger.warning(logger_source_id_, "'omega' setting is deprecated. Use 'omega_start' and 'omega_end'. This setting will be ignored.");
            }
            else {
                logger.warning(logger_source_id_, "Unknown PSO setting: '" + pair.first + "'. This setting will be ignored.");
            }
        } catch (const std::invalid_argument& e) {
            logger.error(logger_source_id_, "Invalid configuration for '" + pair.first + "': " + e.what());
            throw;
        }
    }

    if (omega_start_ < omega_end_ && !use_constriction_factor_) {
        logger.warning(logger_source_id_, "omega_end (" + std::to_string(omega_end_) + ") is greater than omega_start (" + std::to_string(omega_start_) + "). Omega will increase over iterations.");
    }
    if (c1_initial_ < c1_final_) {
         logger.warning(logger_source_id_, "c1_final (" + std::to_string(c1_final_) + ") is greater than c1_initial (" + std::to_string(c1_initial_) + "). Cognitive component c1 will increase (typically decreases).");
    }
     if (c2_initial_ > c2_final_) {
         logger.warning(logger_source_id_, "c2_final (" + std::to_string(c2_final_) + ") is less than c2_initial (" + std::to_string(c2_initial_) + "). Social component c2 will decrease (typically increases).");
    }
}

void ParticleSwarmOptimizer::initializeSwarm(
    std::vector<Particle>& swarm,
    IObjectiveFunction& objectiveFunction,
    IParameterManager& parameterManager,
    Eigen::VectorXd& gbest_position,
    double& gbest_value,
    const Eigen::VectorXd* initial_params)
{
    Logger& logger = Logger::getInstance();
    int n = parameterManager.getParameterCount();
    swarm.resize(swarm_size_);
    gbest_value = -std::numeric_limits<double>::infinity();

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<double> lb(n), ub(n), vmax_abs_component(n);
    for (int k = 0; k < n; ++k) {
        lb[k] = parameterManager.getLowerBoundForParamIndex(k);
        ub[k] = parameterManager.getUpperBoundForParamIndex(k);
        vmax_abs_component[k] = std::abs(ub[k] - lb[k]);
        if (vmax_abs_component[k] < 1e-9) {
             vmax_abs_component[k] = 1.0;
             logger.debug(logger_source_id_, "Parameter " + std::to_string(k) + " has zero/small range. Setting Vmax_abs_component to 1.0 for stability.");
        }
    }

    for (int i = 0; i < swarm_size_; ++i) {
        swarm[i].position.resize(n);
        swarm[i].velocity.resize(n);
        
        // Use initial parameters for first particle if provided
        if (i == 0 && initial_params != nullptr && initial_params->size() == n) {
            logger.info(logger_source_id_, "Initializing first particle with provided initial parameters");
            swarm[i].position = *initial_params;
            
            // Ensure the initial parameters are within bounds
            for (int k = 0; k < n; ++k) {
                swarm[i].position[k] = std::max(lb[k], std::min(ub[k], swarm[i].position[k]));
            }
            
            logger.debug(logger_source_id_, "First particle position: " + vector_to_string(swarm[i].position));
        } else {
            // Random initialization for other particles
            for (int k = 0; k < n; ++k) {
                std::uniform_real_distribution<> dx(lb[k], ub[k]);
                swarm[i].position[k] = dx(gen);
            }
        }
        
        // Initialize velocity
        for (int k = 0; k < n; ++k) {
            if (lb[k] == ub[k]) {
                swarm[i].velocity[k] = 0.0;
            } else {
                std::uniform_real_distribution<> dv(-vmax_abs_component[k] * 0.5, vmax_abs_component[k] * 0.5);
                swarm[i].velocity[k] = dv(gen);
            }
        }

        swarm[i].pbest_position = swarm[i].position;
        swarm[i].pbest_value = objectiveFunction.calculate(swarm[i].position);

        if (swarm[i].pbest_value > gbest_value) {
            gbest_value = swarm[i].pbest_value;
            gbest_position = swarm[i].pbest_position;
        }
    }
    
    if (initial_params != nullptr && initial_params->size() == n) {
        logger.info(logger_source_id_, "Initialized swarm with initial guess. First particle fitness: " + std::to_string(swarm[0].pbest_value));
    }
    logger.info(logger_source_id_, "Initialized swarm of " + std::to_string(swarm_size_) + " particles. Initial gbest value = " + std::to_string(gbest_value));
}

Eigen::VectorXd ParticleSwarmOptimizer::get_lbest_position_for_particle(
    const std::vector<Particle>& swarm,
    int particle_idx,
    int neighborhood_k)
{
    Eigen::VectorXd lbest_pos = swarm[particle_idx].pbest_position;
    double lbest_val = swarm[particle_idx].pbest_value;

    for (int j = 1; j <= neighborhood_k; ++j) {
        int neighbor_idx_right = (particle_idx + j) % swarm_size_;
        if (swarm[neighbor_idx_right].pbest_value > lbest_val) {
            lbest_val = swarm[neighbor_idx_right].pbest_value;
            lbest_pos = swarm[neighbor_idx_right].pbest_position;
        }
        int neighbor_idx_left = (particle_idx - j + swarm_size_) % swarm_size_;
         if (swarm[neighbor_idx_left].pbest_value > lbest_val) {
            lbest_val = swarm[neighbor_idx_left].pbest_value;
            lbest_pos = swarm[neighbor_idx_left].pbest_position;
        }
    }
    return lbest_pos;
}


OptimizationResult ParticleSwarmOptimizer::optimize(
    const Eigen::VectorXd& initialParameters,
    IObjectiveFunction& objectiveFunction,
    IParameterManager& parameterManager)
{
    Logger& logger = Logger::getInstance();
    logger.info(logger_source_id_, "--- Starting PSO (maximization) ---");
    logger.info(logger_source_id_, "Iterations: " + std::to_string(iterations_) + ", Swarm Size: " + std::to_string(swarm_size_));
    if (use_constriction_factor_) {
        logger.info(logger_source_id_, "Using Constriction Factor.");
    } else {
        logger.info(logger_source_id_, "Inertia Weight (omega) will vary from " + std::to_string(omega_start_) + " to " + std::to_string(omega_end_));
    }
    logger.info(logger_source_id_, "Cognitive Coeff (c1) will vary from " + std::to_string(c1_initial_) + " to " + std::to_string(c1_final_));
    logger.info(logger_source_id_, "Social Coeff (c2) will vary from " + std::to_string(c2_initial_) + " to " + std::to_string(c2_final_));
    if (use_lbest_topology_) {
        logger.info(logger_source_id_, "Using LBest topology with neighborhood k=" + std::to_string(lbest_neighborhood_k_));
    } else {
        logger.info(logger_source_id_, "Using GBest topology.");
    }
    logger.info(logger_source_id_, "Logging new gbest: " + std::string(log_new_gbest_ ? "Enabled" : "Disabled"));
    logger.info(logger_source_id_, "Logging particle details: " + std::string(log_particle_details_ ? "Enabled" : "Disabled") + 
                                (log_particle_details_ ? " (Interval: " + std::to_string(particle_log_interval_) + ", Count: " + std::to_string(std::min(particles_to_log_count_, swarm_size_)) + ")" : ""));


    int n = parameterManager.getParameterCount();
    std::vector<Particle> swarm;
    Eigen::VectorXd gbest_position(n);
    double gbest_value;

    const Eigen::VectorXd* initial_params_ptr = nullptr;
    if (initialParameters.size() == n) {
        initial_params_ptr = &initialParameters;
        logger.info(logger_source_id_, "Using provided initial parameters for first particle");
    } else if (initialParameters.size() > 0) {
        logger.warning(logger_source_id_, "Initial parameters size (" + std::to_string(initialParameters.size()) + 
                      ") doesn't match parameter count (" + std::to_string(n) + "). Ignoring initial parameters.");
    }

    initializeSwarm(swarm, objectiveFunction, parameterManager, gbest_position, gbest_value, initial_params_ptr);
    int actual_particles_to_log = std::min(particles_to_log_count_, swarm_size_);


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> unif(0.0, 1.0);

    std::vector<double> lb(n), ub(n), vmax_abs_component(n);
    for (int k = 0; k < n; ++k) {
        lb[k] = parameterManager.getLowerBoundForParamIndex(k);
        ub[k] = parameterManager.getUpperBoundForParamIndex(k);
        vmax_abs_component[k] = std::abs(ub[k] - lb[k]);
        if (vmax_abs_component[k] < 1e-9) vmax_abs_component[k] = 1.0;
    }

    double current_omega, c1_current, c2_current;

    for (int iter = 0; iter < iterations_; ++iter) {
        double iter_ratio = (iterations_ > 1) ? (static_cast<double>(iter) / (iterations_ - 1)) : 0.0;
        current_omega = omega_start_ - (omega_start_ - omega_end_) * iter_ratio;
        current_omega = std::min(std::max(omega_start_, omega_end_), std::max(std::min(omega_start_, omega_end_), current_omega));

        c1_current = c1_initial_ - (c1_initial_ - c1_final_) * iter_ratio;
        c2_current = c2_initial_ + (c2_final_ - c2_initial_) * iter_ratio;

        bool detailed_log_this_iter = log_particle_details_ && 
                                     ((iter + 1) % particle_log_interval_ == 0 || iter == iterations_ -1);

        for (int i = 0; i < swarm_size_; ++i) {
            Eigen::VectorXd attractor_position;
            if (use_lbest_topology_) {
                attractor_position = get_lbest_position_for_particle(swarm, i, lbest_neighborhood_k_);
            } else {
                attractor_position = gbest_position;
            }

            Eigen::VectorXd r1 = Eigen::VectorXd::NullaryExpr(n, [&](){ return unif(gen); });
            Eigen::VectorXd r2 = Eigen::VectorXd::NullaryExpr(n, [&](){ return unif(gen); });


            Eigen::VectorXd cognitive_term = c1_current * r1.cwiseProduct(swarm[i].pbest_position - swarm[i].position);
            Eigen::VectorXd social_term    = c2_current * r2.cwiseProduct(attractor_position - swarm[i].position);

            if (use_constriction_factor_) {
                double phi = c1_current + c2_current;
                double K_val = 0.729;
                if (phi > 4.0) {
                    K_val = 2.0 / std::abs(2.0 - phi - std::sqrt(phi * phi - 4.0 * phi));
                } else {
                    if(iter == 0 && i == 0) {
                         logger.warning(logger_source_id_, "phi (c1_current + c2_current = " + std::to_string(phi) + 
                                                        ") is <= 4.0. Constriction factor formula typically requires phi > 4.0. Defaulting to K=" + std::to_string(K_val));
                    }
                }
                swarm[i].velocity = K_val * (swarm[i].velocity + cognitive_term + social_term);
            } else {
                swarm[i].velocity = current_omega * swarm[i].velocity + cognitive_term + social_term;
            }

            for (int k = 0; k < n; ++k) {
                if (lb[k] == ub[k]) {
                    swarm[i].velocity[k] = 0.0;
                } else {
                    swarm[i].velocity[k] = std::max(-vmax_abs_component[k], std::min(vmax_abs_component[k], swarm[i].velocity[k]));
                }
            }

            swarm[i].position += swarm[i].velocity;
            for (int k = 0; k < n; ++k) {
                swarm[i].position[k] = std::max(lb[k], std::min(ub[k], swarm[i].position[k]));
            }

            double current_value = objectiveFunction.calculate(swarm[i].position);

            if (current_value > swarm[i].pbest_value) {
                swarm[i].pbest_value = current_value;
                swarm[i].pbest_position = swarm[i].position;

                if (current_value > gbest_value) {
                    double old_gbest = gbest_value;
                    gbest_value = current_value;
                    gbest_position = swarm[i].position;
                    if(log_new_gbest_){
                        logger.info(logger_source_id_, "[Iter " + std::to_string(iter+1) + ", Particle " + std::to_string(i) +
                                                    "] New gbest: " + std::to_string(gbest_value) +
                                                    " (Improvement: " + std::to_string(gbest_value - old_gbest) + ")"  
                                                    );
                        logger.debug(logger_source_id_, "[Iter " + std::to_string(iter+1) + "] New gbest position: " + vector_to_string(gbest_position));

                    }
                }
            }
             if(detailed_log_this_iter && i < actual_particles_to_log){
                logger.debug(logger_source_id_, "Iter " + std::to_string(iter+1) + ", P" + std::to_string(i) +
                                            ": Pos=" + vector_to_string(swarm[i].position) +
                                            ", Vel=" + vector_to_string(swarm[i].velocity) +
                                            ", Fit=" + std::to_string(current_value) +
                                            ", PBestFit=" + std::to_string(swarm[i].pbest_value));
            }
        }

        if ((iter + 1) % report_interval_ == 0 || iter == iterations_ - 1) {
            std::string msg = "Iter " + std::to_string(iter + 1) + "/" + std::to_string(iterations_) +
                              " | gbest = " + std::to_string(gbest_value);
            if (!use_constriction_factor_) msg += " | omega = " + std::to_string(current_omega);
            msg += " | c1 = " + std::to_string(c1_current) + " | c2 = " + std::to_string(c2_current);
            logger.info(logger_source_id_, msg);
        }
    }

    logger.info(logger_source_id_, "--- PSO Completed ---");
    logger.info(logger_source_id_, "Final gbest value: " + std::to_string(gbest_value));
    logger.info(logger_source_id_, "Final gbest position: " + vector_to_string(gbest_position));

    OptimizationResult result;
    result.bestParameters = gbest_position;
    result.bestObjectiveValue = gbest_value;
    return result;
}

} // namespace epidemic