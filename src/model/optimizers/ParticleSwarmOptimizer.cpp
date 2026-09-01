#include "model/optimizers/ParticleSwarmOptimizer.hpp"
#include "utils/Logger.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>

namespace epidemic {

void ParticleSwarmOptimization::configure(
    const std::map<std::string, double>& settings) {
    
    Logger& logger = Logger::getInstance();
    
    // Configure base PSO parameters
    for (const auto& [key, value] : settings) {
        try {
            // Standard parameters
            if (key == "iterations") {
                if (value <= 0) throw std::invalid_argument("iterations must be positive");
                iterations_ = static_cast<int>(value);
            } else if (key == "swarm_size") {
                if (value <= 0) throw std::invalid_argument("swarm_size must be positive");
                swarm_size_ = static_cast<int>(value);
            } else if (key == "omega_start") {
                if (value < 0) throw std::invalid_argument("omega_start must be non-negative");
                omega_start_ = value;
            } else if (key == "omega_end") {
                if (value < 0) throw std::invalid_argument("omega_end must be non-negative");
                omega_end_ = value;
            } else if (key == "c1_initial") {
                if (value < 0) throw std::invalid_argument("c1_initial must be non-negative");
                c1_initial_ = value;
            } else if (key == "c1_final") {
                if (value < 0) throw std::invalid_argument("c1_final must be non-negative");
                c1_final_ = value;
            } else if (key == "c2_initial") {
                if (value < 0) throw std::invalid_argument("c2_initial must be non-negative");
                c2_initial_ = value;
            } else if (key == "c2_final") {
                if (value < 0) throw std::invalid_argument("c2_final must be non-negative");
                c2_final_ = value;
            } else if (key == "report_interval") {
                if (value <= 0) throw std::invalid_argument("report_interval must be positive");
                report_interval_ = static_cast<int>(value);
            }
            // Enhanced parameters
            else if (key == "variant") {
                int variant_int = static_cast<int>(value);
                if (variant_int < 0 || variant_int > 4) {
                    throw std::invalid_argument("variant must be between 0 and 4");
                }
                variant_ = static_cast<PSOVariant>(variant_int);
            } else if (key == "topology") {
                int topology_int = static_cast<int>(value);
                if (topology_int < 0 || topology_int > 3) {
                    throw std::invalid_argument("topology must be between 0 and 3");
                }
                topology_ = static_cast<TopologyType>(topology_int);
            } else if (key == "use_opposition_learning") {
                use_opposition_learning_ = (value != 0.0);
            } else if (key == "use_parallel") {
                use_parallel_ = (value != 0.0);
            } else if (key == "use_adaptive_parameters") {
                use_adaptive_parameters_ = (value != 0.0);
            } else if (key == "diversity_threshold") {
                diversity_threshold_ = value;
            } else if (key == "restart_threshold") {
                restart_threshold_ = value;
            } else if (key == "quantum_beta") {
                quantum_beta_ = value;
            } else if (key == "levy_alpha") {
                levy_alpha_ = value;
            } else if (key == "max_stagnation") {
                if (value <= 0) throw std::invalid_argument("max_stagnation must be positive");
                max_stagnation_ = static_cast<int>(value);
            } else if (key == "log_evolutionary_state") {
                log_evolutionary_state_ = (value != 0.0);
            }
        } catch (const std::exception& e) {
            logger.error(logger_source_id_, 
                "Error configuring parameter '" + key + "': " + e.what());
            throw;
        }
    }
    
    // Set number of threads for parallel execution
    if (use_parallel_) {
        omp_set_num_threads(omp_get_max_threads());
        logger.info(logger_source_id_, 
            "Parallel execution enabled with " + 
            std::to_string(omp_get_max_threads()) + " threads");
    }
    
    // Log configuration summary
    std::ostringstream config_summary;
    config_summary << "PSO configured: "
                   << "Variant=" << static_cast<int>(variant_)
                   << ", Topology=" << static_cast<int>(topology_)
                   << ", Adaptive=" << use_adaptive_parameters_
                   << ", Parallel=" << use_parallel_;
    logger.info(logger_source_id_, config_summary.str());
}

OptimizationResult ParticleSwarmOptimization::optimize(
    const Eigen::VectorXd& initialParameters,
    IObjectiveFunction& objectiveFunction,
    IParameterManager& parameterManager) {
    
    Logger& logger = Logger::getInstance();
    logger.info(logger_source_id_, "=== Starting PSO Optimization ===");
    
    int n = parameterManager.getParameterCount();
    std::vector<particle> swarm;
    Eigen::VectorXd gbest_position(n);
    double gbest_value;
    double previous_gbest = -std::numeric_limits<double>::infinity();
    
    // Initialize swarm with advanced strategies
    const Eigen::VectorXd* init_params = 
        (initialParameters.size() == n) ? &initialParameters : nullptr;
    
    initializeSwarm(swarm, objectiveFunction, parameterManager, 
                    gbest_position, gbest_value, init_params);
    
    // Main optimization loop
    for (int iter = 0; iter < iterations_; ++iter) {
        // Check for stagnation
        if (std::abs(gbest_value - previous_gbest) < restart_threshold_) {
            stagnation_counter_++;
            if (stagnation_counter_ > max_stagnation_) {
                logger.info(logger_source_id_, 
                    "Stagnation detected at iteration " + std::to_string(iter) + 
                    ". Applying restart strategy.");
                restartSwarm(swarm, objectiveFunction, parameterManager,
                            gbest_position, gbest_value);
                stagnation_counter_ = 0;
            }
        } else {
            stagnation_counter_ = 0;
        }
        previous_gbest = gbest_value;
        
        // Update particles
        updateParticles(swarm, gbest_position, objectiveFunction, 
                       parameterManager, iter);
        
        // Find new global best
        for (const auto& p : swarm) {
            if (p.pbest_value > gbest_value) {
                gbest_value = p.pbest_value;
                gbest_position = p.pbest_position;
                logger.debug(logger_source_id_, 
                    "New global best found: " + std::to_string(gbest_value));
            }
        }
        
        // Apply elitist learning strategy for best particle
        if ((variant_ == PSOVariant::ADAPTIVE || variant_ == PSOVariant::HYBRID) &&
            (iter % 5 == 0)) {  // Apply every 5 iterations
            
            auto best_it = std::max_element(swarm.begin(), swarm.end(),
                [](const particle& a, const particle& b) {
                    return a.pbest_value < b.pbest_value;
                });
            
            std::vector<double> lb(n), ub(n);
            for (int k = 0; k < n; ++k) {
                lb[k] = parameterManager.getLowerBoundForParamIndex(k);
                ub[k] = parameterManager.getUpperBoundForParamIndex(k);
            }
            
            applyElitistLearningStrategy(*best_it, objectiveFunction, lb, ub);
            
            // Update global best if improved
            if (best_it->pbest_value > gbest_value) {
                gbest_value = best_it->pbest_value;
                gbest_position = best_it->pbest_position;
            }
        }
        
        // Progress reporting
        if ((iter + 1) % report_interval_ == 0 || iter == iterations_ - 1) {
            double diversity = calculateSwarmDiversity(swarm);
            
            std::ostringstream msg;
            msg << "Iteration " << (iter + 1) << "/" << iterations_
                << " | Best: " << gbest_value
                << " | Diversity: " << diversity
                << " | Stagnation: " << stagnation_counter_;
            
            if (use_adaptive_parameters_ && log_evolutionary_state_) {
                EvolutionaryState state = estimateEvolutionaryState(swarm, gbest_position);
                msg << " | State: ";
                switch (state) {
                    case EvolutionaryState::EXPLORATION:
                        msg << "Exploration"; break;
                    case EvolutionaryState::EXPLOITATION:
                        msg << "Exploitation"; break;
                    case EvolutionaryState::CONVERGENCE:
                        msg << "Convergence"; break;
                    case EvolutionaryState::JUMPING_OUT:
                        msg << "Jumping Out"; break;
                }
            }
            
            logger.info(logger_source_id_, msg.str());
        }
    }
    
    logger.info(logger_source_id_, "=== PSO Completed ===");
    logger.info(logger_source_id_, "Final best value: " + std::to_string(gbest_value));
    
    std::ostringstream param_stream;
    param_stream << "[";
    for (int i = 0; i < gbest_position.size(); ++i) {
        if (i > 0) param_stream << ", ";
        param_stream << gbest_position[i];
    }
    param_stream << "]";
    logger.debug(logger_source_id_, "Final best position: " + param_stream.str());
    
    OptimizationResult result;
    result.bestParameters = gbest_position;
    result.bestObjectiveValue = gbest_value;
    
    // Estimate covariance from final swarm for Phase 2 transfer
    Eigen::VectorXd mean_position = Eigen::VectorXd::Zero(n);
    for (const auto& p : swarm) {
        mean_position += p.pbest_position;
    }
    mean_position /= swarm_size_;
    
    result.finalCovariance = Eigen::MatrixXd::Zero(n, n);
    for (const auto& p : swarm) {
        Eigen::VectorXd diff = p.pbest_position - mean_position;
        result.finalCovariance += diff * diff.transpose();
    }
    result.finalCovariance /= (swarm_size_ - 1);
    
    // Regularization for numerical stability
    result.finalCovariance += 1e-6 * Eigen::MatrixXd::Identity(n, n);
    
    logger.info(logger_source_id_, "Estimated covariance from swarm for Phase 2 transfer.");
    
    return result;
}

void ParticleSwarmOptimization::initializeSwarm(
    std::vector<particle>& swarm,
    IObjectiveFunction& objectiveFunction,
    IParameterManager& parameterManager,
    Eigen::VectorXd& gbest_position,
    double& gbest_value,
    const Eigen::VectorXd* initial_params) {
    
    Logger& logger = Logger::getInstance();
    int n = parameterManager.getParameterCount();
    swarm.resize(swarm_size_);
    gbest_value = -std::numeric_limits<double>::infinity();
    
    // Get bounds
    std::vector<double> lb(n), ub(n);
    for (int k = 0; k < n; ++k) {
        lb[k] = parameterManager.getLowerBoundForParamIndex(k);
        ub[k] = parameterManager.getUpperBoundForParamIndex(k);
    }
    
    // Pre-generate seeds for thread safety
    std::vector<unsigned int> seeds(swarm_size_);
    for(int i=0; i<swarm_size_; ++i) seeds[i] = static_cast<unsigned int>(rng_());

    // Initialize particles
    #pragma omp parallel for if(use_parallel_)
    for (int i = 0; i < swarm_size_; ++i) {
        swarm[i].position.resize(n);
        swarm[i].velocity.resize(n);
        swarm[i].quantum_position.resize(n);
        
        // Thread-local RNG for parallel execution
        std::mt19937 local_rng(seeds[i]);
        std::uniform_real_distribution<> local_uniform(0.0, 1.0);
        
        if (i == 0 && initial_params != nullptr) {
            swarm[i].position = *initial_params;
            for (int k = 0; k < n; ++k) {
                swarm[i].position[k] = std::clamp(swarm[i].position[k], lb[k], ub[k]);
            }
        } else {
            for (int k = 0; k < n; ++k) {
                swarm[i].position[k] = lb[k] + local_uniform(local_rng) * (ub[k] - lb[k]);
            }
        }
        
        // Initialize velocity
        for (int k = 0; k < n; ++k) {
            double vmax = 0.2 * (ub[k] - lb[k]);
            swarm[i].velocity[k] = -vmax + 2 * vmax * local_uniform(local_rng);
        }
        
        swarm[i].current_fitness = objectiveFunction.calculate(swarm[i].position);
        swarm[i].pbest_position = swarm[i].position;
        swarm[i].pbest_value = swarm[i].current_fitness;
        swarm[i].quantum_position = swarm[i].position;
    }
    
    if (use_opposition_learning_) {
        oppositionBasedInitialization(swarm, lb, ub);
        
        #pragma omp parallel for if(use_parallel_)
        for (int i = 0; i < swarm_size_; ++i) {
            swarm[i].current_fitness = objectiveFunction.calculate(swarm[i].position);
            swarm[i].pbest_value = swarm[i].current_fitness;
            swarm[i].pbest_position = swarm[i].position;
        }
    }
    
    for (const auto& p : swarm) {
        if (p.pbest_value > gbest_value) {
            gbest_value = p.pbest_value;
            gbest_position = p.pbest_position;
        }
    }
    
    logger.info(logger_source_id_, 
        "Initialized swarm with " + std::to_string(swarm_size_) + 
        " particles. Initial best: " + std::to_string(gbest_value));
}

void ParticleSwarmOptimization::updateParticles(
    std::vector<particle>& swarm,
    const Eigen::VectorXd& gbest_position,
    IObjectiveFunction& objectiveFunction,
    IParameterManager& parameterManager,
    int iteration) {
    
    int n = parameterManager.getParameterCount();
    std::vector<double> lb(n), ub(n);
    for (int k = 0; k < n; ++k) {
        lb[k] = parameterManager.getLowerBoundForParamIndex(k);
        ub[k] = parameterManager.getUpperBoundForParamIndex(k);
    }
    
    // Calculate adaptive parameters if enabled
    double omega = omega_start_;
    double c1 = c1_initial_;
    double c2 = c2_initial_;
    
    if (use_adaptive_parameters_) {
        EvolutionaryState state = estimateEvolutionaryState(swarm, gbest_position);
        adaptParameters(state, iteration, omega, c1, c2);
    } else {
        double ratio = (iterations_ > 1) ? 
            static_cast<double>(iteration) / (iterations_ - 1) : 0.0;
        omega = omega_start_ + (omega_end_ - omega_start_) * ratio;
        c1 = c1_initial_ + (c1_final_ - c1_initial_) * ratio;
        c2 = c2_initial_ + (c2_final_ - c2_initial_) * ratio;
    }
    
    Eigen::VectorXd mean_best_position;
    if (variant_ == PSOVariant::QUANTUM) {
        mean_best_position = calculateMeanBestPosition(swarm);
    }
    
    std::vector<unsigned int> seeds(swarm_size_);
    for(int i=0; i<swarm_size_; ++i) seeds[i] = static_cast<unsigned int>(rng_());

    #pragma omp parallel for if(use_parallel_)
    for (int i = 0; i < swarm_size_; ++i) {
        std::mt19937 local_rng(seeds[i]);
        std::uniform_real_distribution<> local_uniform(0.0, 1.0);

        Eigen::VectorXd neighborhood_best = (topology_ == TopologyType::GLOBAL_BEST) ?
            gbest_position : getNeighborhoodBest(swarm, i);
        
        switch (variant_) {
            case PSOVariant::STANDARD:
                standardPSOUpdate(swarm[i], neighborhood_best,
                                omega, c1, c2, lb, ub, local_rng);
                break;
                
            case PSOVariant::QUANTUM:
                quantumPSOUpdate(swarm[i], gbest_position, mean_best_position,
                               iteration, lb, ub, local_rng);
                break;
                
            case PSOVariant::LEVY_FLIGHT:
                levyFlightUpdate(swarm[i], gbest_position, omega, c1, c2, lb, ub, local_rng);
                break;
                
            case PSOVariant::ADAPTIVE:
                // Use standard update with adaptive parameters
                standardPSOUpdate(swarm[i], neighborhood_best,
                                omega, c1, c2, lb, ub, local_rng);
                break;
                
            case PSOVariant::HYBRID:
                // Choose update method based on particle success rate
                if (swarm[i].success_rate < 0.3 && local_uniform(local_rng) < 0.5) {
                    levyFlightUpdate(swarm[i], gbest_position, omega, c1, c2, lb, ub, local_rng);
                } else if (swarm[i].success_rate > 0.7 && local_uniform(local_rng) < 0.3) {
                    quantumPSOUpdate(swarm[i], gbest_position, mean_best_position,
                                   iteration, lb, ub, local_rng);
                } else {
                    standardPSOUpdate(swarm[i], neighborhood_best,
                                    omega, c1, c2, lb, ub, local_rng);
                }
                break;
        }
        
        // Evaluate new position
        double new_fitness = objectiveFunction.calculate(swarm[i].position);
        swarm[i].current_fitness = new_fitness;
        
        // Update personal best and success tracking
        swarm[i].total_updates++;
        if (new_fitness > swarm[i].pbest_value) {
            swarm[i].pbest_value = new_fitness;
            swarm[i].pbest_position = swarm[i].position;
            swarm[i].success_count++;
        }
        swarm[i].success_rate = (swarm[i].total_updates > 0) ?
            static_cast<double>(swarm[i].success_count) / swarm[i].total_updates : 0.0;
    }
}

ParticleSwarmOptimization::EvolutionaryState 
ParticleSwarmOptimization::estimateEvolutionaryState(
    const std::vector<particle>& swarm,
    const Eigen::VectorXd& gbest_position) {
    
    double evolutionary_factor = calculateEvolutionaryFactor(swarm, gbest_position);
    
    if (evolutionary_factor > 0.7) {
        return EvolutionaryState::EXPLORATION;
    } else if (evolutionary_factor > 0.4) {
        return EvolutionaryState::EXPLOITATION;
    } else if (evolutionary_factor > 0.2) {
        return EvolutionaryState::CONVERGENCE;
    } else {
        return EvolutionaryState::JUMPING_OUT;
    }
}

double ParticleSwarmOptimization::calculateEvolutionaryFactor(
    const std::vector<particle>& swarm,
    const Eigen::VectorXd& gbest_position) {
    
    // Calculate mean distance from global best
    double mean_distance = 0.0;
    double max_distance = 0.0;
    
    for (const auto& p : swarm) {
        double dist = (p.position - gbest_position).norm();
        mean_distance += dist;
        max_distance = std::max(max_distance, dist);
    }
    mean_distance /= swarm_size_;
    
    // Calculate fitness diversity
    double mean_fitness = 0.0;
    double max_fitness = -std::numeric_limits<double>::infinity();
    double min_fitness = std::numeric_limits<double>::infinity();
    
    for (const auto& p : swarm) {
        mean_fitness += p.current_fitness;
        max_fitness = std::max(max_fitness, p.current_fitness);
        min_fitness = std::min(min_fitness, p.current_fitness);
    }
    mean_fitness /= swarm_size_;
    
    double fitness_range = (max_fitness - min_fitness) > 1e-10 ? 
        (max_fitness - min_fitness) : 1e-10;
    
    double distance_factor = (max_distance > 0) ? mean_distance / max_distance : 0.0;
    double fitness_factor = (max_fitness - mean_fitness) / fitness_range;
    
    return 0.5 * distance_factor + 0.5 * (1.0 - fitness_factor);
}

void ParticleSwarmOptimization::adaptParameters(
    EvolutionaryState state,
    int iteration,
    double& omega,
    double& c1,
    double& c2) {
    
    double ratio = (iterations_ > 1) ? 
        static_cast<double>(iteration) / (iterations_ - 1) : 0.0;
    
    switch (state) {
        case EvolutionaryState::EXPLORATION:
            // High exploration: high omega, balanced c1 and c2
            omega = 0.9 - 0.2 * ratio;
            c1 = 1.5 + 0.5 * std::sin(ratio * M_PI);
            c2 = 1.5 - 0.5 * std::sin(ratio * M_PI);
            break;
            
        case EvolutionaryState::EXPLOITATION:
            // Exploitation: medium omega, increasing social component
            omega = 0.7 - 0.3 * ratio;
            c1 = 2.0 - ratio;
            c2 = 1.0 + ratio;
            break;
            
        case EvolutionaryState::CONVERGENCE:
            // Fine-tuning: low omega, high social component
            omega = 0.4 - 0.3 * ratio;
            c1 = 1.0 - 0.5 * ratio;
            c2 = 2.0 + 0.5 * ratio;
            break;
            
        case EvolutionaryState::JUMPING_OUT:
            // Escape local optima: increase randomness
            omega = 0.9 + 0.1 * uniform_dist_(rng_);
            c1 = 2.5 + uniform_dist_(rng_);
            c2 = 0.5 + uniform_dist_(rng_);
            break;
    }
    
    // Ensure parameters are within reasonable bounds
    omega = std::clamp(omega, 0.1, 1.0);
    c1 = std::clamp(c1, 0.0, 4.0);
    c2 = std::clamp(c2, 0.0, 4.0);
}

void ParticleSwarmOptimization::oppositionBasedInitialization(
    std::vector<particle>& swarm,
    const std::vector<double>& lb,
    const std::vector<double>& ub) {
    
    Logger& logger = Logger::getInstance();
    int n = swarm[0].position.size();
    std::vector<particle> opposite_swarm(swarm_size_);
    
    for (int i = 0; i < swarm_size_; ++i) {
        opposite_swarm[i].position.resize(n);
        opposite_swarm[i].velocity.resize(n);
        opposite_swarm[i].quantum_position.resize(n);
        
        for (int k = 0; k < n; ++k) {
            opposite_swarm[i].position[k] = lb[k] + ub[k] - swarm[i].position[k];
            opposite_swarm[i].velocity[k] = -swarm[i].velocity[k];
        }
        
        opposite_swarm[i].quantum_position = opposite_swarm[i].position;
        opposite_swarm[i].pbest_position = opposite_swarm[i].position;
    }
    
    // Select best from original and opposite particles
    std::vector<std::pair<double, int>> fitness_indices;
    fitness_indices.reserve(2 * swarm_size_);
    
    for (int i = 0; i < swarm_size_; ++i) {
        fitness_indices.push_back({swarm[i].pbest_value, i});
        fitness_indices.push_back({opposite_swarm[i].pbest_value, i + swarm_size_});
    }
    
    std::sort(fitness_indices.begin(), fitness_indices.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::vector<particle> new_swarm(swarm_size_);
    for (int i = 0; i < swarm_size_; ++i) {
        int idx = fitness_indices[i].second;
        if (idx < swarm_size_) {
            new_swarm[i] = swarm[idx];
        } else {
            new_swarm[i] = opposite_swarm[idx - swarm_size_];
        }
    }
    
    swarm = std::move(new_swarm);
    logger.debug(logger_source_id_, "Applied opposition-based learning initialization");
}

void ParticleSwarmOptimization::standardPSOUpdate(
    particle& p,
    const Eigen::VectorXd& lbest_position,
    double omega, double c1, double c2,
    const std::vector<double>& lb,
    const std::vector<double>& ub,
    std::mt19937& rng) {
    
    int n = p.position.size();
    std::uniform_real_distribution<> local_uniform(0.0, 1.0);
    
    Eigen::VectorXd r1(n), r2(n);
    for (int i = 0; i < n; ++i) {
        r1[i] = local_uniform(rng);
        r2[i] = local_uniform(rng);
    }
    
    Eigen::VectorXd cognitive_term = c1 * r1.cwiseProduct(
        p.pbest_position - p.position);
    Eigen::VectorXd social_term = c2 * r2.cwiseProduct(
        lbest_position - p.position);
    
    p.velocity = omega * p.velocity + cognitive_term + social_term;
    
    for (int k = 0; k < n; ++k) {
        double vmax = 0.2 * (ub[k] - lb[k]);
        p.velocity[k] = std::clamp(p.velocity[k], -vmax, vmax);
    }
    
    p.position += p.velocity;
    
    // Boundary handling with reflection and velocity dampening
    for (int k = 0; k < n; ++k) {
        if (p.position[k] < lb[k]) {
            p.position[k] = lb[k] + std::abs(p.position[k] - lb[k]);
            p.velocity[k] *= -0.5;
        } else if (p.position[k] > ub[k]) {
            p.position[k] = ub[k] - std::abs(p.position[k] - ub[k]);
            p.velocity[k] *= -0.5;
        }
        p.position[k] = std::clamp(p.position[k], lb[k], ub[k]);
    }
}

void ParticleSwarmOptimization::quantumPSOUpdate(
    particle& p,
    const Eigen::VectorXd& gbest_position,
    const Eigen::VectorXd& mean_best_position,
    int iteration,
    const std::vector<double>& lb,
    const std::vector<double>& ub,
    std::mt19937& rng) {
    
    int n = p.position.size();
    std::uniform_real_distribution<> local_uniform(0.0, 1.0);
    
    // Weighted average of pbest and gbest
    double phi = local_uniform(rng);
    Eigen::VectorXd attractor = phi * p.pbest_position + 
                               (1 - phi) * gbest_position;
    
    double beta = quantum_beta_ * (1.0 - 0.5 * static_cast<double>(iteration) / iterations_);
    
    for (int k = 0; k < n; ++k) {
        double u = local_uniform(rng);
        double L = 2.0 * beta * std::abs(mean_best_position[k] - p.position[k]);
        
        if (local_uniform(rng) < 0.5) {
            p.position[k] = attractor[k] + L * std::log(1.0 / u);
        } else {
            p.position[k] = attractor[k] - L * std::log(1.0 / u);
        }
        
        p.position[k] = std::clamp(p.position[k], lb[k], ub[k]);
    }
    
    p.quantum_position = p.position;
}

void ParticleSwarmOptimization::levyFlightUpdate(
    particle& p,
    const Eigen::VectorXd& gbest_position,
    double omega, double c1, double c2,
    const std::vector<double>& lb,
    const std::vector<double>& ub,
    std::mt19937& rng) {
    
    standardPSOUpdate(p, gbest_position, omega, c1, c2, lb, ub, rng);
    
    std::uniform_real_distribution<> local_uniform(0.0, 1.0);

    double levy_prob = 0.1 * (1.0 + p.success_rate);
    if (local_uniform(rng) < levy_prob) {
        int n = p.position.size();
        Eigen::VectorXd levy_step = generateLevyVector(n, rng);
        
        double step_scale = 0.01 * (1.0 - stagnation_counter_ / static_cast<double>(max_stagnation_));
        
        for (int k = 0; k < n; ++k) {
            double scale = step_scale * (ub[k] - lb[k]);
            p.position[k] += scale * levy_step[k];
            p.position[k] = std::clamp(p.position[k], lb[k], ub[k]);
        }
    }
}

double ParticleSwarmOptimization::calculateSwarmDiversity(
    const std::vector<particle>& swarm) {
    
    int n = swarm[0].position.size();
    
    Eigen::VectorXd swarm_centroid = Eigen::VectorXd::Zero(n);
    for (const auto& p : swarm) {
        swarm_centroid += p.position;
    }
    swarm_centroid /= swarm_size_;
    
    double avg_distance = 0.0;
    double max_distance = 0.0;
    
    for (const auto& p : swarm) {
        double dist = (p.position - swarm_centroid).norm();
        avg_distance += dist;
        max_distance = std::max(max_distance, dist);
    }
    avg_distance /= swarm_size_;
    
    return (max_distance > 0) ? avg_distance / max_distance : 0.0;
}

void ParticleSwarmOptimization::applyElitistLearningStrategy(
    particle& best_particle,
    IObjectiveFunction& objectiveFunction,
    const std::vector<double>& lb,
    const std::vector<double>& ub) {
    
    int n = best_particle.position.size();
    Eigen::VectorXd trial_position = best_particle.position;
    
    double sigma_scale = 0.1 * std::exp(-2.0 * best_particle.success_rate);
    
    for (int attempt = 0; attempt < 3; ++attempt) {
        for (int k = 0; k < n; ++k) {
            double sigma = sigma_scale * (ub[k] - lb[k]);
            trial_position[k] = best_particle.position[k] + sigma * normal_dist_(rng_);
            trial_position[k] = std::clamp(trial_position[k], lb[k], ub[k]);
        }
        
        double trial_fitness = objectiveFunction.calculate(trial_position);
        
        if (trial_fitness > best_particle.pbest_value) {
            best_particle.position = trial_position;
            best_particle.pbest_position = trial_position;
            best_particle.pbest_value = trial_fitness;
            best_particle.current_fitness = trial_fitness;
            
            Logger& logger = Logger::getInstance();
            logger.debug(logger_source_id_, 
                "ELS improved best particle: " + std::to_string(trial_fitness));
            break;
        }
        
        sigma_scale *= 0.5;
    }
}

void ParticleSwarmOptimization::restartSwarm(
    std::vector<particle>& swarm,
    IObjectiveFunction& objectiveFunction,
    IParameterManager& parameterManager,
    Eigen::VectorXd& gbest_position,
    double& gbest_value,
    int keep_best_count) {
    
    Logger& logger = Logger::getInstance();
    
    std::sort(swarm.begin(), swarm.end(),
        [](const particle& a, const particle& b) {
            return a.pbest_value > b.pbest_value;
        });
    
    std::vector<particle> elite_particles(
        swarm.begin(), 
        swarm.begin() + std::min(keep_best_count, swarm_size_));
    
    int n = parameterManager.getParameterCount();
    std::vector<double> lb(n), ub(n);
    for (int k = 0; k < n; ++k) {
        lb[k] = parameterManager.getLowerBoundForParamIndex(k);
        ub[k] = parameterManager.getUpperBoundForParamIndex(k);
    }
    
    std::vector<unsigned int> seeds(swarm_size_);
    for(int i=0; i<swarm_size_; ++i) seeds[i] = static_cast<unsigned int>(rng_());

    #pragma omp parallel for if(use_parallel_)
    for (int i = keep_best_count; i < swarm_size_; ++i) {
        std::mt19937 local_rng(seeds[i]);
        std::uniform_real_distribution<> local_uniform(0.0, 1.0);
        std::normal_distribution<> local_normal(0.0, 1.0);
        
        int elite_idx = i % elite_particles.size();
        
        for (int k = 0; k < n; ++k) {
            if (local_uniform(local_rng) < 0.7) {
                double range = ub[k] - lb[k];
                double sigma = 0.3 * range * (1.0 + 0.5 * local_uniform(local_rng));
                swarm[i].position[k] = elite_particles[elite_idx].position[k] + 
                                      sigma * local_normal(local_rng);
            } else {
                swarm[i].position[k] = lb[k] + local_uniform(local_rng) * (ub[k] - lb[k]);
            }
            
            swarm[i].position[k] = std::clamp(swarm[i].position[k], lb[k], ub[k]);
            
            double vmax = 0.2 * (ub[k] - lb[k]);
            swarm[i].velocity[k] = -vmax + 2 * vmax * local_uniform(local_rng);
        }
        
        swarm[i].current_fitness = objectiveFunction.calculate(swarm[i].position);
        swarm[i].pbest_position = swarm[i].position;
        swarm[i].pbest_value = swarm[i].current_fitness;
        swarm[i].quantum_position = swarm[i].position;
        swarm[i].success_count = 0;
        swarm[i].total_updates = 0;
        swarm[i].success_rate = 0.0;
    }
    
    for (int i = 0; i < std::min(keep_best_count, swarm_size_); ++i) {
        swarm[i] = elite_particles[i];
    }
    
    gbest_value = swarm[0].pbest_value;
    gbest_position = swarm[0].pbest_position;
    
    logger.info(logger_source_id_, 
        "Swarm restarted, keeping " + std::to_string(elite_particles.size()) + 
        " elite particles. Best preserved: " + std::to_string(gbest_value));
}

Eigen::VectorXd ParticleSwarmOptimization::getNeighborhoodBest(
    const std::vector<particle>& swarm,
    int particle_idx) {
    
    std::vector<int> neighbors = getNeighbors(particle_idx);
    
    Eigen::VectorXd best_position = swarm[particle_idx].pbest_position;
    double best_value = swarm[particle_idx].pbest_value;
    
    for (int neighbor_idx : neighbors) {
        if (neighbor_idx >= 0 && neighbor_idx < swarm_size_ &&
            swarm[neighbor_idx].pbest_value > best_value) {
            best_value = swarm[neighbor_idx].pbest_value;
            best_position = swarm[neighbor_idx].pbest_position;
        }
    }
    
    return best_position;
}

std::vector<int> ParticleSwarmOptimization::getNeighbors(int particle_idx) {
    std::vector<int> neighbors;
    
    switch (topology_) {
        case TopologyType::GLOBAL_BEST:
            // All particles are neighbors
            neighbors.reserve(swarm_size_);
            for (int i = 0; i < swarm_size_; ++i) {
                neighbors.push_back(i);
            }
            break;
            
        case TopologyType::LOCAL_BEST:
            {
                int k = 2;
                neighbors.push_back(particle_idx);
                for (int j = 1; j <= k; ++j) {
                    neighbors.push_back((particle_idx - j + swarm_size_) % swarm_size_);
                    neighbors.push_back((particle_idx + j) % swarm_size_);
                }
            }
            break;
            
        case TopologyType::VON_NEUMANN:
            {
                int grid_size = static_cast<int>(std::ceil(std::sqrt(swarm_size_)));
                int row = particle_idx / grid_size;
                int col = particle_idx % grid_size;
                
                neighbors.push_back(particle_idx);
                
                if (row > 0) {
                    int idx = (row - 1) * grid_size + col;
                    if (idx < swarm_size_) neighbors.push_back(idx);
                }
                if (row < grid_size - 1) {
                    int idx = (row + 1) * grid_size + col;
                    if (idx < swarm_size_) neighbors.push_back(idx);
                }
                if (col > 0) {
                    int idx = row * grid_size + (col - 1);
                    if (idx < swarm_size_) neighbors.push_back(idx);
                }
                if (col < grid_size - 1) {
                    int idx = row * grid_size + (col + 1);
                    if (idx < swarm_size_) neighbors.push_back(idx);
                }
            }
            break;
            
        case TopologyType::RANDOM_DYNAMIC:
            {
                neighbors.push_back(particle_idx);
                std::vector<int> candidates;
                candidates.reserve(swarm_size_ - 1);
                
                for (int i = 0; i < swarm_size_; ++i) {
                    if (i != particle_idx) candidates.push_back(i);
                }
                
                std::shuffle(candidates.begin(), candidates.end(), rng_);
                int k = std::min(4, static_cast<int>(candidates.size()));
                neighbors.insert(neighbors.end(), 
                               candidates.begin(), 
                               candidates.begin() + k);
            }
            break;
    }
    
    return neighbors;
}

Eigen::VectorXd ParticleSwarmOptimization::generateLevyVector(int dimension, std::mt19937& rng) {
    Eigen::VectorXd levy_vector(dimension);
    
    for (int i = 0; i < dimension; ++i) {
        levy_vector[i] = generateLevyNumber(rng);
    }
    
    return levy_vector;
}

double ParticleSwarmOptimization::generateLevyNumber(std::mt19937& rng) {
    // Mantegna's algorithm for Lévy distribution
    double sigma_u = std::pow(
        std::tgamma(1 + levy_alpha_) * std::sin(M_PI * levy_alpha_ / 2) /
        (std::tgamma((1 + levy_alpha_) / 2) * levy_alpha_ * 
         std::pow(2, (levy_alpha_ - 1) / 2)), 
        1.0 / levy_alpha_);
    
    std::normal_distribution<> local_normal(0.0, 1.0);
    double u = local_normal(rng) * sigma_u;
    double v = std::max(std::abs(local_normal(rng)), 1e-10);
    
    double levy_step = u / std::pow(v, 1.0 / levy_alpha_);
    
    const double max_levy_magnitude = 100.0;
    return std::clamp(levy_step, -max_levy_magnitude, max_levy_magnitude);
}

Eigen::VectorXd ParticleSwarmOptimization::calculateMeanBestPosition(
    const std::vector<particle>& swarm) {
    
    int n = swarm[0].position.size();
    Eigen::VectorXd mean_best = Eigen::VectorXd::Zero(n);
    
    for (const auto& p : swarm) {
        mean_best += p.pbest_position;
    }
    
    return mean_best / swarm_size_;
}

} // namespace epidemic