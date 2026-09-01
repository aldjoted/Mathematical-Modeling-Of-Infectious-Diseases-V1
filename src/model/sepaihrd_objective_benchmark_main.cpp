#include <Eigen/Dense>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "exceptions/Exceptions.hpp"
#include "model/AgeSEPAIHRDModel.hpp"
#include "model/PieceWiseConstantNPIStrategy.hpp"
#include "model/ModelConstants.hpp"
#include "model/objectives/SEPAIHRDObjectiveFunction.hpp"
#include "model/parameters/SEPAIHRDParameterManager.hpp"
#include "sir_age_structured/caching/SimulationCache.hpp"
#include "sir_age_structured/interfaces/IObjectiveFunction.hpp"
#include "sir_age_structured/optimizers/HillClimbingOptimizer.hpp"
#include "sir_age_structured/optimizers/MetropolisHastingsSampler.hpp"
#include "sir_age_structured/solvers/Dopri5SolverStrategy.hpp"
#include "utils/FileUtils.hpp"
#include "utils/GetCalibrationData.hpp"
#include "utils/Logger.hpp"
#include "utils/ReadCalibrationConfiguration.hpp"
#include "utils/ReadContactMatrix.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using epidemic::AgeSEPAIHRDModel;
using epidemic::CalibrationData;
using epidemic::ConstraintMode;
using epidemic::DataFormatException;
using epidemic::Dopri5SolverStrategy;
using epidemic::InvalidParameterException;
using epidemic::Logger;
using epidemic::LogLevel;
using epidemic::PiecewiseConstantNpiStrategy;
using epidemic::SEPAIHRDObjectiveFunction;
using epidemic::SEPAIHRDParameterManager;
using epidemic::SEPAIHRDParameters;
using epidemic::SimulationCache;

namespace {

struct Args {
    int repeats = 200;
    int jitters = 200;
    size_t cacheSize = 10000;
    bool useCache = true;
    int seed = 1;
    int threads = 0; // 0 => leave OpenMP default
    std::string startDate = "2020-03-01";
    std::string endDate = "2020-12-31";
    std::string constraintMode = "opt"; // opt|mcmc

    // Settings file paths
    std::string hillSettingsPath;
    std::string mcmcSettingsPath;

    // Benchmark mode
    std::string mode = "micro"; // micro|hill|mcmc|hillmcmc|all

    // Iteration overrides for long-running optimizers
    bool useFileIters = false;
    int hillIters = 200;
    int mcmcIters = 2000;
};

void printUsage(const char* programName) {
    std::cout
        << "Usage: " << programName
        << " [--repeats N] [--jitters N] [--cache-size N] [--no-cache] [--seed N] [--threads N]"
        << " [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--constraints opt|mcmc]\n";
    std::cout
        << "       " << programName
        << " --mode micro|hill|mcmc|hillmcmc|all [--hill-settings PATH] [--mcmc-settings PATH]"
        << " [--hill-iters N] [--mcmc-iters N] [--use-file-iters]\n";
}

Args parseArgs(int argc, char** argv) {
    Args args;
    args.hillSettingsPath = FileUtils::joinPaths(FileUtils::getProjectRoot(), "data/configuration/hill_climbing_settings.txt");
    args.mcmcSettingsPath = FileUtils::joinPaths(FileUtils::getProjectRoot(), "data/configuration/mcmc_settings.txt");

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto requireValue = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value after ") + flag);
            }
            return std::string(argv[++i]);
        };

        if (a == "--help" || a == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        }
        if (a == "--no-cache") {
            args.useCache = false;
            continue;
        }
        if (a == "--repeats") {
            args.repeats = std::stoi(requireValue("--repeats"));
            continue;
        }
        if (a == "--jitters") {
            args.jitters = std::stoi(requireValue("--jitters"));
            continue;
        }
        if (a == "--cache-size") {
            args.cacheSize = static_cast<size_t>(std::stoull(requireValue("--cache-size")));
            continue;
        }
        if (a == "--seed") {
            args.seed = std::stoi(requireValue("--seed"));
            continue;
        }
        if (a == "--threads") {
            args.threads = std::stoi(requireValue("--threads"));
            continue;
        }
        if (a == "--start") {
            args.startDate = requireValue("--start");
            continue;
        }
        if (a == "--end") {
            args.endDate = requireValue("--end");
            continue;
        }
        if (a == "--constraints") {
            args.constraintMode = requireValue("--constraints");
            continue;
        }

        if (a == "--mode") {
            args.mode = requireValue("--mode");
            continue;
        }
        if (a == "--hill-settings") {
            args.hillSettingsPath = requireValue("--hill-settings");
            continue;
        }
        if (a == "--mcmc-settings") {
            args.mcmcSettingsPath = requireValue("--mcmc-settings");
            continue;
        }
        if (a == "--use-file-iters") {
            args.useFileIters = true;
            continue;
        }
        if (a == "--hill-iters") {
            args.hillIters = std::stoi(requireValue("--hill-iters"));
            continue;
        }
        if (a == "--mcmc-iters") {
            args.mcmcIters = std::stoi(requireValue("--mcmc-iters"));
            continue;
        }

        throw std::runtime_error("Unknown argument: " + a);
    }

    if (args.repeats < 0 || args.jitters < 0) {
        throw std::runtime_error("repeats/jitters must be non-negative");
    }
    if (args.constraintMode != "opt" && args.constraintMode != "mcmc") {
        throw std::runtime_error("--constraints must be 'opt' or 'mcmc'");
    }

    if (args.mode != "micro" && args.mode != "hill" && args.mode != "mcmc" && args.mode != "hillmcmc" && args.mode != "all") {
        throw std::runtime_error("--mode must be one of: micro, hill, mcmc, hillmcmc, all");
    }
    return args;
}

std::shared_ptr<PiecewiseConstantNpiStrategy> createNpiStrategy(
    const SEPAIHRDParameters& params,
    const std::vector<std::string>& all_kappa_parameter_names,
    const std::map<std::string, std::pair<double, double>>& overall_param_bounds,
    int fixed_kappa_model_index = 0)
{
    std::map<std::string, std::pair<double, double>> npi_specific_bounds;
    for (const auto& name : all_kappa_parameter_names) {
        if (name == "kappa_1") continue; // keep baseline fixed
        auto it = overall_param_bounds.find(name);
        if (it != overall_param_bounds.end()) {
            npi_specific_bounds[name] = it->second;
        }
    }

    const double baseline_kappa_val = params.kappa_values.at(fixed_kappa_model_index);
    const double baseline_end_time_val = params.kappa_end_times.at(fixed_kappa_model_index);
    const bool is_baseline_fixed_val = true;

    std::vector<double> npi_end_times_after_baseline;
    std::vector<double> npi_values_after_baseline;
    std::vector<std::string> param_names_for_npi_values;

    if (params.kappa_end_times.size() > static_cast<size_t>(fixed_kappa_model_index + 1)) {
        npi_end_times_after_baseline.assign(
            params.kappa_end_times.begin() + fixed_kappa_model_index + 1,
            params.kappa_end_times.end());
        npi_values_after_baseline.assign(
            params.kappa_values.begin() + fixed_kappa_model_index + 1,
            params.kappa_values.end());
        param_names_for_npi_values.assign(
            all_kappa_parameter_names.begin() + fixed_kappa_model_index + 1,
            all_kappa_parameter_names.end());
    }

    return std::make_shared<PiecewiseConstantNpiStrategy>(
        npi_end_times_after_baseline,
        npi_values_after_baseline,
        npi_specific_bounds,
        baseline_kappa_val,
        baseline_end_time_val,
        is_baseline_fixed_val,
        param_names_for_npi_values);
}

class NullSimulationCache final : public epidemic::ISimulationCache {
public:
    std::optional<double> get(const Eigen::VectorXd&) override { return std::nullopt; }
    void set(const Eigen::VectorXd&, double) override {}
    void clear() override {}
    size_t size() const override { return 0; }
    std::string createCacheKey(const Eigen::VectorXd&) const override { return std::string(); }
    bool getLikelihood(const std::string&, double&) override { return false; }
    void storeLikelihood(const std::string&, double) override {}
};

class CountingObjectiveFunction final : public epidemic::IObjectiveFunction {
public:
    explicit CountingObjectiveFunction(epidemic::IObjectiveFunction& inner)
        : inner_(inner) {}

    double calculate(const Eigen::VectorXd& parameters) const override {
        calls_.fetch_add(1, std::memory_order_relaxed);
        return inner_.calculate(parameters);
    }

    const std::vector<std::string>& getParameterNames() const override { return inner_.getParameterNames(); }

    void resetCalls() const { calls_.store(0, std::memory_order_relaxed); }
    size_t calls() const { return calls_.load(std::memory_order_relaxed); }

private:
    epidemic::IObjectiveFunction& inner_;
    mutable std::atomic<size_t> calls_{0};
};

void printCacheStats(const SimulationCache* sim_cache_ptr) {
    if (sim_cache_ptr == nullptr) return;

    const size_t calls = sim_cache_ptr->getLikelihoodCalls();
    const size_t hits = sim_cache_ptr->getLikelihoodHits();
    const size_t stores = sim_cache_ptr->storeLikelihoodCalls();
    const double hit_rate = (calls == 0) ? 0.0 : (static_cast<double>(hits) / static_cast<double>(calls));

    std::cout << "Cache stats: size=" << sim_cache_ptr->size() << ", get_calls=" << calls
              << ", hits=" << hits << ", hit_rate=" << hit_rate * 100.0 << "%"
              << ", stores=" << stores << "\n";
}

} // namespace

int main(int argc, char** argv) {
    Logger::getInstance().setLogLevel(LogLevel::INFO);

    try {
        const Args args = parseArgs(argc, argv);

#ifdef _OPENMP
        if (args.threads > 0) {
            omp_set_num_threads(args.threads);
        }
#endif

        const int num_age_classes = epidemic::constants::DEFAULT_NUM_AGE_CLASSES;
        const int fixed_kappa_model_index = 0;
        const std::string project_root = FileUtils::getProjectRoot();

        const std::string data_path = FileUtils::joinPaths(project_root, "data/processed/processed_data.csv");
        const std::string bounds_file = FileUtils::joinPaths(project_root, "data/configuration/param_bounds.txt");
        const std::string proposal_file = FileUtils::joinPaths(project_root, "data/configuration/proposal_sigmas.txt");
        const std::string params_to_calibrate_file = FileUtils::joinPaths(project_root, "data/configuration/params_to_calibrate.txt");
        const std::string init_guess_file = FileUtils::joinPaths(project_root, "data/configuration/initial_guess.txt");
        const std::string contact_matrix_path = FileUtils::joinPaths(project_root, "data/contacts.csv");

        CalibrationData data(data_path, args.startDate, args.endDate);
        MatrixXd C = epidemic::readMatrixFromCSV(contact_matrix_path, num_age_classes, num_age_classes);

        SEPAIHRDParameters params = readSEPAIHRDParameters(init_guess_file, num_age_classes);
        params.N = data.getPopulationByAgeGroup();
        params.M_baseline = C;
        if (!params.validate()) {
            throw DataFormatException("sepaihrd_objective_benchmark", "SEPAIHRDParameters validation failed");
        }

        std::vector<std::string> all_kappa_parameter_names;
        all_kappa_parameter_names.reserve(params.kappa_values.size());
        for (size_t i = 0; i < params.kappa_values.size(); ++i) {
            all_kappa_parameter_names.push_back("kappa_" + std::to_string(i + 1));
        }

        std::map<std::string, std::pair<double, double>> overall_param_bounds = readParamBounds(bounds_file);
        std::map<std::string, double> proposal_sigmas = readProposalSigmas(proposal_file);
        std::vector<std::string> params_to_calibrate = readParamsToCalibrate(params_to_calibrate_file);

        // Time points (run-up + observed)
        const double runup_days = params.runup_days;
        const int num_days = data.getNumDataPoints();
        std::vector<double> time_points;
        time_points.reserve(static_cast<size_t>(static_cast<int>(runup_days) + num_days));
        for (int t = -static_cast<int>(runup_days); t < num_days; ++t) {
            time_points.push_back(static_cast<double>(t));
        }

        // Initial state + run-up seeding (mirrors sepaihrd_age_structured_main)
        VectorXd initial_state = data.getInitialSEPAIHRDState(
            params.sigma, params.gamma_p, params.gamma_A, params.gamma_I, params.p, params.h);

        const int n_ages = num_age_classes;
        if (runup_days > 0.0 && params.seed_exposed > 0.0) {
            const double total_pop = params.N.sum();
            for (int i = 0; i < n_ages; ++i) {
                const double age_fraction = params.N(i) / total_pop;
                initial_state(i + 1 * n_ages) = params.seed_exposed * age_fraction; // E
                initial_state(i + 2 * n_ages) = 0.0;                                // P
                initial_state(i + 3 * n_ages) = 0.0;                                // A
                initial_state(i + 4 * n_ages) = 0.0;                                // I
                initial_state(i + 5 * n_ages) = 0.0;                                // H
                initial_state(i + 6 * n_ages) = 0.0;                                // ICU
                initial_state(i + 7 * n_ages) = 0.0;                                // R
                initial_state(i + 8 * n_ages) = 0.0;                                // D
                initial_state(i + 9 * n_ages) = 0.0;                                // CumH
                initial_state(i + 10 * n_ages) = 0.0;                               // CumICU
            }
        } else {
            initial_state.segment(1 * n_ages, n_ages) *= params.E0_multiplier;
            initial_state.segment(2 * n_ages, n_ages) *= params.P0_multiplier;
            initial_state.segment(3 * n_ages, n_ages) *= params.A0_multiplier;
            initial_state.segment(4 * n_ages, n_ages) *= params.I0_multiplier;
            initial_state.segment(5 * n_ages, n_ages) *= params.H0_multiplier;
            initial_state.segment(6 * n_ages, n_ages) *= params.ICU0_multiplier;
            initial_state.segment(7 * n_ages, n_ages) *= params.R0_multiplier;
            initial_state.segment(8 * n_ages, n_ages) *= params.D0_multiplier;
        }

        for (int i = 0; i < n_ages; ++i) {
            double sum_non_S = 0.0;
            for (int j = 1; j < epidemic::constants::NUM_POPULATION_COMPARTMENTS_SEPAIHRD; ++j) {
                sum_non_S += initial_state(j * n_ages + i);
            }
            initial_state(i) = (sum_non_S > params.N(i)) ? 0.0 : (params.N(i) - sum_non_S);
        }

        auto solver_strategy = std::make_shared<Dopri5SolverStrategy>();
        constexpr double abs_err = 1.0e-6;
        constexpr double rel_err = 1.0e-6;

        auto npi_strategy = createNpiStrategy(params, all_kappa_parameter_names, overall_param_bounds, fixed_kappa_model_index);
        auto model = std::make_shared<AgeSEPAIHRDModel>(params, npi_strategy);

        SEPAIHRDParameterManager parameterManager(model, params_to_calibrate, proposal_sigmas, overall_param_bounds);
        if (args.constraintMode == "mcmc") {
            parameterManager.setConstraintMode(ConstraintMode::MCMC_REFLECT);
        } else {
            parameterManager.setConstraintMode(ConstraintMode::OPTIMIZATION_CLAMP);
        }

        VectorXd base_params = parameterManager.getCurrentParameters();

        std::unique_ptr<epidemic::ISimulationCache> cache_owner;
        epidemic::ISimulationCache* cache_ptr = nullptr;
        SimulationCache* sim_cache_ptr = nullptr;

        if (args.useCache) {
            auto sim_cache = std::make_unique<SimulationCache>(args.cacheSize);
            sim_cache_ptr = sim_cache.get();
            cache_ptr = sim_cache.get();
            cache_owner = std::move(sim_cache);
            sim_cache_ptr->clear();
            sim_cache_ptr->resetLikelihoodStats();
        } else {
            cache_owner = std::make_unique<NullSimulationCache>();
            cache_ptr = cache_owner.get();
        }

        SEPAIHRDObjectiveFunction objective(
            model,
            parameterManager,
            *cache_ptr,
            data,
            time_points,
            initial_state,
            solver_strategy,
            abs_err,
            rel_err);

        CountingObjectiveFunction counting_objective(objective);

        // RNG for jittered parameters (uses configured proposal sigmas)
        std::mt19937 rng(static_cast<uint32_t>(args.seed));
        std::normal_distribution<double> normal(0.0, 1.0);

        auto now = []() { return std::chrono::steady_clock::now(); };
        auto ms = [](auto dt) {
            return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(dt).count();
        };

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\n=== SEPAIHRD Objective Benchmark ===\n";
#ifdef _OPENMP
        std::cout << "OpenMP max threads: " << omp_get_max_threads() << "\n";
#endif
        std::cout << "Cache enabled: " << (args.useCache ? "yes" : "no") << "\n";
        if (args.useCache) {
            std::cout << "Cache capacity: " << args.cacheSize << "\n";
        }
        std::cout << "Mode: " << args.mode << "\n";
        std::cout << "Constraint mode: " << args.constraintMode << "\n";
        std::cout << "Data window: " << args.startDate << " .. " << args.endDate << "\n";

        auto run_micro = [&]() {
            if (sim_cache_ptr != nullptr) {
                sim_cache_ptr->resetLikelihoodStats();
            }
            counting_objective.resetCalls();

            const auto t0 = now();
            volatile double warmup_val = counting_objective.calculate(base_params);
            const auto t1 = now();

            const auto t2 = now();
            volatile double repeat_sum = 0.0;
            for (int i = 0; i < args.repeats; ++i) {
                repeat_sum += counting_objective.calculate(base_params);
            }
            const auto t3 = now();

            const auto t4 = now();
            volatile double jitter_sum = 0.0;
            VectorXd candidate = base_params;
            for (int k = 0; k < args.jitters; ++k) {
                for (int i = 0; i < candidate.size(); ++i) {
                    const double s = parameterManager.getSigmaForParamIndex(i);
                    candidate[i] = base_params[i] + s * normal(rng);
                }
                candidate = parameterManager.applyConstraints(candidate);
                jitter_sum += counting_objective.calculate(candidate);
            }
            const auto t5 = now();

            const double warmup_ms = ms(t1 - t0);
            const double repeats_ms = ms(t3 - t2);
            const double jitters_ms = ms(t5 - t4);

            std::cout << "\n--- Micro ---\n";
            std::cout << "Warmup: 1 eval => " << warmup_ms << " ms (value=" << warmup_val << ")\n";
            if (args.repeats > 0) {
                std::cout << "Repeat: " << args.repeats << " evals => " << repeats_ms << " ms (avg "
                          << (repeats_ms * 1000.0 / args.repeats) << " us/eval)\n";
            }
            if (args.jitters > 0) {
                std::cout << "Jitter: " << args.jitters << " evals => " << jitters_ms << " ms (avg "
                          << (jitters_ms * 1000.0 / args.jitters) << " us/eval)\n";
            }
            std::cout << "Objective calls: " << counting_objective.calls() << "\n";
            printCacheStats(sim_cache_ptr);

            (void)repeat_sum;
            (void)jitter_sum;
        };

        auto run_hill = [&]() {
            if (sim_cache_ptr != nullptr) {
                sim_cache_ptr->resetLikelihoodStats();
            }
            counting_objective.resetCalls();

            std::map<std::string, double> hill_settings = readHillClimbingSettings(args.hillSettingsPath);
            if (!args.useFileIters) {
                hill_settings["iterations"] = static_cast<double>(args.hillIters);
            }

            // Match calibration behavior: clamp for phase 1.
            parameterManager.setConstraintMode(ConstraintMode::OPTIMIZATION_CLAMP);

            epidemic::HillClimbingOptimizer hill;
            hill.configure(hill_settings);

            const auto t0 = now();
            epidemic::OptimizationResult res = hill.optimize(base_params, counting_objective, parameterManager);
            const auto t1 = now();

            std::cout << "\n--- Hill Climbing ---\n";
            std::cout << "Time: " << ms(t1 - t0) << " ms\n";
            std::cout << "Objective calls: " << counting_objective.calls() << "\n";
            std::cout << "Best logL: " << res.bestObjectiveValue << "\n";
            printCacheStats(sim_cache_ptr);
            return res.bestParameters;
        };

        auto run_mcmc = [&](const VectorXd& start_params) {
            if (sim_cache_ptr != nullptr) {
                sim_cache_ptr->resetLikelihoodStats();
            }
            counting_objective.resetCalls();

            std::map<std::string, double> mcmc_settings = readMetropolisHastingsSettings(args.mcmcSettingsPath);
            if (!args.useFileIters) {
                mcmc_settings["mcmc_iterations"] = static_cast<double>(args.mcmcIters);
            }

            // Benchmark friendliness: avoid disk I/O (trace/checkpoints) and sample storage overhead.
            // This keeps the timing focused on objective evaluations + proposal generation.
            mcmc_settings["store_samples"] = 0.0;
            mcmc_settings["write_checkpoints"] = 0.0;
            mcmc_settings["write_trace"] = 0.0;

            epidemic::MetropolisHastingsSampler mcmc;
            mcmc.configure(mcmc_settings);

            const auto t0 = now();
            epidemic::OptimizationResult res = mcmc.optimize(start_params, counting_objective, parameterManager);
            const auto t1 = now();

            std::cout << "\n--- MCMC (SA-MH) ---\n";
            std::cout << "Time: " << ms(t1 - t0) << " ms\n";
            std::cout << "Objective calls: " << counting_objective.calls() << "\n";
            std::cout << "Best logL: " << res.bestObjectiveValue << "\n";
            printCacheStats(sim_cache_ptr);
        };

        if (args.mode == "micro") {
            run_micro();
        } else if (args.mode == "hill") {
            (void)run_hill();
        } else if (args.mode == "mcmc") {
            run_mcmc(base_params);
        } else if (args.mode == "hillmcmc") {
            const VectorXd best = run_hill();
            run_mcmc(best);
        } else if (args.mode == "all") {
            run_micro();
            const VectorXd best = run_hill();
            run_mcmc(best);
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        printUsage(argv[0]);
        return 1;
    }
}
