#include "exceptions/Exceptions.hpp"
#include <stdexcept>
#include "sir_age_structured/ModelFactory.hpp"

namespace epidemic {

    std::shared_ptr<AgeSIRModel> ModelFactory::createAgeSIRModel(
        const Eigen::VectorXd& N,
        const Eigen::MatrixXd& C,
        const Eigen::VectorXd& gamma,
        double q,
        double scale_C)
    {
        try {
            return AgeSIRModel::create(N, C, gamma, q, scale_C);
        } catch (const ModelConstructionException& e) {
            throw;
        } catch (const std::exception& e) {
            throw ModelConstructionException("ModelFactory::createAgeSIRModel", "Unexpected error during AgeSIRModel creation: " + std::string(e.what()));
        }
         catch (...) {
             throw ModelConstructionException("ModelFactory::createAgeSIRModel", "Unknown error during AgeSIRModel creation.");
         }
    }
    
    std::shared_ptr<AgeSEPAIHRDModel> ModelFactory::createAgeSEPAIHRDModel(
        const SEPAIHRDParameters& params,
        std::shared_ptr<INpiStrategy> npi_strategy_ptr)
    {
        try {
            int num_classes = params.N.size();
            if (num_classes <= 0) throw ModelConstructionException("ModelFactory::createAgeSEPAIHRDModel", "Number of age classes must be positive.");
    
            if (!npi_strategy_ptr) {
                THROW_INVALID_PARAM("ModelFactory::createAgeSEPAIHRDModel", "NPI strategy pointer cannot be null.");
            }
    
            if (params.M_baseline.rows() != num_classes || params.M_baseline.cols() != num_classes ||
                params.p.size() != num_classes ||
                params.h.size() != num_classes ||
                params.icu.size() != num_classes ||
                params.d_H.size() != num_classes ||
                params.d_ICU.size() != num_classes) {
                 throw ModelConstructionException("ModelFactory::createAgeSEPAIHRDModel", "Parameter dimension mismatch.");
            }
             if ((params.N.array() < 0).any() || params.beta < 0 || params.theta < 0 ||
                 params.sigma < 0 || params.gamma_p < 0 || params.gamma_A < 0 ||
                 params.gamma_I < 0 || params.gamma_H < 0 || params.gamma_ICU < 0) {
                 throw ModelConstructionException("ModelFactory::createAgeSEPAIHRDModel", "Negative values found in parameters (N, rates, beta, theta, etc.).");
             }
             if ((params.p.array() < 0).any() || (params.h.array() < 0).any() || (params.icu.array() < 0).any() ||
                 (params.d_H.array() < 0).any() || (params.d_ICU.array() < 0).any()) {
                 throw ModelConstructionException("ModelFactory::createAgeSEPAIHRDModel", "Negative values found in vector parameters (p, h, icu, d_H, d_ICU).");
             }
             if ((params.p.array() < 0).any() || (params.p.array() > 1).any() ||
                 (params.h.array() < 0).any() || (params.h.array() > 1).any() ||
                 (params.icu.array() < 0).any() || (params.icu.array() > 1).any()) {
                  throw ModelConstructionException("ModelFactory::createAgeSEPAIHRDModel", "Fraction parameters (p, h, icu) must be between 0 and 1.");
             }
    
            return std::make_shared<AgeSEPAIHRDModel>(params, npi_strategy_ptr);
        } catch (const ModelConstructionException& e) {
            throw;
        } catch (const InvalidParameterException& e) {
             throw ModelConstructionException("ModelFactory::createAgeSEPAIHRDModel", "Validation failed: " + std::string(e.what()));
        } catch (const std::exception& e) {
             throw ModelConstructionException("ModelFactory::createAgeSEPAIHRDModel", "Unexpected error during AgeSEPAIHRDModel creation: " + std::string(e.what()));
        }
         catch (...) {
             throw ModelConstructionException("ModelFactory::createAgeSEPAIHRDModel", "Unknown error during AgeSEPAIHRDModel creation.");
         }
    }
    
    Eigen::VectorXd ModelFactory::createInitialSIRState(
        const Eigen::VectorXd& S0,
        const Eigen::VectorXd& I0,
        const Eigen::VectorXd& R0)
    {
        int n = S0.size();
        if (I0.size() != n || R0.size() != n) {
            THROW_INVALID_PARAM("ModelFactory::createInitialSIRState", "Initial state vectors (S0, I0, R0) must have the same dimensions. Got S0:" + std::to_string(n) + ", I0:" + std::to_string(I0.size()) + ", R0:" + std::to_string(R0.size()) + ".");
        }
        if (n == 0) {
             THROW_INVALID_PARAM("ModelFactory::createInitialSIRState", "Initial state vectors cannot be empty.");
        }
        if ((S0.array() < 0).any() || (I0.array() < 0).any() || (R0.array() < 0).any()) {
            THROW_INVALID_PARAM("ModelFactory::createInitialSIRState", "Initial state components (S0, I0, R0) cannot be negative.");
        }
    
        Eigen::VectorXd initial_state(3 * n);
        initial_state << S0, I0, R0;
        return initial_state;
    }
    
    Eigen::VectorXd ModelFactory::createInitialSEPAIHRDState(
        const Eigen::VectorXd& S0, const Eigen::VectorXd& E0, const Eigen::VectorXd& P0,
        const Eigen::VectorXd& A0, const Eigen::VectorXd& I0, const Eigen::VectorXd& H0,
        const Eigen::VectorXd& ICU0,
        const Eigen::VectorXd& R0, const Eigen::VectorXd& D0)
    {
        int n = S0.size();
        if (E0.size() != n || P0.size() != n || A0.size() != n || I0.size() != n || H0.size() != n || ICU0.size() != n || R0.size() != n || D0.size() != n) {
            THROW_INVALID_PARAM("ModelFactory::createInitialSEPAIHRDState", "All initial state vectors must have the same length as S0 (" + std::to_string(n) + ").");
        }
         if (n == 0) {
             THROW_INVALID_PARAM("ModelFactory::createInitialSEPAIHRDState", "Initial state vectors cannot be empty.");
        }
        if ((S0.array() < 0).any() || (E0.array() < 0).any() || (P0.array() < 0).any() || (A0.array() < 0).any() ||
            (I0.array() < 0).any() || (H0.array() < 0).any() || (ICU0.array() < 0).any() || (R0.array() < 0).any() || (D0.array() < 0).any()) {
            THROW_INVALID_PARAM("ModelFactory::createInitialSEPAIHRDState", "Initial state components cannot be negative.");
        }
    
        Eigen::VectorXd initial_state(9 * n);
        initial_state << S0, E0, P0, A0, I0, H0, ICU0, R0, D0;
        return initial_state;
    }
    
} // namespace epidemic