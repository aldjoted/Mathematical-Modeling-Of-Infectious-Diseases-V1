#include "model/AgeSEPAIHRDsimulator.hpp"
#include "exceptions/Exceptions.hpp"
#include <memory>

namespace epidemic {

    AgeSEPAIHRDSimulator::AgeSEPAIHRDSimulator(
        std::shared_ptr<AgeSEPAIHRDModel> model,
        std::shared_ptr<IOdeSolverStrategy> solver_strategy,
        double start_time,
        double end_time,
        double time_step,
        double abs_error,
        double rel_error
    ) : Simulator(std::static_pointer_cast<EpidemicModel>(model), 
                  solver_strategy,                            
                  start_time, end_time, time_step, abs_error, rel_error) {}
    
    std::shared_ptr<AgeSEPAIHRDModel> AgeSEPAIHRDSimulator::getTypedModel() const {
        // Use dynamic_pointer_cast to safely cast the base model pointer back to the derived type
        return std::dynamic_pointer_cast<AgeSEPAIHRDModel>(model);
    }
    
} // namespace epidemic