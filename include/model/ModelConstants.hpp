#ifndef MODEL_CONSTANTS_HPP
#define MODEL_CONSTANTS_HPP

namespace epidemic {
namespace constants {

    constexpr int DEFAULT_NUM_AGE_CLASSES = 4;
    constexpr double NUMERICAL_EPSILON = 1e-9;
    constexpr double MIN_POPULATION_FOR_DIVISION = 1e-9;

    constexpr double DEFAULT_NPI_LOWER_BOUND = 0.1;
    constexpr double DEFAULT_NPI_UPPER_BOUND = 1.5;
    constexpr double DEFAULT_BASELINE_PERIOD_END_TIME = 13.0;
    constexpr double DEFAULT_BASELINE_KAPPA = 1.0;

    constexpr int NUM_COMPARTMENTS_SEPAIHRD = 9; 

} // namespace constants
} // namespace epidemic

#endif // MODEL_CONSTANTS_HPP