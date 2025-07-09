cmake_minimum_required(VERSION 3.10)
project(SIR_Models LANGUAGES CXX)

# Compiler settings
set(WARNING_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    # -Wconversion
    # -Wshadow
)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Dependencies
find_package(GSL REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Boost REQUIRED COMPONENTS system)
find_package(GTest REQUIRED)
find_package(OpenMP REQUIRED)

# --- Base Model Library ---
set(BASE_MODEL_SRCS
    src/base/SIRModel.cpp
    src/base/SIR_population_variable.cpp
    src/base/SIR_stochastic.cpp
    src/base/main/ModelParameters.cpp
)

add_library(sir_models_lib STATIC ${BASE_MODEL_SRCS})
target_compile_features(sir_models_lib PUBLIC cxx_std_17)
target_compile_options(sir_models_lib PRIVATE ${WARNING_FLAGS})
target_include_directories(sir_models_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        ${Eigen3_INCLUDE_DIRS}
        ${GSL_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
)
target_link_libraries(sir_models_lib
    PUBLIC
        GSL::gsl
        GSL::gslcblas
        Eigen3::Eigen
        Boost::system
        OpenMP::OpenMP_CXX
)

# --- Age-Structured & SEPAIHRD Model Library ---
set(EPIDEMIC_MODELS_SRCS
    # SIR Age Structured
    src/sir_age_structured/AgeSIRModel.cpp
    src/sir_age_structured/InterventionCallback.cpp
    src/sir_age_structured/ModelFactory.cpp
    src/sir_age_structured/Simulator.cpp
    src/sir_age_structured/solvers/Dopri5SolverStrategy.cpp
    src/sir_age_structured/solvers/CashKarpSolverStrategy.cpp
    src/sir_age_structured/solvers/FehlbergSolverStrategy.cpp
    src/sir_age_structured/ModelCalibrator.cpp
    src/sir_age_structured/SimulationResultProcessor.cpp
    src/sir_age_structured/parameters/SIRParameterManager.cpp
    src/sir_age_structured/objectives/PoissonLikelihoodObjective.cpp
    src/sir_age_structured/optimizers/HillClimbingOptimizer.cpp
    src/sir_age_structured/optimizers/MetropolisHastingsSampler.cpp
    src/sir_age_structured/caching/SimulationCache.cpp

    # SEPAIHRD Age Structured
    src/model/AgeSEPAIHRDModel.cpp
    src/model/AgeSEPAIHRDsimulator.cpp
    src/model/PieceWiseConstantNPIStrategy.cpp
    src/model/parameters/SEPAIHRDParameterManager.cpp
    src/model/objectives/SEPAIHRDObjectiveFunction.cpp
    src/model/SEPAIHRDModelCalibration.cpp
    src/model/optimizers/ParticleSwarmOptimizer.cpp
    src/model/optimizers/NUTSSampler.cpp
    src/model/ReproductionNumberCalculator.cpp
    src/model/PostCalibrationAnalyser.cpp
)
set(EPIDEMIC_UTILS_SRCS
    src/utils/FileUtils.cpp
    src/utils/ReadContactMatrix.cpp
    src/utils/GetCalibrationData.cpp
    src/utils/ReadCalibrationConfiguration.cpp
    src/exceptions/CSVReadException.cpp

)
add_library(epidemic_models_lib STATIC ${EPIDEMIC_MODELS_SRCS} ${EPIDEMIC_UTILS_SRCS})
target_compile_features(epidemic_models_lib PUBLIC cxx_std_17)
target_compile_options(epidemic_models_lib PRIVATE ${WARNING_FLAGS})
target_include_directories(epidemic_models_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        ${Eigen3_INCLUDE_DIRS}
        ${GSL_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
)
target_link_libraries(epidemic_models_lib
    PUBLIC
        GSL::gsl
        GSL::gslcblas
        Eigen3::Eigen
        Boost::system
        OpenMP::OpenMP_CXX
)

# --- Application Entry Points ---
set(BASE_MAIN_NAMES
    sir_model
    sir_pop_var
    sir_stochastic
)
set(BASE_MAIN_SRCS
    src/base/main/sir_main.cpp
    src/base/main/sir_pop_va_main.cpp
    src/base/main/sir_sto_main.cpp
)

set(AGE_SIR_MAIN_NAME sir_age_structured_main)
set(AGE_SIR_MAIN_SRC src/sir_age_structured/main.cpp)

set(AGE_SIR_CALIB_DEMO_NAME sir_age_structured_calibration_demo)
set(AGE_SIR_CALIB_DEMO_SRC src/sir_age_structured/CalibrationDemo.cpp)

set(SEPAIHRD_MAIN_NAME sepaihrd_age_structured_main)
set(SEPAIHRD_MAIN_SRC src/model/main.cpp)

set(ALL_MAIN_NAMES ${BASE_MAIN_NAMES} ${AGE_SIR_MAIN_NAME} ${AGE_SIR_CALIB_DEMO_NAME} ${SEPAIHRD_MAIN_NAME})

# Build base executables
list(LENGTH BASE_MAIN_NAMES _num_base_apps)
if(_num_base_apps GREATER 0)
    math(EXPR _last_base_index "${_num_base_apps} - 1")
    foreach(_i RANGE 0 ${_last_base_index})
        list(GET BASE_MAIN_NAMES ${_i} _name)
        list(GET BASE_MAIN_SRCS  ${_i} _src)
        add_executable(${_name} ${_src})
        target_compile_features(${_name} PUBLIC cxx_std_17)
        target_compile_options(${_name} PRIVATE ${WARNING_FLAGS})
        target_link_libraries(${_name} PRIVATE sir_models_lib epidemic_models_lib) # Add epidemic_models_lib
    endforeach()
endif()

# Build age-structured SIR executables
add_executable(${AGE_SIR_MAIN_NAME} ${AGE_SIR_MAIN_SRC})
target_compile_features(${AGE_SIR_MAIN_NAME} PUBLIC cxx_std_17)
target_compile_options(${AGE_SIR_MAIN_NAME} PRIVATE ${WARNING_FLAGS})
target_link_libraries(${AGE_SIR_MAIN_NAME} PRIVATE epidemic_models_lib)

add_executable(${AGE_SIR_CALIB_DEMO_NAME} ${AGE_SIR_CALIB_DEMO_SRC})
target_compile_features(${AGE_SIR_CALIB_DEMO_NAME} PUBLIC cxx_std_17)
target_compile_options(${AGE_SIR_CALIB_DEMO_NAME} PRIVATE ${WARNING_FLAGS})
target_link_libraries(${AGE_SIR_CALIB_DEMO_NAME} PRIVATE epidemic_models_lib) 

# Build age-structured SEPAIHRD executable
add_executable(${SEPAIHRD_MAIN_NAME} ${SEPAIHRD_MAIN_SRC})
target_compile_features(${SEPAIHRD_MAIN_NAME} PUBLIC cxx_std_17)
target_compile_options(${SEPAIHRD_MAIN_NAME} PRIVATE ${WARNING_FLAGS})
target_link_libraries(${SEPAIHRD_MAIN_NAME} PRIVATE epidemic_models_lib) 


# --- Test Setup ---
enable_testing()

# --- Utils Tests ---
file(GLOB UTILS_TEST_SRCS
    "${CMAKE_SOURCE_DIR}/tests/utils/*.cpp"
)

add_executable(utils_tests ${UTILS_TEST_SRCS})
target_compile_features(utils_tests PUBLIC cxx_std_17)
target_compile_options(utils_tests PRIVATE ${WARNING_FLAGS})
target_include_directories(utils_tests PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(utils_tests
    PRIVATE
        sir_models_lib 
        epidemic_models_lib 
        GTest::gtest
        GTest::gtest_main
)

add_test(NAME UtilsTests COMMAND utils_tests)

# --- Model Tests ---
file(GLOB MODEL_TEST_SRCS
    "${CMAKE_SOURCE_DIR}/tests/model/*.cpp"
)

add_executable(model_tests ${MODEL_TEST_SRCS})
target_compile_features(model_tests PUBLIC cxx_std_17)
target_compile_options(model_tests PRIVATE ${WARNING_FLAGS})
target_include_directories(model_tests PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(model_tests
    PRIVATE
        epidemic_models_lib
        GTest::gtest
        GTest::gmock
        GTest::gtest_main
)
add_test(NAME ModelTests COMMAND model_tests)

# --- SIR Age-Structured Tests ---
file(GLOB SIR_AGE_STRUCTURED_TEST_SRCS
    "${CMAKE_SOURCE_DIR}/tests/sir_age_structured/*.cpp"
)

add_executable(sir_age_structured_tests ${SIR_AGE_STRUCTURED_TEST_SRCS})
target_compile_features(sir_age_structured_tests PUBLIC cxx_std_17)
target_compile_options(sir_age_structured_tests PRIVATE ${WARNING_FLAGS})
target_include_directories(sir_age_structured_tests PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sir_age_structured_tests
    PRIVATE
        epidemic_models_lib
        GTest::gtest
        GTest::gmock
        GTest::gtest_main
)
add_test(NAME SirAgeStructuredTests COMMAND sir_age_structured_tests)

# --- Output directory ---
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE PATH "Executable output directory")

# Valgrind memory checks
find_program(VALGRIND_EXECUTABLE valgrind)
if (VALGRIND_EXECUTABLE)
    set(VALGRIND_OPTS --leak-check=full --show-leak-kinds=all --track-origins=yes)
    foreach(_name IN LISTS ALL_MAIN_NAMES)
        add_custom_target(memcheck_${_name}
            COMMAND ${VALGRIND_EXECUTABLE} ${VALGRIND_OPTS} $<TARGET_FILE:${_name}>
            DEPENDS ${_name}
            COMMENT "Valgrind memcheck on ${_name}"
        )
    endforeach()
endif()

# Installation
install(TARGETS ${ALL_MAIN_NAMES}
        RUNTIME DESTINATION bin)