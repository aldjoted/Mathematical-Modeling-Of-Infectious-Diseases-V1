#ifndef GET_CALIBRATION_DATA_HPP
#define GET_CALIBRATION_DATA_HPP

#include <vector>
#include <Eigen/Dense>
#include <string>

namespace epidemic {

/**
 * @class CalibrationData
 * @brief Loads and provides access to epidemiological data for model calibration.
 *
 * This class reads and processes calibration data from CSV files or accepts pre-loaded
 * Eigen matrices. The data typically includes:
 * - Population by age group
 * - New confirmed cases by date and age group
 * - New deaths by date and age group
 * - New hospitalizations by date and age group
 * - New ICU admissions by date and age group
 * - Cumulative figures for the above where relevant.
 *
 * Data can be filtered by a specified date range when loading from a file.
 * The class also provides a method to construct an initial state vector for
 * an SEPAIHRD model based on the loaded data and provided model parameters.
 *
 * @throws std::runtime_error If file operations fail, required data is missing,
 *                            or data formats are invalid.
 */
class CalibrationData {
    public:
        /**
         * @brief Constructs a CalibrationData object by loading data from a CSV file.
         *
         * @param filename Path to the CSV file containing epidemiological data.
         * @param start_date Optional start date for filtering data (format: "YYYY-MM-DD").
         *                   If empty, no start date filter is applied.
         * @param end_date Optional end date for filtering data (format: "YYYY-MM-DD").
         *                 If empty, no end date filter is applied.
         * @throws std::runtime_error If the file cannot be read or has an invalid format.
         */
        explicit CalibrationData(const std::string& filename, 
                               const std::string& start_date = "", 
                               const std::string& end_date = "");
        /**
         * @brief Constructs a CalibrationData object with provided data matrices.
         *        Useful for testing or when data is sourced externally.
         *
         * @param new_confirmed_cases_in Matrix of new confirmed cases (rows: time, cols: age_group).
         * @param new_hospitalizations_in Matrix of new hospitalizations.
         * @param new_icu_in Matrix of new ICU admissions.
         * @param new_deaths_in Matrix of new deaths.
         * @param population_by_age_in Vector of population sizes for each age group.
         * @param initial_cumulative_confirmed_row0_in Vector of cumulative confirmed cases at t0.
         * @param initial_cumulative_deaths_row0_in Vector of cumulative deaths at t0.
         * @param initial_cumulative_hospitalizations_row0_in Vector of cumulative hospitalizations at t0.
         * @param initial_cumulative_icu_row0_in Vector of cumulative ICU admissions at t0.
         * @param num_age_classes_in The number of age classes.
         */
        explicit CalibrationData(
            const Eigen::MatrixXd& new_confirmed_cases_in,
            const Eigen::MatrixXd& new_hospitalizations_in,
            const Eigen::MatrixXd& new_icu_in,
            const Eigen::MatrixXd& new_deaths_in,
            const Eigen::VectorXd& population_by_age_in,
            const Eigen::VectorXd& initial_cumulative_confirmed_row0_in,
            const Eigen::VectorXd& initial_cumulative_deaths_row0_in,
            const Eigen::VectorXd& initial_cumulative_hospitalizations_row0_in,
            const Eigen::VectorXd& initial_cumulative_icu_row0_in,
            int num_age_classes_in
        );
    
        /** @return const Eigen::MatrixXd& New confirmed cases (rows: time, cols: age_group). */
        const Eigen::MatrixXd& getNewConfirmedCases() const { return new_confirmed_cases; }
        /** @return const Eigen::MatrixXd& New deaths. */
        const Eigen::MatrixXd& getNewDeaths() const { return new_deaths; }
        /** @return const Eigen::MatrixXd& New hospitalizations. */
        const Eigen::MatrixXd& getNewHospitalizations() const { return new_hospitalizations; }
        /** @return const Eigen::MatrixXd& New ICU admissions. */
        const Eigen::MatrixXd& getNewICU() const { return new_icu; }
        /** @return const Eigen::MatrixXd& Cumulative confirmed cases. */
        const Eigen::MatrixXd& getCumulativeConfirmedCases() const { return cumulative_confirmed_cases; }
        /** @return const Eigen::MatrixXd& Cumulative deaths. */
        const Eigen::MatrixXd& getCumulativeDeaths() const { return cumulative_deaths; }
        /** @return const Eigen::MatrixXd& Cumulative hospitalizations. */
        const Eigen::MatrixXd& getCumulativeHospitalizations() const { return cumulative_hospitalizations; }
        /** @return const Eigen::MatrixXd& Cumulative ICU admissions. */
        const Eigen::MatrixXd& getCumulativeICU() const { return cumulative_icu; }
        /** @return const Eigen::VectorXd& Population sizes for each age group. */
        const Eigen::VectorXd& getPopulationByAgeGroup() const { return population_by_age; }
        
        /**
         * @brief Returns the initial cumulative confirmed cases (proxy for active cases at t0).
         * @return Eigen::VectorXd Cumulative confirmed cases from the first data point.
         * @throws std::runtime_error If data is empty.
         */
        Eigen::VectorXd getInitialActiveCases() const;

        /**
         * @brief Constructs the initial state vector for an SEPAIHRD model.
         *
         * This method uses data from the first time point (t0) and provided model parameters
         * to estimate the initial number of individuals in each compartment (S, E, P, A, I, H, ICU, R, D)
         * for each age group.
         * - Observable compartments (I0, H0, ICU0, D0) are seeded from the data.
         * - Unobservable compartments (E0, P0, A0) are estimated based on I0 and model transition rates.
         * - S0 is calculated as the remainder of the population.
         * - R0 is typically assumed to be zero initially if not directly observed.
         * All values are constrained to be non-negative and consistent with population sizes.
         *
         * @param sigma_rate Rate of progression from Exposed (E) to Presymptomatic (P).
         * @param gamma_p_rate Rate of progression from Presymptomatic (P).
         * @param gamma_a_rate Rate of recovery for Asymptomatic (A) individuals.
         * @param gamma_i_rate Rate of recovery for Symptomatic (I) individuals (non-hospitalized).
         * @param p_asymptomatic_fractions Vector of fractions of infections becoming asymptomatic per age group.
         * @param h_hospitalization_rates Vector of rates at which symptomatic individuals are hospitalized per age group.
         * @return Eigen::VectorXd The initial state vector for the SEPAIHRD model.
         * @throws std::runtime_error If essential data is missing or parameters have invalid dimensions.
         */
            Eigen::VectorXd getInitialSEPAIHRDState(
                double sigma_rate,
                double gamma_p_rate,
                double gamma_a_rate,
                double gamma_i_rate,
                const Eigen::VectorXd& p_asymptomatic_fractions,
                const Eigen::VectorXd& h_hospitalization_rates
            ) const;

        /** @return int The number of time points (dates) in the loaded data. */
        int getNumDataPoints() const { return n_data_points; }
        /** @return int The number of age classes defined for the data. */
        int getNumAgeClasses() const { return num_age_classes; }
        /** @return const std::vector<std::string>& List of date strings (YYYY-MM-DD). */
        const std::vector<std::string>& getDates() const { return dates; }
    
    private:
        /**
         * @brief Reads and parses epidemiological data from a CSV file.
         *
         * This is the core data loading and processing logic. It handles file reading,
         * date filtering, and populating the internal Eigen matrices.
         *
         * @param filename Path to the CSV data file.
         * @param start_date Start date for filtering (inclusive).
         * @param end_date End date for filtering (inclusive).
         * @return True if data was successfully loaded and processed, false otherwise.
         * @throws std::runtime_error If file operations fail, or if data is malformed or missing.
         */
        bool readCSVData(const std::string& filename,
                        const std::string& start_date,
                        const std::string& end_date);
        
        /**
         * @brief Checks if a given date string falls within the specified date range.
         *
         * @param date The date string to check (format: "YYYY-MM-DD").
         * @param start_date The start of the range (inclusive, "YYYY-MM-DD"). Empty means no lower bound.
         * @param end_date The end of the range (inclusive, "YYYY-MM-DD"). Empty means no upper bound.
         * @return True if the date is within the range (or if the range is unbounded), false otherwise.
         */
        bool isDateInRange(const std::string& date,
                        const std::string& start_date,
                        const std::string& end_date) const;

        /** @brief New confirmed cases by date (rows) and age group (columns). */
        Eigen::MatrixXd new_confirmed_cases;
        /** @brief New deaths by date and age group. */
        Eigen::MatrixXd new_deaths;
        /** @brief New hospitalizations by date and age group. */
        Eigen::MatrixXd new_hospitalizations;
        /** @brief New ICU admissions by date and age group. */
        Eigen::MatrixXd new_icu;

        /** @brief Population sizes for each age group. */
        Eigen::VectorXd population_by_age;

        /** @brief Cumulative confirmed cases by date and age group. */
        Eigen::MatrixXd cumulative_confirmed_cases;
        /** @brief Cumulative deaths by date and age group. */
        Eigen::MatrixXd cumulative_deaths;
        /** @brief Cumulative hospitalizations by date and age group. */
        Eigen::MatrixXd cumulative_hospitalizations;
        /** @brief Cumulative ICU admissions by date and age group. */
        Eigen::MatrixXd cumulative_icu;

        /** @brief List of dates corresponding to each row in the data matrices. */
        std::vector<std::string> dates;

        /** @brief Number of time points (dates) in the loaded and filtered data. */
        int n_data_points;
        /** @brief Number of age groups/classes considered in the data. */
        int num_age_classes;
};

} // namespace epidemic

#endif // GET_CALIBRATION_DATA_HPP