#include "utils/GetCalibrationData.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <charconv>
#include <vector>
#include <string_view>
#include <algorithm>
#include <stdexcept>
#include <iomanip>

namespace epidemic {

CalibrationData::CalibrationData(const std::string& filename,
                                 const std::string& start_date,
                                 const std::string& end_date)
    : n_data_points(0), num_age_classes(4) {
    if (!readCSVData(filename, start_date, end_date)) {
        throw std::runtime_error("Failed to initialize CalibrationData from file: " + filename);
    }
}

CalibrationData::CalibrationData(
    const Eigen::MatrixXd& new_confirmed_cases_in,
    const Eigen::MatrixXd& new_hospitalizations_in,
    const Eigen::MatrixXd& new_icu_in,
    const Eigen::MatrixXd& new_deaths_in,
    const Eigen::VectorXd& population_by_age_in,
    const Eigen::VectorXd& initial_cumulative_confirmed_row0_in,
    const Eigen::VectorXd& initial_cumulative_deaths_row0_in,
    const Eigen::VectorXd& initial_cumulative_hospitalizations_row0_in,
    const Eigen::VectorXd& initial_cumulative_icu_row0_in,
    int num_age_classes_in)
    : new_confirmed_cases(new_confirmed_cases_in),
      new_deaths(new_deaths_in),
      new_hospitalizations(new_hospitalizations_in),
      new_icu(new_icu_in),
      population_by_age(population_by_age_in),
      n_data_points(new_confirmed_cases_in.rows()),
      num_age_classes(num_age_classes_in)
{
    if (num_age_classes <= 0) {
        throw std::invalid_argument("Number of age classes must be positive.");
    }
    if (population_by_age_in.size() != num_age_classes) {
        throw std::invalid_argument("Population vector size mismatch with num_age_classes.");
    }
    if (new_confirmed_cases_in.cols() != num_age_classes ||
        new_hospitalizations_in.cols() != num_age_classes ||
        new_icu_in.cols() != num_age_classes ||
        new_deaths_in.cols() != num_age_classes) {
        throw std::invalid_argument("Incidence data matrix column count mismatch with num_age_classes.");
    }
    if (initial_cumulative_confirmed_row0_in.size() != num_age_classes ||
        initial_cumulative_deaths_row0_in.size() != num_age_classes ||
        initial_cumulative_hospitalizations_row0_in.size() != num_age_classes ||
        initial_cumulative_icu_row0_in.size() != num_age_classes) {
        throw std::invalid_argument("Initial cumulative data vector size mismatch with num_age_classes.");
    }
    /**
     * @brief Initializes a cumulative data matrix based on new incidence and an initial cumulative row.
     * 
     * This helper lambda function is used within the constructor to populate
     * the cumulative data matrices (e.g., cumulative_confirmed_cases, cumulative_deaths).
     * It takes an empty cumulative matrix, a matrix of corresponding new daily incidence,
     * and a vector representing the cumulative count at the very first time point (t0).
     * 
     * For each subsequent time point t > t0, the cumulative count is calculated as:
     * Cumulative(t) = Cumulative(t-1) + NewIncidence(t-1)
     * 
     * If n_data_points is 0, the cumulative_matrix is resized to be empty (0 rows).
     * 
     * @param cumulative_matrix (Eigen::MatrixXd&) A reference to the cumulative matrix to be populated.
     *                                           It will be resized according to n_data_points and num_age_classes.
     * @param new_cases_matrix (const Eigen::MatrixXd&) A constant reference to the matrix of new daily
     *                                                 incidence data. Rows correspond to time, columns to age groups.
     * @param initial_cumulative_row0 (const Eigen::VectorXd&) A constant reference to the vector containing
     *                                                        the cumulative counts for each age group at the
     *                                                        first time point (t0). This forms the first row
     *                                                        of the `cumulative_matrix`.
     */
    auto initialize_cumulative = [&](Eigen::MatrixXd& cumulative_matrix,
                                     const Eigen::MatrixXd& new_cases_matrix,
                                     const Eigen::VectorXd& initial_cumulative_row0) {
        if (this->n_data_points > 0) {
            cumulative_matrix.resize(this->n_data_points, this->num_age_classes);
            cumulative_matrix.row(0) = initial_cumulative_row0;
            for (int i = 1; i < this->n_data_points; ++i) {
                if (i -1 < new_cases_matrix.rows()){
                     cumulative_matrix.row(i) = cumulative_matrix.row(i - 1) + new_cases_matrix.row(i -1);
                } else {
                    cumulative_matrix.row(i) = cumulative_matrix.row(i-1);
                }
            }
        } else {
            cumulative_matrix.resize(0, this->num_age_classes);
        }
    };

    initialize_cumulative(this->cumulative_confirmed_cases, this->new_confirmed_cases, initial_cumulative_confirmed_row0_in);
    initialize_cumulative(this->cumulative_deaths, this->new_deaths, initial_cumulative_deaths_row0_in);
    initialize_cumulative(this->cumulative_hospitalizations, this->new_hospitalizations, initial_cumulative_hospitalizations_row0_in);
    initialize_cumulative(this->cumulative_icu, this->new_icu, initial_cumulative_icu_row0_in);

    this->dates.resize(this->n_data_points);
    for (int i = 0; i < this->n_data_points; ++i) {
        this->dates[i] = "mock_date_" + std::to_string(i);
    }
}

bool CalibrationData::isDateInRange(const std::string& date,
                                    const std::string& start_date,
                                    const std::string& end_date) const {
    if (start_date.empty() && end_date.empty()) return true;
    if (!start_date.empty() && date < start_date) return false;
    if (!end_date.empty() && date > end_date) return false;
    return true;
}

Eigen::VectorXd CalibrationData::getInitialActiveCases() const {
    if (cumulative_confirmed_cases.rows() == 0) {
        throw std::runtime_error("Cannot get initial active cases: cumulative_confirmed_cases data is empty.");
    }
    return cumulative_confirmed_cases.row(0);
}

Eigen::VectorXd CalibrationData::getInitialSEPAIHRDState(
    double sigma_rate,
    double gamma_p_rate,
    double gamma_a_rate,
    double gamma_i_rate,
    const Eigen::VectorXd& p_asymptomatic_fractions,
    const Eigen::VectorXd& h_hospitalization_rates
) const {
    if (n_data_points == 0) {
        throw std::runtime_error("Cannot get initial SEPAIHRD state: No data points loaded.");
    }
    if (population_by_age.size() != num_age_classes) {
        throw std::runtime_error("Cannot get initial SEPAIHRD state: Population data size mismatch with num_age_classes.");
    }
    if (p_asymptomatic_fractions.size() != num_age_classes || h_hospitalization_rates.size() != num_age_classes) {
        throw std::runtime_error("Cannot get initial SEPAIHRD state: p_asymptomatic_fractions or h_hospitalization_rates size mismatch with num_age_classes.");
    }
    if (cumulative_deaths.rows() == 0 || cumulative_icu.rows() == 0 || cumulative_hospitalizations.rows() == 0 || new_confirmed_cases.rows() == 0) {
        throw std::runtime_error("Cannot get initial SEPAIHRD state: Required data matrices (D0, ICU0, H0, I0 proxies) are empty.");
    }

    int n_comps = 9; // S, E, P, A, I, H, ICU, R, D
    Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(n_comps * num_age_classes);

    const Eigen::VectorXd& N = getPopulationByAgeGroup();

    Eigen::VectorXd D0_i = cumulative_deaths.row(0).cwiseMax(0.0);
    Eigen::VectorXd ICU0_i = cumulative_icu.row(0).cwiseMax(0.0);
    Eigen::VectorXd H0_i = cumulative_hospitalizations.row(0).cwiseMax(0.0);
    Eigen::VectorXd I0_i = (cumulative_confirmed_cases.row(0).transpose() + H0_i + ICU0_i).cwiseMax(0.0);

    Eigen::VectorXd R0_i = Eigen::VectorXd::Zero(num_age_classes);
    Eigen::VectorXd E0_i(num_age_classes);
    Eigen::VectorXd P0_i(num_age_classes);
    Eigen::VectorXd A0_i(num_age_classes);
    Eigen::VectorXd S0_i(num_age_classes);

    for (int i = 0; i < num_age_classes; ++i) {
        double I0_val = I0_i(i);
        double p_i_val = p_asymptomatic_fractions(i);
        double h_i_val = h_hospitalization_rates(i);

        p_i_val = std::max(0.0, std::min(1.0, p_i_val));
        double one_minus_p_i_val = std::max(1e-9, 1.0 - p_i_val);

        if (sigma_rate > 1e-9 && gamma_p_rate > 1e-9 && gamma_a_rate > 1e-9 && gamma_i_rate > 1e-9) {
            P0_i(i) = I0_val * (gamma_i_rate + h_i_val) / (one_minus_p_i_val * gamma_p_rate);
            E0_i(i) = P0_i(i) * gamma_p_rate / sigma_rate;
            A0_i(i) = P0_i(i) * p_i_val * gamma_p_rate / gamma_a_rate;
        } else {
            std::cerr << "Warning: Initializing E0, P0, A0 with fallback due to non-positive rates for age group " << i << std::endl;
            P0_i(i) = I0_val;
            E0_i(i) = I0_val * 1.5;
            A0_i(i) = I0_val * (p_i_val / one_minus_p_i_val);
        }
        E0_i(i) = std::max(0.0, E0_i(i));
        P0_i(i) = std::max(0.0, P0_i(i));
        A0_i(i) = std::max(0.0, A0_i(i));

        D0_i(i) = std::min(D0_i(i), N(i));
        ICU0_i(i) = std::min(ICU0_i(i), std::max(0.0, N(i) - D0_i(i)));
        H0_i(i) = std::min(H0_i(i), std::max(0.0, N(i) - D0_i(i) - ICU0_i(i)));
        I0_i(i) = std::min(I0_i(i), std::max(0.0, N(i) - D0_i(i) - ICU0_i(i) - H0_i(i)));

        double sum_non_S = E0_i(i) + P0_i(i) + A0_i(i) + I0_i(i) + H0_i(i) + ICU0_i(i) + R0_i(i) + D0_i(i);

        if (sum_non_S > N(i)) {
            double sum_data_derived_and_R = I0_i(i) + H0_i(i) + ICU0_i(i) + R0_i(i) + D0_i(i);
            double sum_calculated_hidden = E0_i(i) + P0_i(i) + A0_i(i);

            if (sum_data_derived_and_R >= N(i)) {
                E0_i(i) = 0.0; P0_i(i) = 0.0; A0_i(i) = 0.0;
                S0_i(i) = 0.0;
            } else {
                double available_for_hidden = N(i) - sum_data_derived_and_R;
                if (sum_calculated_hidden > available_for_hidden) {
                    if (sum_calculated_hidden > 1e-9) {
                        double scale_factor = available_for_hidden / sum_calculated_hidden;
                        E0_i(i) *= scale_factor;
                        P0_i(i) *= scale_factor;
                        A0_i(i) *= scale_factor;
                    } else {
                        E0_i(i) = 0.0; P0_i(i) = 0.0; A0_i(i) = 0.0;
                    }
                    S0_i(i) = 0.0;
                } else { 
                    S0_i(i) = available_for_hidden - sum_calculated_hidden;
                }
            }
        } else {
            S0_i(i) = N(i) - sum_non_S;
        }
        S0_i(i) = std::max(0.0, S0_i(i));
    }

    initial_state.segment(0 * num_age_classes, num_age_classes) = S0_i;
    initial_state.segment(1 * num_age_classes, num_age_classes) = E0_i;
    initial_state.segment(2 * num_age_classes, num_age_classes) = P0_i;
    initial_state.segment(3 * num_age_classes, num_age_classes) = A0_i;
    initial_state.segment(4 * num_age_classes, num_age_classes) = I0_i;
    initial_state.segment(5 * num_age_classes, num_age_classes) = H0_i;
    initial_state.segment(6 * num_age_classes, num_age_classes) = ICU0_i;
    initial_state.segment(7 * num_age_classes, num_age_classes) = R0_i;
    initial_state.segment(8 * num_age_classes, num_age_classes) = D0_i;

    for (int i = 0; i < num_age_classes; ++i) {
        double total_for_age_group = 0;
        for (int j = 0; j < n_comps; ++j) {
            total_for_age_group += initial_state(j * num_age_classes + i);
        }
        if (std::abs(total_for_age_group - N(i)) > std::max(1e-9, 1e-6 * N(i))) {
             std::cerr << "Warning: Initial state sum for age group " << i << " ("
                       << std::fixed << std::setprecision(2) << total_for_age_group
                       << ") does not precisely match population N(" << i << ") = "
                       << std::fixed << std::setprecision(2) << N(i)
                       << ". Discrepancy: " << std::scientific << (total_for_age_group - N(i))
                       << std::endl;
        }
    }
    return initial_state;
}

bool CalibrationData::readCSVData(const std::string& filename,
                                  const std::string& start_date,
                                  const std::string& end_date) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return false;
    }

    std::string header;
    if (!std::getline(file, header)) {
        std::cerr << "Error: Empty file or unable to read header." << std::endl;
        return false;
    }

    std::map<std::string, int> column_indices;
    int idx = 0;
    std::istringstream header_stream(header);
    std::string column;
    while (std::getline(header_stream, column, ',')) {
        column_indices[column] = idx++;
    }
    /**
     * @brief Lambda function to find column index by name
     * @param name Name of the column to find
     * @return int Index of the named column
     * @throws std::runtime_error If the column is not found
     */
    auto get_index = [&](const std::string& name) {
        auto it = column_indices.find(name);
        if (it == column_indices.end()) {
            throw std::runtime_error("Missing required column: " + name);
        }
        return it->second;
    };

    const int date_idx = get_index("date");
    const int confirmed_idx[] = {
        get_index("new_confirmed_0_30"), get_index("new_confirmed_30_60"),
        get_index("new_confirmed_60_80"), get_index("new_confirmed_80_plus")
    };
    const int deceased_idx[] = {
        get_index("new_deceased_0_30"), get_index("new_deceased_30_60"),
        get_index("new_deceased_60_80"), get_index("new_deceased_80_plus")
    };
    const int hospitalized_idx[] = {
        get_index("new_hospitalized_patients_0_30"), get_index("new_hospitalized_patients_30_60"),
        get_index("new_hospitalized_patients_60_80"), get_index("new_hospitalized_patients_80_plus")
    };
    const int icu_idx[] = {
        get_index("new_intensive_care_patients_0_30"), get_index("new_intensive_care_patients_30_60"),
        get_index("new_intensive_care_patients_60_80"), get_index("new_intensive_care_patients_80_plus")
    };
    const int population_idx[] = {
        get_index("population_0_30"), get_index("population_30_60"),
        get_index("population_60_80"), get_index("population_80_plus")
    };
    const int cumulative_confirmed_idx[] = {
        get_index("cumulative_confirmed_0_30"), get_index("cumulative_confirmed_30_60"),
        get_index("cumulative_confirmed_60_80"), get_index("cumulative_confirmed_80_plus")
    };
    const int cumulative_deceased_idx[] = {
        get_index("cumulative_deceased_0_30"), get_index("cumulative_deceased_30_60"),
        get_index("cumulative_deceased_60_80"), get_index("cumulative_deceased_80_plus")
    };
    const int cumulative_hospitalized_idx[] = {
        get_index("cumulative_hospitalized_patients_0_30"), get_index("cumulative_hospitalized_patients_30_60"),
        get_index("cumulative_hospitalized_patients_60_80"), get_index("cumulative_hospitalized_patients_80_plus")
    };
    const int cumulative_icu_idx[] = {
        get_index("cumulative_intensive_care_patients_0_30"), get_index("cumulative_intensive_care_patients_30_60"),
        get_index("cumulative_intensive_care_patients_60_80"), get_index("cumulative_intensive_care_patients_80_plus")
    };

    std::streampos data_start = file.tellg();
    n_data_points = 0;
    std::string line;
    std::vector<bool> valid_rows;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream line_stream(line);
        std::string date;
        for (int i = 0; i <= date_idx; ++i) {
            std::getline(line_stream, date, ',');
        }
        bool is_valid = isDateInRange(date, start_date, end_date);
        valid_rows.push_back(is_valid);
        if (is_valid) ++n_data_points;
    }

    if (n_data_points == 0) {
        std::cerr << "Error: No data points found in specified date range." << std::endl;
        return false;
    }

    file.clear();
    file.seekg(data_start);
    // Resize Eigen matrices
    new_confirmed_cases.resize(n_data_points, num_age_classes);
    new_deaths.resize(n_data_points, num_age_classes);
    new_hospitalizations.resize(n_data_points, num_age_classes);
    new_icu.resize(n_data_points, num_age_classes);
    cumulative_confirmed_cases.resize(n_data_points, num_age_classes);
    cumulative_deaths.resize(n_data_points, num_age_classes);
    cumulative_hospitalizations.resize(n_data_points, num_age_classes);
    cumulative_icu.resize(n_data_points, num_age_classes);
    population_by_age.resize(num_age_classes);
    /**
     * @brief Lambda function to parse a CSV row into vector of strings
     * @param line String containing a CSV row
     * @return std::vector<std::string> Vector of cell values
     */
    auto parse_csv_row = [](const std::string& line) {
        std::vector<std::string> row;
        std::istringstream line_stream(line);
        std::string cell;
        while (std::getline(line_stream, cell, ',')) row.push_back(cell);
        return row;
    };

    int row_idx = 0;
    size_t valid_idx = 0;
    bool population_set = false;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (!valid_rows[valid_idx++]) continue;

        auto row = parse_csv_row(line);
        size_t required_cols = static_cast<size_t>(std::max({
            date_idx, confirmed_idx[0], confirmed_idx[1], confirmed_idx[2], confirmed_idx[3],
            deceased_idx[0], deceased_idx[1], deceased_idx[2], deceased_idx[3],
            hospitalized_idx[0], hospitalized_idx[1], hospitalized_idx[2], hospitalized_idx[3],
            icu_idx[0], icu_idx[1], icu_idx[2], icu_idx[3],
            population_idx[0], population_idx[1], population_idx[2], population_idx[3],
            cumulative_confirmed_idx[0], cumulative_confirmed_idx[1], cumulative_confirmed_idx[2], cumulative_confirmed_idx[3],
            cumulative_deceased_idx[0], cumulative_deceased_idx[1], cumulative_deceased_idx[2], cumulative_deceased_idx[3],
            cumulative_hospitalized_idx[0], cumulative_hospitalized_idx[1], cumulative_hospitalized_idx[2], cumulative_hospitalized_idx[3],
            cumulative_icu_idx[0], cumulative_icu_idx[1], cumulative_icu_idx[2], cumulative_icu_idx[3]
        })) + 1;
        if (row.size() < required_cols) {
            std::cerr << "Error: Insufficient columns in data row " << row_idx << std::endl;
            return false;
        }

        dates.push_back(row[date_idx]);
        /**
         * @brief Lambda function to parse a string to double
         * @param s String to parse
         * @return double Parsed numeric value
         * @throws std::runtime_error If parsing fails
         */
        auto parse_value = [&](const std::string& s) {
            double value;
            auto result = std::from_chars(s.data(), s.data() + s.size(), value);
            if (result.ec != std::errc()) throw std::runtime_error("Failed to parse value: " + s);
            return value;
        };

        for (int age = 0; age < num_age_classes; ++age) {
            new_confirmed_cases(row_idx, age) = parse_value(row[confirmed_idx[age]]);
            new_deaths(row_idx, age) = parse_value(row[deceased_idx[age]]);
            new_hospitalizations(row_idx, age) = parse_value(row[hospitalized_idx[age]]);
            new_icu(row_idx, age) = parse_value(row[icu_idx[age]]);
            cumulative_confirmed_cases(row_idx, age) = parse_value(row[cumulative_confirmed_idx[age]]);
            cumulative_deaths(row_idx, age) = parse_value(row[cumulative_deceased_idx[age]]);
            cumulative_hospitalizations(row_idx, age) = parse_value(row[cumulative_hospitalized_idx[age]]);
            cumulative_icu(row_idx, age) = parse_value(row[cumulative_icu_idx[age]]);
        }

        if (!population_set) {
            for (int age = 0; age < num_age_classes; ++age) {
                population_by_age[age] = parse_value(row[population_idx[age]]);
            }
            population_set = true;
        }

        ++row_idx;
    }

    return true;
}

}