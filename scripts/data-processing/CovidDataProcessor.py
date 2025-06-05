import pandas as pd

class CovidDataProcessor:
    """Processes COVID-19 data by aggregating age-related statistics."""

    def __init__(self):
        """Initialize processor with allowed columns and age group definitions."""
        self.allowed_non_age_columns = {
            "date", "population",
            "new_confirmed", "new_deceased", "cumulative_confirmed", "cumulative_deceased",
            "new_hospitalized_patients", "cumulative_hospitalized_patients",
            "new_intensive_care_patients", "cumulative_intensive_care_patients",
        }
        self.define_aggregated_groups()

    def define_aggregated_groups(self):
        """Sets up age group aggregation rules for COVID metrics.

        Creates mapping of input age-specific columns to aggregated output columns.
        Defines four age groups: 0-30, 30-60, 60-80, and 80+ years.
        """
        self.aggregated_groups = []
        metric_bases = [
            "new_confirmed", "cumulative_confirmed", "new_deceased", "cumulative_deceased",
            "new_hospitalized_patients", "cumulative_hospitalized_patients",
            "new_intensive_care_patients", "cumulative_intensive_care_patients"
        ]

        # Map metrics to age ranges
        for base in metric_bases:
            self.aggregated_groups.extend([
                {"output_name": f"{base}_0_30", "input_patterns": [f"{base}_age_{i}" for i in range(3)]},
                {"output_name": f"{base}_30_60", "input_patterns": [f"{base}_age_{i}" for i in range(3, 6)]},
                {"output_name": f"{base}_60_80", "input_patterns": [f"{base}_age_{i}" for i in range(6, 8)]},
                {"output_name": f"{base}_80_plus", "input_patterns": [f"{base}_age_8"]}
            ])

        # Population groups
        self.aggregated_groups.extend([
            {"output_name": "population_0_30", 
             "input_patterns": ["population_age_00_09", "population_age_10_19", "population_age_20_29"]},
            {"output_name": "population_30_60",
             "input_patterns": ["population_age_30_39", "population_age_40_49", "population_age_50_59"]},
            {"output_name": "population_60_80",
             "input_patterns": ["population_age_60_69", "population_age_70_79"]},
            {"output_name": "population_80_plus",
             "input_patterns": ["population_age_80_and_older"]}
        ])

    def validate_header(self, columns):
        """Checks if required columns exist in the dataset.

        Args:
            columns: List of column names in the dataset.

        Returns:
            bool: True if all required columns present, False otherwise.
        """
        if "country_code" not in columns:
            print("Error: Required column 'country_code' missing.")
            return False
        if "date" not in columns:
            print("Error: Required column 'date' missing.")
            return False
        return True

    def process_data(self, input_filename, output_filename, start_date='2020-03-01', end_date='2020-12-31'):
        """Processes and aggregates COVID-19 data by age groups.
    
        Args:
            input_filename (str): Path to input CSV file.
            output_filename (str): Path for output CSV file.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
    
        Returns:
            bool: True if processing successful, False otherwise.
        """
        try:
            # print start date and end date
            print(f"Processing data from {start_date} to {end_date}")
            # Load and validate data
            df = pd.read_csv(input_filename)
            if not self.validate_header(df.columns):
                return False
    
            # Filter data by date range
            df['date'] = pd.to_datetime(df['date'])
            original_count = len(df)
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            filtered_count = original_count - len(df)
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
            # Extract relevant columns
            non_age_columns = [col for col in df.columns if col in self.allowed_non_age_columns]
            
            # Identify valid aggregation groups
            valid_groups = [
                {"output_name": group["output_name"],
                 "input_columns": [col for col in group["input_patterns"] if col in df.columns]}
                for group in self.aggregated_groups
                if any(col in df.columns for col in group["input_patterns"])
            ]
    
            # Prepare aggregations
            aggregated_data = {}
            for group in valid_groups:
                # Convert columns to numeric
                numeric_cols = {col: pd.to_numeric(df[col], errors='coerce') 
                              for col in group["input_columns"]}
                # Calculate sum for the group
                aggregated_data[group["output_name"]] = pd.DataFrame(numeric_cols).sum(axis=1)
    
            # Combine original and aggregated data
            if aggregated_data:
                base_df = df[non_age_columns].copy()
                agg_df = pd.DataFrame(aggregated_data)
                result_df = pd.concat([base_df, agg_df], axis=1)
            else:
                result_df = df[non_age_columns].copy()
    
            # Save results
            result_df.to_csv(output_filename, index=False)
            
            print(f"Data aggregated and saved to {output_filename}")
            print(f"Processed {original_count} rows. Filtered {filtered_count} non-2020 rows.")
            return True
    
        except Exception as e:
            print(f"Error processing data: {e}")
            return False