from pathlib import Path

import pandas as pd


class BillboardTidier:
    """
    Handles loading, validation, inspection, and preprocessing
    for the Billboard Top 100 dataset.
    """

    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.data = None
        self.long_data = None
        self.cleaned_data = None
        self.id_columns = []
        self.week_columns = []

    def load_data(self):
        """
        Load the raw Billboard dataset from a local relative path.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Billboard dataset not found: {self.file_path}"
            )

        self.data = pd.read_csv(self.file_path)

        return self.data

    def validate_schema(self):
        """
        Validate the required identifier columns and weekly ranking columns.
        """
        if self.data is None:
            raise ValueError(
                "Dataset has not been loaded. Call load_data() first."
            )

        required_columns = {
            "year",
            "artist",
            "track",
            "time",
            "date.entered"
        }

        missing_required = required_columns - set(self.data.columns)

        if missing_required:
            raise ValueError(
                f"Missing required columns: {sorted(missing_required)}"
            )

        self.week_columns = [
            column
            for column in self.data.columns
            if column.startswith("wk")
            and column[2:].isdigit()
        ]

        self.week_columns = sorted(
            self.week_columns,
            key=lambda column: int(column.replace("wk", ""))
        )

        self.id_columns = [
            column
            for column in self.data.columns
            if column not in self.week_columns
        ]

        if len(self.week_columns) == 0:
            raise ValueError(
                "No weekly Billboard ranking columns were found."
            )

        return self.id_columns, self.week_columns

    def inspect_data(self):
        """
        Return important information for the initial dataset inspection.
        """
        if self.data is None:
            raise ValueError(
                "Dataset has not been loaded. Call load_data() first."
            )

        return {
            "shape": self.data.shape,
            "columns": self.data.columns.tolist(),
            "dtypes": self.data.dtypes,
            "head": self.data.head(),
            "missing_values": self.data.isna().sum(),
            "duplicate_rows": self.data.duplicated().sum()
        }

    def reshape_data(self):
        """
        Reshape the Billboard dataset from wide format to long/tidy format.

        The weekly ranking columns (wk1 to wk76) are converted into:
        - Week: the original weekly column label
        - Rank: the Billboard ranking value
        """
        if self.data is None:
            raise ValueError(
                "Dataset has not been loaded. Call load_data() first."
            )

        # Ensure identifier and week columns have been identified
        if not self.id_columns or not self.week_columns:
            self.validate_schema()

        self.long_data = self.data.melt(
            id_vars=self.id_columns,
            value_vars=self.week_columns,
            var_name="Week",
            value_name="Rank"
        )

        return self.long_data


    def clean_week_labels(self):
        """
        Clean Billboard week labels using Pandas string operations
        and create a numeric Week_Number column.
        """
        if self.long_data is None:
            raise ValueError(
                "Data has not been reshaped. Call reshape_data() first."
            )

        cleaned = self.long_data.copy()

        # Preserve the original week label for comparison
        cleaned["Week_Original"] = cleaned["Week"]

        # String operation 1: remove leading/trailing whitespace
        cleaned["Week"] = cleaned["Week"].str.strip()

        # String operation 2: standardize text to lowercase
        cleaned["Week"] = cleaned["Week"].str.lower()

        # String operation 3: replace 'wk' with a clearer label
        cleaned["Week"] = cleaned["Week"].str.replace(
            "wk",
            "week ",
            regex=False
        )

        # String operation 4: extract the numeric week value
        cleaned["Week_Number"] = cleaned["Week"].str.extract(
            r"(\d+)",
            expand=False
        )

        # Convert extracted week number to numeric
        cleaned["Week_Number"] = pd.to_numeric(
            cleaned["Week_Number"],
            errors="coerce"
        )

        self.long_data = cleaned

        return self.long_data


    def clean_rankings(self):
        """
        Convert Billboard ranking values to numeric, assess missing values,
        validate the ranking range, and create an analysis-ready dataset.

        Missing ranking rows are removed because they represent weeks in
        which no chart ranking is recorded. They are not imputed because
        doing so would create artificial Billboard positions.
        """
        if self.long_data is None:
            raise ValueError(
                "Long-format data is unavailable. "
                "Call reshape_data() and clean_week_labels() first."
            )

        cleaned = self.long_data.copy()

        total_rows_before = len(cleaned)

        # Convert Rank to numeric.
        # Non-numeric values, if present, become NaN.
        cleaned["Rank"] = pd.to_numeric(
            cleaned["Rank"],
            errors="coerce"
        )

        missing_rank_count = cleaned["Rank"].isna().sum()

        missing_rank_percentage = (
            missing_rank_count / total_rows_before
        ) * 100

        # Billboard Hot 100 ranks should fall between 1 and 100.
        invalid_rank_mask = (
            cleaned["Rank"].notna()
            & ~cleaned["Rank"].between(1, 100)
        )

        invalid_rank_count = invalid_rank_mask.sum()

        # Remove observations without an actual recorded chart rank.
        cleaned = cleaned.dropna(
            subset=["Rank"]
        ).copy()
        # Billboard rankings are whole-number positions
        cleaned["Rank"] = cleaned["Rank"].astype("int64")

        cleaned.reset_index(
            drop=True,
            inplace=True
        )

        self.cleaned_data = cleaned

        summary = {
            "Rows Before Treatment": total_rows_before,
            "Missing Rankings": int(missing_rank_count),
            "Missing Percentage": missing_rank_percentage,
            "Invalid Rankings Outside 1-100": int(invalid_rank_count),
            "Rows After Treatment": len(cleaned),
            "Rows Removed": total_rows_before - len(cleaned),
            "Minimum Valid Rank": cleaned["Rank"].min(),
            "Maximum Valid Rank": cleaned["Rank"].max()
        }

        return self.cleaned_data, summary

    