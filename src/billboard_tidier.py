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