from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


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
        Falls back to unicode_escape encoding if UTF-8 loading fails.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Billboard dataset not found: {self.file_path}"
            )

        try:
            self.data = pd.read_csv(self.file_path)
        except UnicodeDecodeError:
            self.data = pd.read_csv(
                self.file_path,
                encoding="unicode_escape"
            )

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

    def add_chart_dates(self):
        """
        Convert the Billboard entry date to datetime and calculate the
        corresponding chart date for each weekly ranking observation.

        Formula:
            Chart_Date = Entry_Date + (Week_Number - 1) * 7 days

        Week 1 therefore corresponds to the original entry date.
        """
        if self.cleaned_data is None:
            raise ValueError(
                "Cleaned ranking data is unavailable. "
                "Call clean_rankings() first."
            )

        dated = self.cleaned_data.copy()

        # Convert the original entry-date string to a datetime value
        dated["Entry_Date"] = pd.to_datetime(
            dated["date.entered"],
            errors="coerce"
        )

        # Calculate the date corresponding to each weekly ranking
        dated["Chart_Date"] = (
            dated["Entry_Date"]
            + pd.to_timedelta(
                (dated["Week_Number"] - 1) * 7,
                unit="D"
            )
        )

        self.cleaned_data = dated

        return self.cleaned_data

    def get_song_trend(self, artist, track):
        """
        Return the weekly Billboard ranking history for a selected song.
        """
        if self.cleaned_data is None:
            raise ValueError(
                "Cleaned Billboard data is unavailable. "
                "Complete the preprocessing steps first."
            )

        song_data = self.cleaned_data[
            (self.cleaned_data["artist"] == artist)
            & (self.cleaned_data["track"] == track)
        ].copy()

        if song_data.empty:
            raise ValueError(
                f"No data found for artist='{artist}' and track='{track}'."
            )

        return song_data.sort_values("Week_Number")


    def plot_song_trend(self, artist, track):
        """
        Plot the Billboard ranking trend for a selected song.

        The y-axis is inverted because rank 1 is the best position.
        """
        song_data = self.get_song_trend(
            artist,
            track
        )

        plt.figure(figsize=(10, 6))

        plt.plot(
            song_data["Week_Number"],
            song_data["Rank"],
            marker="o"
        )

        # Billboard rank 1 is the best possible position
        upper_limit = min(100, int(song_data["Rank"].max()) + 5)
        plt.ylim(upper_limit, 1)

        plt.title(
            f"Billboard Ranking Trend: {artist} - {track}"
        )
        plt.xlabel("Week Number")
        plt.ylabel("Billboard Rank")

        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        return song_data