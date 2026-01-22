import polars as pl
from pathlib import Path

DATA_PATH = Path("movie_dataset/wiki_movie_plots_deduped.csv")
OUTPUT_PATH = Path("movie_dataset/processed_movie_plots.csv")


def load_movie_plots(n_rows: int = 300, seed: int = 42) -> pl.DataFrame:
    """Load and preprocess a subset of the Wikipedia Movie Plots dataset.

    Args:
        n_rows: Number of rows to sample (200-500 recommended)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with 'Title' and 'Plot' columns
    """
    df = pl.read_csv(DATA_PATH, columns=["Title", "Plot"])

    df = df.drop_nulls()

    df = df.with_columns(
        pl.col("Plot").str.replace_all(r"\s+", " ").str.strip_chars().alias("Plot")
    )

    df = df.filter(pl.col("Plot").str.len_chars() > 300)

    df = df.sample(n=min(n_rows, len(df)), seed=seed)

    return df


def save_processed_data(df: pl.DataFrame) -> None:
    """Save the processed DataFrame to a CSV file.

    Args:
        df: DataFrame to save
    """
    df.write_csv(OUTPUT_PATH)


# if __name__ == "__main__":
#     df = load_movie_plots(n_rows=300)
#     save_processed_data(df)
