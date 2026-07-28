"""
eda.py
-------
Exploratory Data Analysis (EDA) for the Titanic dataset.

This module is responsible ONLY for producing visualizations that help
explore relationships between columns before modelling. Nothing in here
transforms the data used by the pipeline - it reads the (already cleaned)
training DataFrame and writes PNG figures to outputs/figures/.

Five chart types are used, each picked because it's the right tool for
the relationship it's exploring:

1. Histogram        -> distribution of a single numeric column (Age)
2. Boxplot           -> spread/outliers of a numeric column across groups
                        (Age by Survived, Fare by Pclass)
3. Heatmap           -> correlation strength between every numeric column
4. Line plot         -> trend of survival rate across ordered Age groups
5. Scatter plot      -> relationship between two continuous variables
                        (Age vs Fare), coloured by survival outcome
"""

from pathlib import Path

import matplotlib

# Non-interactive backend so figures save to disk instead of blocking
# execution with a pop-up window when run as a plain script.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def plot_age_histogram(dataframe: pd.DataFrame, figures_dir: Path) -> None:
    """Histogram: how passenger ages are distributed.

    Why a histogram: 'Age' is a single continuous numeric column, and a
    histogram is the standard way to see its shape (skew, common ranges,
    where most passengers cluster) at a glance.

    Args:
        dataframe: DataFrame containing an 'Age' column.
        figures_dir: Folder to save the resulting PNG into.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    dataframe["Age"].hist(bins=30, ax=ax, color="#4C72B0", edgecolor="black")
    ax.set_title("Distribution of Passenger Age")
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of Passengers")

    output_path = figures_dir / "age_histogram.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Age histogram to {output_path}")


def plot_boxplots(dataframe: pd.DataFrame, figures_dir: Path) -> None:
    """Boxplots: 'Age' split by 'Survived', and 'Fare' split by 'Pclass'.

    Why a boxplot: boxplots are the right tool for comparing the spread
    and outliers of a numeric column across a handful of categories.
    - Age vs Survived shows whether survivors skewed younger/older.
    - Fare vs Pclass shows how much wealthier passengers (by fare) paid,
      and highlights extreme outliers in first class fares.

    Args:
        dataframe: DataFrame containing 'Age', 'Survived', 'Fare', and 'Pclass'.
        figures_dir: Folder to save the resulting PNG into.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Age grouped by Survived (0 = did not survive, 1 = survived)
    age_groups = [
        dataframe.loc[dataframe["Survived"] == 0, "Age"].dropna(),
        dataframe.loc[dataframe["Survived"] == 1, "Age"].dropna(),
    ]
    axes[0].boxplot(age_groups, labels=["Did Not Survive", "Survived"])
    axes[0].set_title("Age by Survival Outcome")
    axes[0].set_ylabel("Age")

    # Fare grouped by Pclass (1st, 2nd, 3rd)
    fare_groups = [
        dataframe.loc[dataframe["Pclass"] == pclass, "Fare"].dropna()
        for pclass in sorted(dataframe["Pclass"].unique())
    ]
    axes[1].boxplot(fare_groups, labels=[f"Class {p}" for p in sorted(dataframe["Pclass"].unique())])
    axes[1].set_title("Fare by Passenger Class")
    axes[1].set_ylabel("Fare")

    fig.tight_layout()
    output_path = figures_dir / "boxplots.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved boxplots to {output_path}")


def plot_correlation_heatmap(dataframe: pd.DataFrame, figures_dir: Path) -> None:
    """Heatmap: correlation between all numeric columns.

    Why a heatmap: it's the fastest way to scan many pairwise
    relationships at once (e.g. does 'Pclass' correlate with 'Fare'?
    Does 'Sex' correlate with 'Survived'?) without plotting dozens of
    individual scatter plots.

    'Sex' is temporarily mapped to 0/1 (male/female) on a COPY of the
    data purely so it can be included in the correlation matrix - this
    does not affect the DataFrame used by the rest of the pipeline.

    Args:
        dataframe: DataFrame containing the Titanic columns.
        figures_dir: Folder to save the resulting PNG into.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    numeric_data = dataframe.copy()
    if "Sex" in numeric_data.columns and not pd.api.types.is_numeric_dtype(numeric_data["Sex"]):
        numeric_data["Sex"] = numeric_data["Sex"].map({"male": 0, "female": 1})

    candidate_columns = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
    columns_present = [col for col in candidate_columns if col in numeric_data.columns]
    correlation_matrix = numeric_data[columns_present].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(range(len(columns_present)))
    ax.set_yticks(range(len(columns_present)))
    ax.set_xticklabels(columns_present, rotation=45, ha="right")
    ax.set_yticklabels(columns_present)

    # Annotate each cell with its correlation value for easy reading.
    for row in range(len(columns_present)):
        for col in range(len(columns_present)):
            value = correlation_matrix.iloc[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color="black", fontsize=8)

    ax.set_title("Correlation Heatmap of Numeric Features")
    fig.colorbar(im, ax=ax, label="Correlation Coefficient")
    fig.tight_layout()

    output_path = figures_dir / "correlation_heatmap.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved correlation heatmap to {output_path}")


def plot_survival_rate_by_age_group(dataframe: pd.DataFrame, figures_dir: Path) -> None:
    """Line plot: survival rate trend across ordered Age groups.

    Why a line plot: 'Age' bucketed into ordered groups (0-10, 10-20, ...)
    forms a natural sequence, and a line plot is the clearest way to show
    a trend across an ordered axis - here, whether survival rate rises or
    falls as passengers get older.

    Args:
        dataframe: DataFrame containing 'Age' and 'Survived' columns.
        figures_dir: Folder to save the resulting PNG into.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    bin_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    bin_labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80"]

    age_groups = pd.cut(dataframe["Age"], bins=bin_edges, labels=bin_labels, include_lowest=True)
    survival_rate_by_group = dataframe.groupby(age_groups, observed=True)["Survived"].mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        survival_rate_by_group.index.astype(str),
        survival_rate_by_group.values,
        marker="o",
        color="#C44E52",
    )
    ax.set_title("Survival Rate by Age Group")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Survival Rate")
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.5)

    output_path = figures_dir / "survival_rate_by_age_group.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved survival rate line plot to {output_path}")


def plot_age_vs_fare_scatter(dataframe: pd.DataFrame, figures_dir: Path) -> None:
    """Scatter plot: relationship between 'Age' and 'Fare', coloured by survival.

    Why a scatter plot: both 'Age' and 'Fare' are continuous numeric
    columns, and a scatter plot is the standard way to look for a
    relationship (or lack of one) between two continuous variables -
    colouring by 'Survived' additionally reveals whether that
    relationship differs between passengers who lived and died.

    Args:
        dataframe: DataFrame containing 'Age', 'Fare', and 'Survived'.
        figures_dir: Folder to save the resulting PNG into.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        dataframe["Age"],
        dataframe["Fare"],
        c=dataframe["Survived"],
        cmap="coolwarm",
        alpha=0.6,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_title("Age vs Fare, Coloured by Survival")
    ax.set_xlabel("Age")
    ax.set_ylabel("Fare")

    legend_labels = ["Did Not Survive", "Survived"]
    handles, _ = scatter.legend_elements()
    ax.legend(handles, legend_labels, title="Outcome")

    output_path = figures_dir / "age_vs_fare_scatter.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Age vs Fare scatter plot to {output_path}")


def run_full_eda(dataframe: pd.DataFrame, figures_dir: Path) -> None:
    """Run every EDA plot in sequence and save them all to figures_dir.

    Call this once on the cleaned training DataFrame (after missing
    values are filled, but before 'Survived' is split off into `y` and
    before categorical columns are encoded), so every plot has access
    to the original, human-readable column values.

    Args:
        dataframe: The cleaned training DataFrame.
        figures_dir: Folder to save all resulting PNGs into.
    """
    print("Running exploratory data analysis...")
    plot_age_histogram(dataframe, figures_dir)
    plot_boxplots(dataframe, figures_dir)
    plot_correlation_heatmap(dataframe, figures_dir)
    plot_survival_rate_by_age_group(dataframe, figures_dir)
    plot_age_vs_fare_scatter(dataframe, figures_dir)
    print("Exploratory data analysis complete.")