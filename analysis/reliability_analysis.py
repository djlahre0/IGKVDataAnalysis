import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


def load_sheets(file_path: str, sheet_config: List[Tuple]) -> Dict[str, pd.DataFrame]:
    """Load multiple sheets from Excel file with configuration."""
    sheets = {}
    for sheet_name, cols, skiprows, label in sheet_config:
        try:
            df = pd.read_excel(
                file_path, sheet_name=sheet_name, usecols=cols, skiprows=skiprows
            )
            df = df.drop(0)  # removes row with Q1, Q2..
            sheets[label] = df
        except Exception as e:
            print(f"Error loading {label}: {str(e)}")
    return sheets


def reliability_analysis(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """Calculate reliability metrics for multiple dataframes."""

    def _cronbach_alpha(df: pd.DataFrame) -> float:
        """Calculate Cronbach's Alpha for reliability assessment."""
        k = df.shape[1]
        item_variances = df.var(ddof=1)
        total_scores = df.sum(axis=1)
        total_variance = total_scores.var(ddof=1)
        return (k / (k - 1)) * (1 - item_variances.sum() / total_variance)

    return {label: _cronbach_alpha(df) for label, df in dataframes.items()}


def display_results(results, title="Cronbach Alpha Results"):
    results_dict = {}
    for label, alpha in results.items():
        results_dict[label] = f"{alpha:.3f}"

    df = pd.DataFrame(
        list(results_dict.items()), columns=["Constructs", "Cronbach’s Alpha(α)"]
    )
    print(f"------------------{title}--------------------")
    print(df)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")  # Hide axes
    pd.plotting.table(ax, df, loc="center", cellLoc="center", colWidths=[0.4, 0.2])
    plt.tight_layout()
    plt.savefig(f"./analysis/results/{title}")
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        SHEET_CONFIG = [
            ("Sheet1", "C:F", 3, "Resource Related Issues"),
            ("Sheet2", "C:E", 3, "Level of Computer Illiteracy"),
            ("Sheet3", "C:E", 3, "Limited citizens’ Awareness"),
            ("Sheet4", "C:D", 3, "Challenges of Language in Rural Influence"),
            ("Sheet5", "C:E", 3, "Resistance to change"),
            ("Sheet6", "C:E", 3, "Lack Of Trained Persons"),
            ("Sheet7", "C:E", 3, "Shortage Of Equipments"),
            ("Sheet8", "C:E", 3, "Level of Difficulty"),
            ("Sheet9", "C:D", 3, "Gender"),
        ]

        sheets = load_sheets("Agriculture Data.xlsx", SHEET_CONFIG)
        results = reliability_analysis(sheets)
        display_results(results, "Agriculture Data Cronbach Alpha Results")

        sheets = load_sheets("Horticulture Data.xlsx", SHEET_CONFIG)
        results = reliability_analysis(sheets)
        display_results(results, "Horticulture Data Cronbach Alpha Results")

    except FileNotFoundError:
        print("Error: Excel file not found. Please check the file path.")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
