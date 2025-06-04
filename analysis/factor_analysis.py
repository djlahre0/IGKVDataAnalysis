import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo


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


def factor_analysis(dataframes: Dict[str, pd.DataFrame]):
    """Factor analysis for multiple dataframes."""
    df = pd.DataFrame()
    for label, curr_df in dataframes.items():
        df = pd.concat([df, curr_df], axis=1)

    df.dropna(inplace=True)

    # df = pd.read_excel("Agriculture Data.xlsx", usecols="J:N", skiprows=5, nrows=26)
    # df = df.set_index(df.columns[0])
    # # df = df.drop(df.columns[0], axis=1)
    # df = df.T
    # df.reset_index(inplace=True, drop=True)

    # 1. KMO test
    df = df.astype(int)
    kmo_all, kmo_model = calculate_kmo(df)
    print(f"KMO Measure: {kmo_model:.3f}")

    # 2. Scree plot to determine number of factors
    fa = FactorAnalyzer(rotation=None)
    fa.fit(df)
    ev, _ = fa.get_eigenvalues()

    plt.plot(range(1, len(ev) + 1), ev, marker="o")
    plt.axhline(y=1, color="r", linestyle="--")
    plt.title("Scree Plot")
    plt.xlabel("Factors")
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    plt.show()

    # 3. Factor Analysis with 2 components and varimax rotation
    fa = FactorAnalyzer(n_factors=2, rotation="varimax")
    fa.fit(df)

    # 4. Get and print rotated component matrix (factor loadings)
    loadings = pd.DataFrame(fa.loadings_, index=df.columns)
    loadings = loadings.round(3)
    loadings.columns = ["PC1", "PC2"]
    loadings.drop_duplicates(inplace=True)
    print("\nRotated Component Matrix:\n")
    print(loadings)

    # loadings = loadings[loadings.abs().gt(0.7).any(axis=1)]
    # loadings = loadings.abs()
    # # loadings = loadings.dropna(how='all')
    # print(loadings)

    # Split components based on higher loading for visualization
    factor2_vars = loadings[loadings["PC1"] > loadings["PC2"]]
    factor1_vars = loadings[loadings["PC2"] > loadings["PC1"]]

    # Plotting
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # Arrows for Factor 1 (red)
    for i in factor1_vars.index:
        x, y = loadings.loc[i, ["PC1", "PC2"]]
        ax.arrow(
            0,
            0,
            -x,
            y,
            color="red",
            width=0.003,
            head_width=0.03,
            length_includes_head=True,
        )
        plt.text(-x * 1.05, y * 1.05, i, color="red", ha="right", va="top")

    # Arrows for Factor 2 (blue)
    for i in factor2_vars.index:
        x, y = loadings.loc[i, ["PC1", "PC2"]]
        ax.arrow(
            0,
            0,
            x,
            y,
            color="blue",
            width=0.003,
            head_width=0.03,
            length_includes_head=True,
        )
        plt.text(x * 1.05, y * 1.05, i, color="blue", ha="left", va="center")

    # Axes and Labels
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Factor 1")
    plt.ylabel("Factor 2")
    plt.title(
        "Rotated Component Matrix Graph of Agriculture Scheme\n(Factor 1 vs Factor 2)",
        fontsize=12,
        weight="bold",
    )
    plt.grid(False)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


if __name__ == "__main__":
    try:
        SHEET_CONFIG = [
            ("Sheet1", "C:F", 3, "Resource Related Issues"),
            # ("Sheet2", "C:E", 3, "Level of Computer Illiteracy"),
            ("Sheet3", "C:E", 3, "Limited citizens’ Awareness"),
            ("Sheet3", "C:D", 3, "Challenges of Language in Rural Influence"),
            # ("Sheet5", "C:E", 3, "Resistance to change"),
            ("Sheet6", "C:E", 3, "Lack Of Trained Persons"),
            # ("Sheet7", "C:E", 3, "Shortage Of Equipments"),
            # ("Sheet8", "C:E", 3, "Level of Difficulty"),
            # ("Sheet9", "C:D", 3, "Gender")
        ]

        sheets = load_sheets("Agriculture Data.xlsx", SHEET_CONFIG)
        results = factor_analysis(sheets)

        # sheets = load_sheets("Horticulture Data.xlsx", SHEET_CONFIG)
        # results = factor_analysis(sheets)

    except FileNotFoundError:
        print("Error: Excel file not found. Please check the file path.")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise e
