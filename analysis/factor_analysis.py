import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer, calculate_kmo


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


def factor_analysis(sheets, img_label=""):
    """Factor analysis"""
    df = pd.DataFrame()
    for label, curr_df in sheets.items():
        df = pd.concat([df, curr_df], axis=1)

    # Step 1: Clean and prepare the data
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.loc[:, df.std() > 1e-6]
    df.dropna(inplace=True)
    df = df.astype(int)

    # Drop non-numeric columns
    df_numeric = df.select_dtypes(include="number").dropna()

    # Heatmap of correlations
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm")
    # plt.title("Correlation Heatmap")
    # plt.tight_layout()
    # plt.show()

    # Step 2: Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numeric)

    # Step 3: KMO
    kmo_all, kmo_model = calculate_kmo(df_scaled)
    print(f"KMO Value: {kmo_model:.3f}")

    # Step 4: Initial Factor Analysis
    fa = FactorAnalyzer(rotation=None)
    fa.fit(df_scaled)
    ev, _ = fa.get_eigenvalues()

    # Scree Plot
    # plt.plot(range(1, len(ev) + 1), ev, marker="o")
    # plt.title("Scree Plot")
    # plt.xlabel("Factors")
    # plt.ylabel("Eigenvalue")
    # plt.grid()
    # plt.tight_layout()
    # plt.show()

    # Step 5: Final Factor Analysis (e.g., 2 factors with Varimax rotation)
    fa = FactorAnalyzer(n_factors=2, rotation="varimax")
    fa.fit(df_scaled)
    loadings = pd.DataFrame(
        fa.loadings_, index=df_numeric.columns, columns=["PC1", "PC2"]
    )
    loadings = loadings.round(3)
    print("\nFactor Loadings:\n", loadings)

    variance = fa.get_factor_variance()
    # Multiply proportions by 100 and round
    percent_variance = (variance[1] * 100).round(2)
    cumulative_variance = (variance[2] * 100).round(2)
    print(
        "\nFactor Variance:\n",
        pd.DataFrame(
            {
                "Eigen Value": variance[0],
                "Percent of Total Variance": percent_variance,
                "Cumulative Variance": cumulative_variance,
            },
            index=["PC1", "PC2"],
        ),
    )

    loadings = loadings.drop(['Somewhat Oppose', 'Strongly Oppose'], axis=0)

    final_table = loadings.copy()
    final_table["KMO"] = ""  # Leave blank, as it's a single value
    final_table.loc["Eigen value"] = list(variance[0]) + [""]
    final_table.loc["Percent of total variation"] = list(percent_variance) + [""]
    final_table.loc["Cumulative variance explain %"] = list(cumulative_variance) + [""]
    final_table["KMO"].iloc[0] = round(kmo_model, 3)  # Add KMO in first row
    print(final_table)

    # Step 7: Filter loadings by threshold for clarity
    threshold = 0.4
    loadings_filtered = loadings[
        (abs(loadings["PC1"]) >= threshold) | (abs(loadings["PC2"]) >= threshold)
    ]

    # Heatmap of Loadings
    plt.figure(figsize=(8, 6))
    sns.heatmap(loadings_filtered, annot=True, cmap="coolwarm", center=0)
    plt.title(f"Factor Loadings Heatmap {img_label}", fontsize=12, weight="bold")
    plt.tight_layout()
    plt.show()

    # Separate by dominant factor
    factor1_vars = loadings_filtered[
        loadings_filtered["PC1"] > loadings_filtered["PC2"]
    ]
    factor1_vars = factor1_vars.reindex(
        factor1_vars["PC1"].abs().sort_values(ascending=False).index
    )

    factor2_vars = loadings_filtered[
        loadings_filtered["PC2"] > loadings_filtered["PC1"]
    ]
    factor2_vars = factor2_vars.reindex(
        factor2_vars["PC2"].abs().sort_values(ascending=False).index
    )

    # Step 8: Visualization
    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    y = 0.2
    for i in factor1_vars.index:
        x, _y = loadings.loc[i, ["PC1", "PC2"]]
        x -= 0.33
        ax.arrow(
            0,
            0,
            -x,
            y,
            color="red",
            width=0.002,
            head_width=0.03,
            length_includes_head=True,
        )
        plt.text(-x * 1.05, y * 1.05, i, color="red", ha="right", va="top")
        y += 0.12

    y = 0.2
    for i in factor2_vars.index:
        x, _y = loadings.loc[i, ["PC1", "PC2"]]
        x += 0.25
        ax.arrow(
            0,
            0,
            x,
            y,
            color="blue",
            width=0.002,
            head_width=0.03,
            length_includes_head=True,
        )
        plt.text(x * 1.05, y * 1.05, i, color="blue", ha="left", va="center")
        y += 0.15

    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Factor 1", color="red", weight="bold", fontsize=12)
    plt.ylabel("Factor 2", color="blue", weight="bold", fontsize=12)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(f"./analysis/results/Rotated Component Matrix Graph of {img_label}")
    plt.title(
        f"Rotated Component Matrix Graph of {img_label} Scheme",
        fontsize=14,
        weight="bold",
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        SHEET_CONFIG = [
            ("Sheet1", "C:F", 3, "Resource Related Issues"),
            # ("Sheet2", "C:E", 3, "Level of Computer Illiteracy"),
            ("Sheet3", "C:E", 3, "Limited citizens’ Awareness"),
            ("Sheet3", "C:D", 3, "Challenges of Language in Rural Influence"),
            ("Sheet5", "C:E", 3, "Resistance to change"),
            ("Sheet6", "C:E", 3, "Lack Of Trained Persons"),
            # ("Sheet7", "C:E", 3, "Shortage Of Equipments"),
            # ("Sheet8", "C:E", 3, "Level of Difficulty"),
            # ("Sheet9", "C:D", 3, "Gender")
        ]

        sheets = load_sheets("Agriculture Data.xlsx", SHEET_CONFIG)
        results = factor_analysis(sheets, "Agriculture")

        sheets = load_sheets("Horticulture Data.xlsx", SHEET_CONFIG)
        results = factor_analysis(sheets, "Horticulture")

    except FileNotFoundError:
        print("Error: Excel file not found. Please check the file path.")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise e
