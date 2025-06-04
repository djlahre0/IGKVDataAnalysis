import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage


def cluster_analysis(df: pd.DataFrame, title="Hierarchical Clustering Dendrogram"):
    """Cluster analysis."""
    df = df.set_index(df.iloc[:, 0].astype(str) + " " + df.iloc[:, 1].astype(str))
    df = df.drop(df.columns[0], axis=1)
    df.columns = ["1", "2", "3", "4"]
    df.dropna(inplace=True)
    df = df.astype(int)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Perform hierarchical clustering
    linked = linkage(scaled_data, method="weighted", optimal_ordering=True)
    # Plot the dendrogram
    plt.figure(figsize=(8, 12))
    dendrogram(
        linked,
        orientation="right",
        distance_sort="descending",
        show_leaf_counts=False,
        labels=df.index.to_list(),
    )
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"./analysis/results/{title}")
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        df = pd.read_excel("Agriculture Data.xlsx", usecols="J:N", skiprows=5, nrows=26)
        cluster_analysis(df, "Agriculture Data Hierarchical Clustering Dendrogram")

        df = pd.read_excel(
            "Horticulture Data.xlsx", usecols="I:M", skiprows=11, nrows=26
        )
        cluster_analysis(df, "Horticulture Data Hierarchical Clustering Dendrogram")

    except FileNotFoundError:
        print("Error: Excel file not found. Please check the file path.")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise e
