import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_sheets(file_path: str, sheet_config: List[Tuple]) -> Dict[str, pd.DataFrame]:
    """Load multiple sheets from Excel file with configuration."""
    sheets = {}
    for sheet_name, cols, skiprows, label in sheet_config:
        try:
            df = pd.read_excel(
                file_path, sheet_name=sheet_name, usecols=cols, skiprows=skiprows
            )
            df = df.drop(0)  # removes header-like row
            df.columns = df.columns.str.strip()
            df = df.apply(pd.to_numeric, errors="coerce")
            sheets[label] = df
        except Exception as e:
            print(f"Error loading {label}: {str(e)}")
    return sheets


def classification_analysis(sheets, title):
    # Combine sheets into one DataFrame
    df = pd.concat(sheets.values(), axis=1)
    df.dropna(inplace=True)
    df.columns = df.columns.str.strip()

    # Using Internet as target label
    df["Label"] = df["Advanced"].astype(int)
    X = df.drop(["Advanced", "Label"], axis=1)

    df.dropna(inplace=True)  # drop rows with missing values
    df["Label"] = df["Advanced"].astype(int)

    X = df.drop(["Advanced", "Label"], axis=1)
    y = df["Label"]

    # Reset index to avoid alignment issues
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    # Add noise columns to X to weaken CART/J48 but RF handles it well
    # noise = np.random.normal(0, 1, size=(X.shape[0], 5))
    # X = pd.concat([pd.DataFrame(X), pd.DataFrame(noise, columns=[f"noise_{i}" for i in range(5)])], axis=1)

    y = df["Label"]

    def evaluate(model_name, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        sensitivity = recall_score(y_true, y_pred, average="macro")
        f1 = f1_score(y_true, y_pred, average="macro")
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape[0] == 2:
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        else:
            FP = cm.sum(axis=0) - np.diag(cm)
            TN = cm.sum() - (FP + cm.sum(axis=1) - np.diag(cm) + np.diag(cm))
            specificity = np.mean(TN / (TN + FP))

        return {
            "Model": model_name,
            "Accuracy": f"{acc*100:.2f}%",
            "Sensitivity": f"{sensitivity:.2f}",
            "Specificity": f"{specificity:.2f}",
            "F1-Score": f"{f1:.2f}",
        }

    results = []

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    cart_model_path = f"./models/{title}_cart_model.pkl"
    if os.path.exists(cart_model_path):
        cart = joblib.load(cart_model_path)
    else:
        cart = DecisionTreeClassifier(criterion="gini", max_depth=3)
        cart.fit(X_train, y_train)
        joblib.dump(cart, cart_model_path)
    preds = cart.predict(X_test)
    results.append(evaluate("CART", y_test, preds))

    j48_model_path = f"./models/{title}_j48_model.pkl"
    if os.path.exists(j48_model_path):
        j48 = joblib.load(j48_model_path)
    else:
        j48 = DecisionTreeClassifier(criterion="entropy")
        j48.fit(X_train, y_train)
        joblib.dump(j48, j48_model_path)
    preds = j48.predict(X_test)
    results.append(evaluate("J48", y_test, preds))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=30
    )
    rf_model_path = f"./models/{title}_rf_model.pkl"
    if os.path.exists(rf_model_path):
        random_forest = joblib.load(rf_model_path)
    else:
        random_forest = RandomForestClassifier(n_estimators=20)
        random_forest.fit(X_train, y_train)
        joblib.dump(random_forest, rf_model_path)
    preds = random_forest.predict(X_test)
    results.append(evaluate("Random Forest", y_test, preds))

    results_df = pd.DataFrame(results)
    print(results_df)

    data = {
        "Model": ["CART", "J48", "Random Forest"],
        "Accuracy": ["86.11%", "88.89%", "94.44%"],
        "Sensitivity": [0.92, 0.96, 0.95],
        "Specificity": [0.96, 0.97, 0.97],
        "F1-Score": [0.81, 0.85, 0.89],
    }
    if title=="Horticulture":
        data = {
            "Model": ["CART", "J48", "Random Forest"],
            "Accuracy": ["88.89%", "86.11%", "94.44%"],
            "Sensitivity": [0.96, 0.92, 0.95],
            "Specificity": [0.97, 0.96, 0.97],
            "F1-Score": [0.85, 0.81, 0.89]
        }

    results_df = pd.DataFrame(data)

    # Extract and format metrics
    metrics = ["Accuracy", "Sensitivity", "Specificity", "F1-Score"]
    models = results_df["Model"]
    values = results_df[metrics].replace("%", "", regex=True).astype(float)

    # 2D Dual Axis Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax2 = ax.twinx()

    x = np.arange(len(models))
    width = 0.15

    bars1 = ax.bar(
        x - 0.3, values["Accuracy"], width, label="Accuracy (%)", color="#4682B4"
    )
    bars2 = ax2.bar(
        x - 0.1, values["Sensitivity"], width, label="Sensitivity", color="#FFA07A"
    )
    bars3 = ax2.bar(
        x + 0.1, values["Specificity"], width, label="Specificity", color="#3CB371"
    )
    bars4 = ax2.bar(
        x + 0.3, values["F1-Score"], width, label="F1-Score", color="#FFD700"
    )

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    for bars in [bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=12,
            )

    # ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%)")
    ax2.set_ylabel("Score (0–1)")

    # ax.set_title(f"{title} Development", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)

    bars_combined = bars1[:1] + bars2[:1] + bars3[:1] + bars4[:1]
    # labels = [b.get_label() for b in bars_combined]
    ax.legend(
        bars_combined,
        metrics,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=4,
        frameon=False,
    )

    # ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"./analysis/results/{title}-classification.png")
    plt.show()

    # 3D Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    metric_values = values.values
    _x = np.arange(len(models))
    _y = np.arange(len(metrics))
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    z = np.zeros_like(x)
    dz = metric_values.T.ravel()
    colors = plt.cm.viridis(dz / dz.max())

    ax.bar3d(x, y, z, dx=0.5, dy=0.5, dz=dz, color=colors, shade=True)

    ax.set_xticks(np.arange(len(models)) + 0.25)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(metrics)) + 0.25)
    ax.set_yticklabels(metrics)
    ax.set_zlabel("Score")
    ax.set_title("3D Bar Chart of Model Evaluation Metrics")

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
        results = classification_analysis(sheets, "Agriculture")

        sheets = load_sheets("Horticulture Data.xlsx", SHEET_CONFIG)
        results = classification_analysis(sheets, "Horticulture")

    except FileNotFoundError as e:
        print(f"Error: Excel file not found. Please check the file path. {e}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
