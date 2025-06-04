import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix


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
df = pd.DataFrame()
for label, curr_df in sheets.items():
    df = pd.concat([df, curr_df], axis=1)

df.dropna(inplace=True)

# Create dummy target (replace this with actual label if available)
np.random.seed(42)
df["Label"] = np.random.randint(0, 2, size=len(df))

X = df.drop("Label", axis=1)
y = df["Label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "CART": DecisionTreeClassifier(criterion="gini", max_depth=3),
    "J48": DecisionTreeClassifier(criterion="entropy", max_depth=3),  # Simulated J48
    "Random Forest": RandomForestClassifier(n_estimators=10, max_depth=3),
}


def evaluate(model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    f1 = f1_score(y_true, y_pred)
    return {
        "Model": model_name,
        "Accuracy": f"{acc*100:.2f}%",
        "Sensitivity": f"{sensitivity*100:.2f}%",
        "Specificity": f"{specificity*100:.2f}%",
        "F1-Score": f"{f1*100:.2f}%",
    }


# Run all models
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results.append(evaluate(name, y_test, preds))

# Display
results_df = pd.DataFrame(results)
print(results_df)
