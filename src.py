"""
Optimized Machine Learning Pipeline for Predicting Online Shoppers Behavior
===========================================================================
Author: Pavan Kumar (gujjaripavan222)
Dataset: Online Shoppers Intention (UCI ML Repository)
Goal: Classify e-commerce sessions as buyer (1) or non-buyer (0)

Pipeline Steps:
    1. Data loading & preprocessing
    2. Feature engineering
    3. Train/test split
    4. Model benchmarking with 10-fold cross-validation (ROC-AUC)
    5. Best model training & evaluation
"""

import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as IMBPipeline

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_URL = (
    "https://raw.githubusercontent.com/gujjaripavan222/"
    "Optimized-Machine-Learning-Pipeline-for-Predicting-Online-Shoppers-Behavior/"
    "refs/heads/main/online_shoppers_intention.csv"
)
TEST_SIZE = 0.3
RANDOM_STATE = 0
CV_FOLDS = 10
N_BEST_FEATURES = 6


# ---------------------------------------------------------------------------
# Step 1: Load Data
# ---------------------------------------------------------------------------
def load_data(url: str) -> pd.DataFrame:
    """Load the Online Shoppers Intention dataset from a URL."""
    print("\n[Step 1] Loading dataset...")
    df = pd.read_csv(url)
    print(f"  Shape: {df.shape}")
    print(df.head())
    return df


# ---------------------------------------------------------------------------
# Step 2: Feature Engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering:
    - Encode boolean columns (Weekend, Revenue) as 0/1
    - Create Returning_Visitor binary column from VisitorType
    - Ordinal-encode Month column
    """
    print("\n[Step 2] Engineering features...")

    # Encode boolean columns
    df["Weekend"] = df["Weekend"].astype(int)
    df["Revenue"] = df["Revenue"].astype(int)

    # Binary feature: is the visitor a returning visitor?
    df["Returning_Visitor"] = (df["VisitorType"] == "Returning_Visitor").astype(int)
    df.drop(columns=["VisitorType"], inplace=True)

    # Ordinal-encode the Month column
    ordinal_encoder = OrdinalEncoder()
    df["Month"] = ordinal_encoder.fit_transform(df[["Month"]])

    # Log top correlations with target
    correlations = df.corr()["Revenue"].drop("Revenue").sort_values(ascending=False)
    print("\n  Top feature correlations with Revenue:")
    print(correlations.head(8).to_string())

    return df


# ---------------------------------------------------------------------------
# Step 3: Train/Test Split
# ---------------------------------------------------------------------------
def split_data(df: pd.DataFrame):
    """Split the dataset into training and test sets (70/30)."""
    print("\n[Step 3] Splitting dataset (70% train / 30% test)...")
    X = df.drop(columns=["Revenue"])
    y = df["Revenue"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Step 4: Build Pipeline
# ---------------------------------------------------------------------------
def build_pipeline(X: pd.DataFrame, model) -> IMBPipeline:
    """
    Construct an IMBPipeline for a given classifier:
        ColumnTransformer → SMOTE → SelectKBest → model
    """
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant")),
        ("scaler", MinMaxScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ], remainder="passthrough")

    steps = [
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=1)),
        ("feature_selection", SelectKBest(score_func=chi2, k=N_BEST_FEATURES)),
        ("model", model),
    ]
    return IMBPipeline(steps=steps)


# ---------------------------------------------------------------------------
# Step 5: Benchmark All Models
# ---------------------------------------------------------------------------
def benchmark_models(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """
    Evaluate a suite of classifiers using 10-fold cross-validation (ROC-AUC).
    Returns a DataFrame of results sorted by ROC-AUC descending.
    """
    classifiers = {
        "RandomForestClassifier": RandomForestClassifier(),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "RidgeClassifier": RidgeClassifier(),
        "SVC": SVC(),
        "DummyClassifier": DummyClassifier(strategy="most_frequent"),
        "LGBMClassifier": LGBMClassifier(),
        "ExtraTreeClassifier": ExtraTreeClassifier(),
        "ExtraTreesClassifier": ExtraTreesClassifier(),
        "BernoulliNB": BernoulliNB(),
        "XGBClassifier": XGBClassifier(),
        "SGDClassifier": SGDClassifier(),
        "AdaBoostClassifier": AdaBoostClassifier(),
        "BaggingClassifier": BaggingClassifier(),
        "MLPClassifier": MLPClassifier(
            hidden_layer_sizes=(27, 50),
            max_iter=300,
            activation="relu",
            solver="adam",
            random_state=1,
        ),
    }

    print(f"\n[Step 4] Benchmarking {len(classifiers)} models ({CV_FOLDS}-fold CV)...")
    results = []

    for name, model in classifiers.items():
        start = time.time()
        pipeline = build_pipeline(X_train, model)
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=CV_FOLDS, scoring="roc_auc"
        )
        elapsed = round((time.time() - start) / 60, 2)
        results.append({
            "Model": name,
            "ROC-AUC (mean)": round(cv_scores.mean(), 4),
            "ROC-AUC (std)": round(cv_scores.std(), 4),
            "Run Time (min)": elapsed,
        })
        print(f"  ✔ {name:35s}  ROC-AUC={cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    df_results = pd.DataFrame(results).sort_values("ROC-AUC (mean)", ascending=False)
    return df_results


# ---------------------------------------------------------------------------
# Step 6: Train Best Model & Evaluate
# ---------------------------------------------------------------------------
def evaluate_best_model(X_train, X_test, y_train, y_test) -> None:
    """Train the best model (MLPClassifier) and print full evaluation metrics."""
    print("\n[Step 5] Training best model: MLPClassifier...")

    best_model = MLPClassifier(
        hidden_layer_sizes=(27, 50),
        max_iter=300,
        activation="relu",
        solver="adam",
        random_state=1,
    )
    pipeline = build_pipeline(X_train, best_model)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc:.4f}")
    print("\n  Confusion Matrix:")
    print(cm)
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-Buyer", "Buyer"]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Online Shoppers Behavior — ML Pipeline")
    print("=" * 60)

    df = load_data(DATASET_URL)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    model_rankings = benchmark_models(X_train, y_train)
    print("\n[Results] Model Rankings by ROC-AUC:")
    print(model_rankings.to_string(index=False))

    evaluate_best_model(X_train, X_test, y_train, y_test)

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
