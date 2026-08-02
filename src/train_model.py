from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "diabetes_prediction_dataset.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest_binary_diabetes.joblib"
)


# ---------------------------------------------------------
# Create binary outcome
# ---------------------------------------------------------

def create_binary_status(hba1c):
    """
    NutriHelp binary modelling definition:

    Healthy / No Diabetes:
        HbA1c < 6.0

    Diabetes:
        HbA1c > 6.5

    Intermediate range:
        HbA1c 6.0 to 6.5
        Excluded from AI model training.
    """

    if hba1c < 6.0:
        return 0

    elif hba1c > 6.5:
        return 1

    else:
        return None


# ---------------------------------------------------------
# Train Random Forest
# ---------------------------------------------------------

def train_random_forest():

    df = pd.read_csv(DATA_PATH)

    print("\nOriginal dataset size:")
    print(len(df))

    # Create binary outcome
    df["diabetes_status"] = (
        df["HbA1c_level"]
        .apply(create_binary_status)
    )

    # Exclude HbA1c 6.0–6.5
    df = df.dropna(
        subset=["diabetes_status"]
    ).copy()

    df["diabetes_status"] = (
        df["diabetes_status"].astype(int)
    )

    print("\nDataset size after excluding HbA1c 6.0-6.5:")
    print(len(df))

    # -----------------------------------------------------
    # Predictor variables
    # -----------------------------------------------------

    # HbA1c is excluded because it defines the target.
    # The original diabetes variable is also excluded
    # to avoid target leakage.

    feature_columns = [
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "smoking_history",
        "bmi",
        "blood_glucose_level",
    ]

    X = df[feature_columns]
    y = df["diabetes_status"]

    categorical_features = [
        "gender",
        "smoking_history",
    ]

    numeric_features = [
        "age",
        "hypertension",
        "heart_disease",
        "bmi",
        "blood_glucose_level",
    ]

    # -----------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features,
            ),
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
        ]
    )

    # -----------------------------------------------------
    # Random Forest
    # -----------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # -----------------------------------------------------
    # Train/test split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    # Train
    pipeline.fit(
        X_train,
        y_train,
    )

    # Predict
    predictions = pipeline.predict(
        X_test
    )

    probabilities = (
        pipeline.predict_proba(X_test)[:, 1]
    )

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            predictions,
        )
    )

    f1 = f1_score(
        y_test,
        predictions,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
            predictions,
        ).ravel()
    )

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print(
        "\nNutriHelp Type 2 Diabetes Enhancement"
    )

    print(
        "Binary Random Forest Model"
    )

    print(
        "------------------------------------"
    )

    print(
        f"Accuracy:          {accuracy:.3f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy:.3f}"
    )

    print(
        f"F1-score:          {f1:.3f}"
    )

    print(
        f"ROC-AUC:           {roc_auc:.3f}"
    )

    print(
        f"Sensitivity:       {sensitivity:.3f}"
    )

    print(
        f"Specificity:       {specificity:.3f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Healthy/No Diabetes",
                "Diabetes",
            ],
        )
    )

    print(
        "\nClass Distribution:"
    )

    print(
        y.value_counts()
    )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "pipeline": pipeline,
            "features": feature_columns,
            "classes": [
                "Healthy/No Diabetes",
                "Diabetes",
            ],
            "healthy_threshold": "< 6.0",
            "diabetes_threshold": "> 6.5",
            "excluded_range": "6.0-6.5",
        },
        MODEL_PATH,
    )

    print(
        "\nModel saved successfully to:"
    )

    print(
        MODEL_PATH
    )


if __name__ == "__main__":
    train_random_forest()