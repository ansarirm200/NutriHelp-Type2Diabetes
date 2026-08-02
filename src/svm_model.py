from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


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
    / "svm_binary_diabetes.joblib"
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
# Train SVM
# ---------------------------------------------------------

def train_svm():

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print("\nOriginal dataset size:")
    print(len(df))

    # Create binary outcome
    df["diabetes_status"] = (
        df["HbA1c_level"]
        .apply(create_binary_status)
    )

    # Remove intermediate HbA1c cases
    df = df.dropna(
        subset=["diabetes_status"]
    ).copy()

    df["diabetes_status"] = (
        df["diabetes_status"].astype(int)
    )

    print(
        "\nDataset size after excluding HbA1c 6.0-6.5:"
    )
    print(len(df))

    # -----------------------------------------------------
    # Predictor variables
    # -----------------------------------------------------

    # HbA1c is excluded because it defines the outcome.
    # The original diabetes field is also excluded
    # to prevent target leakage.

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
                StandardScaler(),
                numeric_features,
            ),
        ]
    )

    # -----------------------------------------------------
    # SVM model
    # -----------------------------------------------------

    # probability=False makes training much faster.
    # ROC-AUC will be calculated using decision_function.

    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=False,
        random_state=42,
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

    print("\nTraining SVM...")

    # Train model
    pipeline.fit(
        X_train,
        y_train,
    )

    print("SVM training completed.")

    # -----------------------------------------------------
    # Predictions
    # -----------------------------------------------------

    predictions = pipeline.predict(
        X_test
    )

    # Use decision scores instead of probabilities
    decision_scores = (
        pipeline.decision_function(X_test)
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
        decision_scores,
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
    # Print results
    # -----------------------------------------------------

    print(
        "\nNutriHelp Type 2 Diabetes Enhancement"
    )

    print(
        "Binary SVM Classification Model"
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
    train_svm()