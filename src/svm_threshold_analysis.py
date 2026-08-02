from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "diabetes_prediction_dataset.csv"
)


def create_binary_status(hba1c):

    if hba1c < 6.0:
        return 0

    elif hba1c > 6.5:
        return 1

    else:
        return None


def main():

    df = pd.read_csv(DATA_PATH)

    df["diabetes_status"] = (
        df["HbA1c_level"]
        .apply(create_binary_status)
    )

    df = df.dropna(
        subset=["diabetes_status"]
    ).copy()

    df["diabetes_status"] = (
        df["diabetes_status"].astype(int)
    )

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

    pipeline.fit(
        X_train,
        y_train,
    )

    decision_scores = (
        pipeline.decision_function(X_test)
    )

    roc_auc = roc_auc_score(
        y_test,
        decision_scores,
    )

    print("\nROC-AUC:")
    print(round(roc_auc, 3))

    thresholds = [
        -1.0,
        -0.75,
        -0.50,
        -0.25,
        0.0,
        0.25,
        0.50,
    ]

    print("\nThreshold Analysis")
    print(
        "Threshold | Accuracy | Balanced Acc | "
        "Sensitivity | Specificity | F1"
    )

    print("-" * 75)

    for threshold in thresholds:

        predictions = (
            decision_scores >= threshold
        ).astype(int)

        tn, fp, fn, tp = (
            confusion_matrix(
                y_test,
                predictions,
            ).ravel()
        )

        sensitivity = (
            tp / (tp + fn)
        )

        specificity = (
            tn / (tn + fp)
        )

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

        print(
            f"{threshold:8.2f} | "
            f"{accuracy:8.3f} | "
            f"{balanced_accuracy:12.3f} | "
            f"{sensitivity:11.3f} | "
            f"{specificity:11.3f} | "
            f"{f1:5.3f}"
        )


if __name__ == "__main__":
    main()