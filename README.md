# NutriHelp Type 2 Diabetes Enhancement

## Overview

This repository contains an independent research prototype for a proposed Type 2 Diabetes enhancement of NutriHelp.

The project investigates machine-learning approaches for distinguishing between individuals in a Healthy/No Diabetes group and a Diabetes group using demographic, lifestyle and health-related predictors.

The project is maintained separately from the existing NutriHelp repositories.

## Binary Outcome Definition

For the current research prototype:

- Healthy / No Diabetes: HbA1c < 6.0%
- Diabetes: HbA1c > 6.5%
- HbA1c values from 6.0% to 6.5% are excluded from binary model training.

HbA1c is used to define the outcome category and is therefore deliberately excluded from the predictor variables to prevent target leakage.

The original binary diabetes field in the source dataset is also excluded from model predictors.

## Predictor Variables

The current models use:

- Gender
- Age
- Hypertension
- Heart disease
- Smoking history
- BMI
- Blood glucose level

## Machine-Learning Models

Three classification algorithms have been evaluated:

1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM)

### Initial Results

| Model | Accuracy | Balanced Accuracy | Sensitivity | Specificity | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.628 | 0.589 | 0.528 | 0.650 | 0.624 |
| Random Forest | 0.720 | 0.561 | 0.308 | 0.814 | 0.588 |
| SVM | 0.751 | 0.608 | 0.381 | 0.836 | 0.629 |

SVM produced the strongest overall discrimination among the three initial models, while Logistic Regression demonstrated higher sensitivity for the Diabetes group.

## SVM Threshold Analysis

A preliminary SVM threshold analysis was undertaken to examine the trade-off between sensitivity and specificity.

A decision threshold of -0.50 produced approximately:

- Accuracy: 0.686
- Balanced Accuracy: 0.602
- Sensitivity: 0.470
- Specificity: 0.735
- F1-score: 0.357

This threshold is experimental and represents a machine-learning decision threshold, not a clinical HbA1c diagnostic threshold.

## Project Structure

```text
NutriHelp-Type2Diabetes/
├── app/
├── data/
├── docs/
├── models/
├── notebooks/
├── src/
│   ├── train_model.py
│   ├── logistic_regression.py
│   ├── svm_model.py
│   ├── svm_threshold_analysis.py
│   └── predict.py
├── tests/
├── .gitignore
├── README.md
└── requirements.txt