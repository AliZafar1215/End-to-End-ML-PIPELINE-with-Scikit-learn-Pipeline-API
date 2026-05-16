# Telco Customer Churn Prediction: End-to-End ML Pipeline

This repository contains an end-to-end production-ready Machine Learning pipeline built using Scikit-learn to predict customer churn using the IBM Telco dataset. The pipeline seamlessly integrates missing value imputation, feature scaling, one-hot encoding, model training, hyperparameter tuning, and model serialization.

---

## 1. Project Overview

The goal of this project is to build a robust binary classification model that accurately identifies customers at risk of churning. By utilizing Scikit-learn's `Pipeline` and `ColumnTransformer` APIs, the project prevents data leakage during training/validation and ensures the final model artifact is fully self-contained and ready for production deployment.

### Key Highlights

* **Zero Data Leakage:** Feature scaling and categorical encoding parameters are learned strictly from training folds during Cross-Validation.
* **Automated Preprocessing:** Handles numerical missing values and categorical encoding simultaneously in a single executable graph.
* **Hyperparameter Optimization:** Employs `GridSearchCV` paired with `StratifiedKFold` to balance optimization against an imbalanced target distribution (~26% churn rate).
* **Production Ready:** Exports the full preprocessing + model architecture into a single `.joblib` file.

---

## 2. Tech Stack & Dependencies

* **Language:** Python 3.10+
* **Core Libraries:** Data Manipulation (Pandas, NumPy), Machine Learning (Scikit-learn), Model Serialization (Joblib), and Visualization (Matplotlib, Seaborn).

---

## 3. Repository Structure

1. Core Jupyter Notebook containing the full development lifecycle, data cleaning steps, and model evaluation visualizations.
2. The final serialized production-ready pipeline artifact saved as a `.joblib` file.
3. Project documentation (README.md).

---

## 4. Pipeline Architecture

The end-to-end architecture isolates data preprocessing by feature type before feeding the transformed data directly into the optimized classifier:

### 4.1. Numerical Features

The features `tenure`, `MonthlyCharges`, and `TotalCharges` undergo a two-step transformation:

1. Median Imputation to handle any missing values cleanly.
2. Standardization using a standard scaler to normalize feature scales.

### 4.2. Categorical Features

All remaining categorical features are processed in parallel:

1. Most Frequent Imputation to fill in missing entries with the most common class.
2. One-Hot Encoding to convert categories into binary flags while ignoring unknown categories during inference.

### 4.3. Model Classifier

1. Baseline Testing: Evaluating performance across both Logistic Regression and Random Forest architectures.
2. Final Model: Selecting the Random Forest Classifier and tuning it via Grid Search.

---

## 5. Model Evaluation & Results

The baseline execution compared both architectures, leading to a focused grid optimization on the Random Forest pipeline.

### 5.1. Final Metrics

* **Accuracy:** Evaluated dynamically upon notebook execution.
* **F1-Score:** Optimized specifically for the minority 'Churned' class to ensure reliable predictions.
* **ROC-AUC:** Used to determine the overall discriminative strength of the classifier.

### 5.2. Top Predictors

The feature importance extraction highlights that a customer's tenure length, monthly charge amount, and contract type act as the strongest signals for identifying potential churn.

---

## 6. Usage & Inference Workflow

The final trained pipeline is saved as a single, unified object containing all preprocessing steps alongside the optimized model parameters.

To use this file in production:

1. Load the exported pipeline artifact directly into your environment using Joblib.
2. Pass completely raw customer data directly to the pipeline object. The input data must retain the original raw column structure used during training.
3. Extract final predictions or target probabilities instantly. The pipeline handles all imputation, scaling, and encoding under the hood without requiring manual data preparation.

---

## 7. Insights & Takeaways

1. **Pipeline API Utility:** Writing transformers inside Scikit-learn pipelines guarantees absolute consistency between training and production environments, eliminating deployment bugs.
2. **Imbalance Handling:** Using Stratified K-Fold splits ensures that small minority classes (like churned users) are fairly distributed across training validation slices, preventing biased evaluations.
