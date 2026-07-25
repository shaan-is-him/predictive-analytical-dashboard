"""
Data loading and preprocessing utilities for the Predictive Analytics Dashboard.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any, Optional


def load_dataset(dataset_name: str = "breast_cancer") -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    meta = {}
    
    if dataset_name == "breast_cancer":
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
        meta = {
            "name": "Breast Cancer Wisconsin",
            "task": "classification",
            "target_names": list(data.target_names),
            "description": "Predict whether a breast mass is malignant or benign based on cell nuclei features.",
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "class_balance": y.value_counts().to_dict(),
        }
    elif dataset_name == "wine":
        data = load_wine()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
        meta = {
            "name": "Wine Recognition",
            "task": "classification",
            "target_names": list(data.target_names),
            "description": "Classify wines into three cultivars based on chemical analysis.",
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "class_balance": y.value_counts().to_dict(),
        }
    elif dataset_name == "california_housing":
        data = fetch_california_housing()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="MedHouseVal")
        meta = {
            "name": "California Housing",
            "task": "regression",
            "target_names": ["Median House Value"],
            "description": "Predict median house value in California districts from census data.",
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "target_stats": {
                "mean": float(y.mean()),
                "std": float(y.std()),
                "min": float(y.min()),
                "max": float(y.max()),
            },
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return X, y, meta


def prepare_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
) -> Dict[str, Any]:
    stratify = y if (y.nunique() < 20 and y.nunique() > 1) else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X.columns, index=X_test.index
        )
    else:
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": list(X.columns),
    }


def generate_synthetic_churn(n_samples: int = 2000, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    
    tenure = rng.integers(1, 73, n_samples)
    monthly_charges = rng.normal(65, 30, n_samples).clip(20, 120)
    total_charges = monthly_charges * tenure * rng.uniform(0.8, 1.2, n_samples)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n_samples, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n_samples, p=[0.35, 0.40, 0.25])
    support = rng.choice(["Yes", "No"], n_samples, p=[0.3, 0.7])
    senior = rng.choice([0, 1], n_samples, p=[0.85, 0.15])
    partners = rng.choice([0, 1], n_samples, p=[0.5, 0.5])
    dependents = rng.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    churn_prob = (
        0.35 * (contract == "Month-to-month")
        + 0.15 * (internet == "Fiber optic")
        + 0.10 * (support == "No")
        + 0.08 * senior
        - 0.12 * (tenure > 24)
        - 0.08 * (partners == 1)
        + 0.05 * (monthly_charges > 80)
        + rng.normal(0, 0.08, n_samples)
    )
    churn_prob = np.clip(churn_prob, 0.05, 0.85)
    churn = (rng.random(n_samples) < churn_prob).astype(int)
    
    df = pd.DataFrame({
        "tenure": tenure,
        "MonthlyCharges": monthly_charges.round(2),
        "TotalCharges": total_charges.round(2),
        "Contract": contract,
        "InternetService": internet,
        "TechSupport": support,
        "SeniorCitizen": senior,
        "Partner": partners,
        "Dependents": dependents,
        "Churn": churn,
    })
    
    le_contract = LabelEncoder()
    le_internet = LabelEncoder()
    le_support = LabelEncoder()
    
    X = df.drop(columns=["Churn"]).copy()
    X["Contract"] = le_contract.fit_transform(X["Contract"])
    X["InternetService"] = le_internet.fit_transform(X["InternetService"])
    X["TechSupport"] = le_support.fit_transform(X["TechSupport"])
    
    y = df["Churn"]
    
    meta = {
        "name": "Synthetic Customer Churn",
        "task": "classification",
        "target_names": ["No Churn", "Churn"],
        "description": "Predict customer churn based on tenure, charges, contract type, and service features.",
        "n_samples": n_samples,
        "n_features": X.shape[1],
        "class_balance": y.value_counts().to_dict(),
    }
    
    return X, y, meta


def process_uploaded_data(
    df: pd.DataFrame,
    target_column: str,
    task: str = "auto"
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Process a user-uploaded DataFrame.
    - Drops rows with missing target
    - Fills numeric missing values with median
    - Label-encodes categorical columns
    - Auto-detects classification vs regression if task="auto"
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")
    
    # Drop rows where target is missing
    df = df.dropna(subset=[target_column]).copy()
    
    y = df[target_column].copy()
    X = df.drop(columns=[target_column]).copy()
    
    # Handle missing values in features
    for col in X.columns:
        if X[col].dtype in ["float64", "int64", "float32", "int32"]:
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Unknown")
    
    # Encode categorical columns
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Auto-detect task
    if task == "auto":
        n_unique = y.nunique()
        if n_unique <= 15 and (pd.api.types.is_integer_dtype(y) or n_unique <= 10):
            task = "classification"
        else:
            task = "regression"
    
    # For classification: encode target if needed
    target_names = None
    if task == "classification":
        if y.dtype == "object" or y.dtype.name == "category":
            le_target = LabelEncoder()
            y = pd.Series(le_target.fit_transform(y.astype(str)), name=target_column)
            target_names = list(le_target.classes_)
        else:
            target_names = [str(c) for c in sorted(y.unique())]
    
    meta = {
        "name": "Custom Uploaded Dataset",
        "task": task,
        "target_names": target_names,
        "description": f"User-uploaded dataset. Target column: '{target_column}'. Automatically processed.",
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
    }
    
    if task == "classification":
        meta["class_balance"] = y.value_counts().to_dict()
    else:
        meta["target_stats"] = {
            "mean": float(y.mean()),
            "std": float(y.std()),
            "min": float(y.min()),
            "max": float(y.max()),
        }
    
    return X, y, meta