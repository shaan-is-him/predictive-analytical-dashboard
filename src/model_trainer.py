"""
Model training, evaluation, and feature importance utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_curve, precision_recall_curve,
)
from sklearn.model_selection import cross_val_score
import joblib
import os

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def get_model(model_name: str, task: str = "classification", **params):
    """Factory for model instances. Only relevant hyperparameters are applied."""
    
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.linear_model import LinearRegression

    tree_keys = {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "learning_rate", "subsample"}
    linear_keys = {"C", "alpha", "penalty", "solver"}
    knn_keys = {"n_neighbors", "weights", "p"}
    svm_keys = {"C", "kernel", "gamma"}
    dt_keys = {"max_depth", "min_samples_split", "min_samples_leaf", "criterion"}

    def filter_params(allowed):
        return {k: v for k, v in params.items() if k in allowed}

    if task == "classification":
        if model_name == "Random Forest":
            return RandomForestClassifier(random_state=42, n_jobs=-1, **filter_params(tree_keys))
        if model_name == "Gradient Boosting":
            return GradientBoostingClassifier(random_state=42, **filter_params(tree_keys))
        if model_name == "Logistic Regression":
            return LogisticRegression(random_state=42, max_iter=1000, **filter_params(linear_keys))
        if model_name == "Decision Tree":
            return DecisionTreeClassifier(random_state=42, **filter_params(dt_keys))
        if model_name == "SVM":
            return SVC(random_state=42, probability=True, **filter_params(svm_keys))
        if model_name == "KNN":
            return KNeighborsClassifier(n_jobs=-1, **filter_params(knn_keys))
        if model_name == "XGBoost" and HAS_XGB:
            return XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1, **filter_params(tree_keys))
    
    else:  # regression
        if model_name == "Random Forest":
            return RandomForestRegressor(random_state=42, n_jobs=-1, **filter_params(tree_keys))
        if model_name == "Gradient Boosting":
            return GradientBoostingRegressor(random_state=42, **filter_params(tree_keys))
        if model_name == "Ridge Regression":
            return Ridge(random_state=42, **filter_params(linear_keys))
        if model_name == "Linear Regression":
            return LinearRegression(**filter_params(set()))  # no special params
        if model_name == "Decision Tree":
            return DecisionTreeRegressor(random_state=42, **filter_params(dt_keys))
        if model_name == "SVM":
            return SVR(**filter_params(svm_keys))
        if model_name == "KNN":
            return KNeighborsRegressor(n_jobs=-1, **filter_params(knn_keys))
        if model_name == "XGBoost" and HAS_XGB:
            return XGBRegressor(random_state=42, n_jobs=-1, **filter_params(tree_keys))

    raise ValueError(f"Unknown model: {model_name} for task={task}")

def train_model(model, X_train, y_train, X_test, y_test, task="classification"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results = {"model": model, "y_pred": y_pred, "task": task}
    
    if task == "classification":
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            results["y_proba"] = y_proba
        
        results["metrics"] = {
            "Accuracy": float(accuracy_score(y_test, y_pred)),
            "Precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "Recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "F1 Score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        }
        
        if y_proba is not None and y_test.nunique() == 2:
            results["metrics"]["ROC AUC"] = float(roc_auc_score(y_test, y_proba[:, 1]))
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            results["roc_curve"] = {"fpr": fpr, "tpr": tpr}
            prec, rec, _ = precision_recall_curve(y_test, y_proba[:, 1])
            results["pr_curve"] = {"precision": prec, "recall": rec}
        
        results["confusion_matrix"] = confusion_matrix(y_test, y_pred)
        results["classification_report"] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
        results["cv_accuracy_mean"] = float(cv_scores.mean())
        results["cv_accuracy_std"] = float(cv_scores.std())
        
    else:
        results["metrics"] = {
            "R² Score": float(r2_score(y_test, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "MSE": float(mean_squared_error(y_test, y_pred)),
        }
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
        results["cv_r2_mean"] = float(cv_scores.mean())
        results["cv_r2_std"] = float(cv_scores.std())
    
    return results


def get_feature_importance(model, feature_names, top_n=20):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            importances = np.abs(coef).mean(axis=0)
        else:
            importances = np.abs(coef)
    else:
        return pd.DataFrame({"feature": feature_names, "importance": [0.0] * len(feature_names)})
    
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    
    return df


def save_model(model, path, metadata=None):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata or {}}, path)