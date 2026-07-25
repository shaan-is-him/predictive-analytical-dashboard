"""
Interactive visualization helpers using Plotly.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

COLORS = {
    "primary": "#6366F1",
    "secondary": "#8B5CF6",
    "success": "#10B981",
    "danger": "#EF4444",
    "warning": "#F59E0B",
    "info": "#3B82F6",
    "muted": "#6B7280",
}

PALETTE = ["#6366F1", "#8B5CF6", "#EC4899", "#F59E0B", "#10B981", "#3B82F6", "#EF4444", "#14B8A6"]


def style_fig(fig, title="", height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#E2E8F0"), x=0.02),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#CBD5E1"),
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.15)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.15)"),
    )
    return fig


def plot_class_distribution(y, target_names=None):
    counts = y.value_counts().sort_index()
    labels = [target_names[i] if target_names and i < len(target_names) else str(i) for i in counts.index]
    
    fig = go.Figure(data=[
        go.Bar(x=labels, y=counts.values, marker_color=PALETTE[:len(counts)],
               text=counts.values, textposition="outside")
    ])
    return style_fig(fig, "Target Class Distribution", height=360)


def plot_feature_distributions(X, n_cols=3, max_features=12):
    cols = X.columns[:max_features]
    n_rows = int(np.ceil(len(cols) / n_cols))
    
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[str(c) for c in cols])
    
    for i, col in enumerate(cols):
        r, c = divmod(i, n_cols)
        fig.add_trace(
            go.Histogram(x=X[col], nbinsx=30, marker_color=PALETTE[i % len(PALETTE)], showlegend=False),
            row=r+1, col=c+1
        )
    
    fig.update_layout(
        title=dict(text="Feature Distributions", font=dict(size=18, color="#E2E8F0"), x=0.02),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E1"), height=280 * n_rows, margin=dict(l=40, r=20, t=80, b=40)
    )
    return fig


def plot_correlation_heatmap(X, max_features=15):
    subset = X.iloc[:, :max_features]
    corr = subset.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu_r", zmid=0,
        text=np.round(corr.values, 2), texttemplate="%{text}", textfont=dict(size=9)
    ))
    return style_fig(fig, "Feature Correlation Matrix", height=520)


def plot_metrics_cards(metrics):
    names = list(metrics.keys())
    values = list(metrics.values())
    
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=PALETTE[:len(names)],
        text=[f"{v:.3f}" for v in values], textposition="outside"
    ))
    fig.update_layout(xaxis_range=[0, max(1.05, max(values) * 1.15)])
    return style_fig(fig, "Model Performance Metrics", height=280)


def plot_confusion_matrix(cm, labels=None):
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels, colorscale="Blues",
        text=cm, texttemplate="%{text}", textfont=dict(size=16, color="white"),
        showscale=True
    ))
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual", yaxis=dict(autorange="reversed"))
    return style_fig(fig, "Confusion Matrix", height=400)


def plot_roc_curve(fpr, tpr, auc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {auc:.3f})",
        line=dict(color=COLORS["primary"], width=3),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.15)"
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random",
        line=dict(color=COLORS["muted"], width=2, dash="dash")
    ))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return style_fig(fig, "ROC Curve", height=420)


def plot_feature_importance(importance_df, title="Feature Importance"):
    df = importance_df.sort_values("importance", ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df["importance"], y=df["feature"], orientation="h",
        marker=dict(color=df["importance"], colorscale="Viridis", showscale=False),
        text=[f"{v:.4f}" for v in df["importance"]], textposition="outside"
    ))
    fig.update_layout(xaxis_title="Importance", yaxis_title="")
    return style_fig(fig, title, height=max(360, 28 * len(df) + 80))


def plot_prediction_vs_actual(y_true, y_pred, task="regression"):
    if task == "regression":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_true, y=y_pred, mode="markers",
            marker=dict(color=COLORS["primary"], size=6, opacity=0.6), name="Predictions"
        ))
        min_v = min(y_true.min(), y_pred.min())
        max_v = max(y_true.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_v, max_v], y=[min_v, max_v], mode="lines",
            line=dict(color=COLORS["danger"], dash="dash"), name="Perfect Fit"
        ))
        fig.update_layout(xaxis_title="Actual", yaxis_title="Predicted")
        return style_fig(fig, "Predicted vs Actual Values", height=420)
    return plot_class_distribution(pd.Series(y_pred))


def plot_cv_scores(mean, std, metric_name="Accuracy"):
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="number+delta", value=mean,
        number=dict(valueformat=".3f", font=dict(size=48, color=COLORS["success"])),
        title=dict(text=f"5-Fold CV {metric_name} (±{std:.3f})", font=dict(size=16, color="#94A3B8")),
        domain=dict(x=[0, 1], y=[0, 1])
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig