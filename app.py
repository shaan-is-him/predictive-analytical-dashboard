import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import (
    load_dataset,
    prepare_data,
    generate_synthetic_churn,
    process_uploaded_data,
)
from src.model_trainer import get_model, train_model, get_feature_importance, save_model
from src.visualizer import (
    plot_class_distribution,
    plot_feature_distributions,
    plot_correlation_heatmap,
    plot_metrics_cards,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_prediction_vs_actual,
    plot_cv_scores,
)
from src.reporter import generate_pdf_report


st.set_page_config(
    page_title="Predictive Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        background: #0A4D3C;
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .main-header p { margin: 0.4rem 0 0; opacity: 0.9; }
    .metric-card {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        text-align: center;
    }
    .metric-card .label { font-size: 0.8rem; color: #94A3B8; text-transform: uppercase; }
    .metric-card .value { font-size: 1.75rem; font-weight: 700; color: #E2E8F0; }
    .success-box {
        background: rgba(16, 185, 129, 0.12);
        border-left: 4px solid #10B981;
        padding: 0.9rem 1.1rem;
        border-radius: 0 8px 8px 0;
    }
    .info-box {
        background: rgba(99, 102, 241, 0.12);
        border-left: 4px solid #6366F1;
        padding: 0.9rem 1.1rem;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

if "trained" not in st.session_state:
    st.session_state.trained = False
if "results" not in st.session_state:
    st.session_state.results = None
if "data_bundle" not in st.session_state:
    st.session_state.data_bundle = None
if "meta" not in st.session_state:
    st.session_state.meta = None
if "X" not in st.session_state:
    st.session_state.X = None
if "y" not in st.session_state:
    st.session_state.y = None


st.markdown("""
<div class="main-header">
    <h1>Predictive Analytics Dashboard</h1>
    <p>Interactive visualization · Real-time model tracking · Feature importance · Automated reporting</p>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### Configuration")

    data_source = st.radio(
        "Data Source",
        ["Built-in Dataset", "Upload Your Own CSV"],
        index=0
    )

    uploaded_file = None
    target_column = None
    custom_task = "auto"
    dataset_choice = None

    if data_source == "Built-in Dataset":
        dataset_choice = st.selectbox(
            "Dataset",
            ["Synthetic Customer Churn", "Breast Cancer", "Wine", "California Housing"],
            index=0
        )
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                preview_df = pd.read_csv(uploaded_file)
                st.success(f"Loaded: {preview_df.shape[0]} rows × {preview_df.shape[1]} columns")
                uploaded_file.seek(0)  # reset pointer

                target_column = st.selectbox(
                    "Select Target Column",
                    options=preview_df.columns.tolist(),
                    index=len(preview_df.columns) - 1
                )

                custom_task = st.radio(
                    "Task Type",
                    ["auto", "classification", "regression"],
                    index=0,
                    help="Auto decides based on number of unique values in the target."
                )

                with st.expander("Preview first 5 rows"):
                    st.dataframe(preview_df.head())
            except Exception as e:
                st.error(f"Error reading file: {e}")
                uploaded_file = None

    st.markdown("---")
    st.markdown("###  Model Settings")


    # All available models
    all_models = [
        "Random Forest",
        "Gradient Boosting",
        "Decision Tree",
        "SVM",
        "KNN",
        "Logistic Regression",
        "Linear Regression",
        "Ridge Regression",
        "XGBoost"
    ]
    model_name = st.selectbox("Algorithm", all_models, index=0)

    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    scale_features = st.checkbox("Standardize Features", value=True)

    st.markdown("#### Hyperparameters")
    params = {}

    if model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
        params["n_estimators"] = st.slider("n_estimators", 50, 400, 150, 25)
        params["max_depth"] = st.slider("max_depth", 2, 20, 8)
        if model_name == "XGBoost":
            params["learning_rate"] = st.slider("learning_rate", 0.01, 0.3, 0.1, 0.01)

    elif model_name == "Decision Tree":
        params["max_depth"] = st.slider("max_depth", 2, 20, 8)
        params["min_samples_split"] = st.slider("min_samples_split", 2, 20, 2)

    elif model_name == "KNN":
        params["n_neighbors"] = st.slider("n_neighbors", 1, 30, 5)
        params["weights"] = st.selectbox("weights", ["uniform", "distance"])

    elif model_name == "SVM":
        params["C"] = st.slider("C", 0.1, 10.0, 1.0, 0.1)
        params["kernel"] = st.selectbox("kernel", ["rbf", "linear", "poly"])

    elif model_name == "Logistic Regression":
        params["C"] = st.slider("C (regularization)", 0.01, 10.0, 1.0, 0.01)

    elif model_name == "Ridge Regression":
        params["alpha"] = st.slider("alpha", 0.01, 10.0, 1.0, 0.01)


    st.markdown("---")
    train_btn = st.button("Train Model", type="primary", use_container_width=True)


@st.cache_data(show_spinner="Loading dataset...")
def get_builtin_data(choice: str):
    if choice == "Synthetic Customer Churn":
        return generate_synthetic_churn(n_samples=2500)
    mapping = {
        "Breast Cancer": "breast_cancer",
        "Wine": "wine",
        "California Housing": "california_housing",
    }
    return load_dataset(mapping[choice])


X, y, meta = None, None, None

if data_source == "Built-in Dataset":
    X, y, meta = get_builtin_data(dataset_choice)
else:
    if uploaded_file is not None and target_column is not None:
        try:
            df = pd.read_csv(uploaded_file)
            X, y, meta = process_uploaded_data(df, target_column, task=custom_task)
            st.sidebar.success(
                f"Ready → {meta['n_samples']} samples | {meta['n_features']} features | {meta['task']}"
            )
        except Exception as e:
            st.sidebar.error(f"Processing error: {e}")
            st.stop()
    else:
        st.info("Please upload a CSV file and select the target column in the sidebar.")
        st.stop()

st.session_state.X = X
st.session_state.y = y
st.session_state.meta = meta


if meta["task"] == "classification":
    valid_models = [
        "Random Forest", "Gradient Boosting", "Decision Tree",
        "SVM", "KNN", "Logistic Regression", "XGBoost"
    ]
else:
    valid_models = [
        "Random Forest", "Gradient Boosting", "Decision Tree",
        "SVM", "KNN", "Linear Regression", "Ridge Regression", "XGBoost"
    ]

if model_name not in valid_models:
    model_name = valid_models[0]
    st.sidebar.warning(f"Switched to **{model_name}** (compatible with {meta['task']})")

# model training
if train_btn:
    with st.spinner("Training model & computing metrics..."):
        bundle = prepare_data(X, y, test_size=test_size, scale=scale_features)

        model = get_model(model_name, task=meta["task"], **params)

        X_tr = bundle["X_train_scaled"] if scale_features else bundle["X_train"]
        X_te = bundle["X_test_scaled"] if scale_features else bundle["X_test"]

        results = train_model(
            model, X_tr, bundle["y_train"], X_te, bundle["y_test"], task=meta["task"]
        )

        importance_df = get_feature_importance(model, bundle["feature_names"])
        results["importance"] = importance_df
        results["model_name"] = model_name
        results["params"] = params

        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join(
            "models",
            f"{model_name.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        )
        save_model(model, model_path, metadata={"dataset": meta["name"], "metrics": results["metrics"]})

        st.session_state.trained = True
        st.session_state.results = results
        st.session_state.data_bundle = bundle

    st.success(f"{model_name} trained successfully!")


tab_overview, tab_performance, tab_importance, tab_predict, tab_report = st.tabs([
    "Data Overview",
    "Model Performance",
    "Feature Importance",
    "Live Predictions",
    "Automated Report",
])

# Overview 
with tab_overview:
    st.subheader(meta["name"])
    st.markdown(f"*{meta['description']}*")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples", f"{meta['n_samples']:,}")
    c2.metric("Features", meta["n_features"])
    c3.metric("Task", meta["task"].title())
    if meta["task"] == "classification":
        balance = meta.get("class_balance", {})
        majority = max(balance.values()) / meta["n_samples"] * 100 if balance else 0
        c4.metric("Majority Class %", f"{majority:.1f}%")
    else:
        stats = meta.get("target_stats", {})
        c4.metric("Target Mean", f"{stats.get('mean', 0):.2f}")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        if meta["task"] == "classification":
            st.plotly_chart(
                plot_class_distribution(y, meta.get("target_names")),
                use_container_width=True
            )
        else:
            st.plotly_chart(
                plot_feature_distributions(pd.DataFrame({"Target": y}), n_cols=1, max_features=1),
                use_container_width=True
            )

    with col_right:
        st.markdown("##### Sample Data Preview")
        preview = X.copy()
        preview["target"] = y.values
        st.dataframe(preview.head(12), use_container_width=True, height=340)

    with st.expander("📊 Feature Distributions"):
        st.plotly_chart(plot_feature_distributions(X, max_features=9), use_container_width=True)

    with st.expander("🔗 Correlation Heatmap"):
        st.plotly_chart(plot_correlation_heatmap(X), use_container_width=True)

# Performance 
with tab_performance:
    if not st.session_state.trained:
        st.markdown(
            '<div class="info-box"><strong>No model trained yet.</strong><br>'
            'Configure and click <strong>Train Model</strong> in the sidebar.</div>',
            unsafe_allow_html=True
        )
    else:
        results = st.session_state.results
        metrics = results["metrics"]

        st.subheader(f"Performance — {results['model_name']}")

        cols = st.columns(len(metrics))
        for col, (name, val) in zip(cols, metrics.items()):
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="label">{name}</div>'
                    f'<div class="value">{val:.3f}</div></div>',
                    unsafe_allow_html=True
                )

        st.markdown("<br>", unsafe_allow_html=True)

        if "cv_accuracy_mean" in results:
            st.plotly_chart(
                plot_cv_scores(results["cv_accuracy_mean"], results["cv_accuracy_std"], "Accuracy"),
                use_container_width=True
            )
        elif "cv_r2_mean" in results:
            st.plotly_chart(
                plot_cv_scores(results["cv_r2_mean"], results["cv_r2_std"], "R²"),
                use_container_width=True
            )

        if results["task"] == "classification":
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    plot_confusion_matrix(results["confusion_matrix"], meta.get("target_names")),
                    use_container_width=True
                )
            with c2:
                if "roc_curve" in results and "ROC AUC" in metrics:
                    st.plotly_chart(
                        plot_roc_curve(
                            results["roc_curve"]["fpr"],
                            results["roc_curve"]["tpr"],
                            metrics["ROC AUC"]
                        ),
                        use_container_width=True
                    )
        else:
            bundle = st.session_state.data_bundle
            st.plotly_chart(
                plot_prediction_vs_actual(bundle["y_test"], results["y_pred"], task="regression"),
                use_container_width=True
            )

# Feature Importance
with tab_importance:
    if not st.session_state.trained:
        st.markdown(
            '<div class="info-box">Train a model first to unlock feature importance analysis.</div>',
            unsafe_allow_html=True
        )
    else:
        results = st.session_state.results
        imp = results["importance"]

        st.subheader("Feature Importance Analysis")
        top_n = st.slider("Show top N features", 5, min(30, len(imp)), min(15, len(imp)))
        imp_display = imp.head(top_n)

        st.plotly_chart(plot_feature_importance(imp_display), use_container_width=True)
        st.dataframe(imp_display, use_container_width=True)

# Live Predictions 
with tab_predict:
    if not st.session_state.trained:
        st.markdown(
            '<div class="info-box">Train a model to enable live predictions.</div>',
            unsafe_allow_html=True
        )
    else:
        st.subheader("Make a Prediction")
        results = st.session_state.results
        model = results["model"]
        feature_names = st.session_state.data_bundle["feature_names"]
        X_ref = st.session_state.X

        input_vals = {}
        cols = st.columns(3)
        for i, feat in enumerate(feature_names):
            with cols[i % 3]:
                col_data = X_ref[feat]
                if col_data.dtype in [np.float64, np.float32, float] or col_data.nunique() > 15:
                    input_vals[feat] = st.number_input(
                        feat,
                        value=float(col_data.median()),
                        min_value=float(col_data.min()),
                        max_value=float(col_data.max()),
                        key=f"pred_{feat}"
                    )
                else:
                    options = sorted(col_data.unique().tolist())
                    input_vals[feat] = st.selectbox(
                        feat, options, index=len(options)//2, key=f"pred_{feat}"
                    )

        if st.button(" Predict", type="primary"):
            input_df = pd.DataFrame([input_vals])
            bundle = st.session_state.data_bundle

            if bundle["scaler"] is not None:
                input_scaled = pd.DataFrame(
                    bundle["scaler"].transform(input_df), columns=feature_names
                )
            else:
                input_scaled = input_df

            pred = model.predict(input_scaled)[0]

            if results["task"] == "classification":
                proba = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_scaled)[0]

                target_names = meta.get("target_names", [str(i) for i in range(2)])
                pred_label = target_names[int(pred)] if int(pred) < len(target_names) else str(pred)

                st.markdown(
                    f'<div class="success-box"><h3>Prediction: '
                    f'<span style="color:#10B981">{pred_label}</span></h3></div>',
                    unsafe_allow_html=True
                )

                if proba is not None:
                    proba_df = pd.DataFrame({
                        "Class": target_names[:len(proba)],
                        "Probability": proba
                    })
                    st.bar_chart(proba_df.set_index("Class"))
            else:
                st.markdown(
                    f'<div class="success-box"><h3>Predicted Value: '
                    f'<span style="color:#10B981">{pred:.4f}</span></h3></div>',
                    unsafe_allow_html=True
                )

# Automated Report 
with tab_report:
    st.subheader("Automated PDF Report")

    if not st.session_state.trained:
        st.markdown(
            '<div class="info-box">Train a model first, then generate a report.</div>',
            unsafe_allow_html=True
        )
    else:
        results = st.session_state.results
        notes = st.text_area(
            "Optional notes",
            value="Model trained interactively via the Predictive Analytics Dashboard.",
            height=80
        )

        if st.button("Generate PDF Report", type="primary"):
            with st.spinner("Creating report..."):
                os.makedirs("reports", exist_ok=True)
                fname = f"report_{results['model_name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                path = os.path.join("reports", fname)

                cv_mean = results.get("cv_accuracy_mean") or results.get("cv_r2_mean")
                cv_std = results.get("cv_accuracy_std") or results.get("cv_r2_std")

                generate_pdf_report(
                    path,
                    meta,
                    results["model_name"],
                    results["metrics"],
                    results["importance"],
                    cv_mean,
                    cv_std,
                    results.get("classification_report"),
                    notes
                )

                with open(path, "rb") as f:
                    st.download_button(
                        "⬇Download PDF Report",
                        f,
                        file_name=fname,
                        mime="application/pdf"
                    )
                st.success(f"Report generated: `{path}`")


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#64748B; font-size:0.85rem;'>"
    "Predictive Analytics Dashboard · Built with Streamlit, scikit-learn, lots of hardwork & Plotly"
    "</div>",
    unsafe_allow_html=True
)