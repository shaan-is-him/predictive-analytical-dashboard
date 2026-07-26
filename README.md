# Predictive Analytics Dashboard  
### app link [https://visual-dash-garvil.streamlit.app]

An interactive **no-code / low-code** machine learning dashboard built with Streamlit.  
Upload your own CSV or use built-in datasets, train powerful models, analyze feature importance, make live predictions, and generate professional PDF reports — all from a clean web interface.

## Features

| Feature | Description |
|---------|-------------|
| **Custom Dataset Upload** | Upload any CSV, select the target column, and the app auto-detects Classification or Regression |
| **Built-in Datasets** | Synthetic Customer Churn, Breast Cancer, Wine Recognition, California Housing |
| **Multiple Algorithms** | Random Forest, Gradient Boosting, XGBoost, Logistic Regression, Ridge Regression |
| **Real-time Performance** | Accuracy, Precision, Recall, F1, ROC-AUC, R², RMSE, MAE + 5-Fold Cross Validation |
| **Feature Importance** | Interactive ranked bar charts + detailed table |
| **Live Predictions** | Adjust feature values and get instant predictions with probability scores |
| **Automated PDF Reports** | One-click professional report with metrics, feature rankings & classification report |
| **Beautiful Visuals** | Plotly charts with modern dark-friendly theme |
| **Model Persistence** | Trained models are saved as `.joblib` files |

---

## Tech Stack

- **Frontend / UI**: Streamlit
- **Machine Learning**: scikit-learn, XGBoost
- **Data Handling**: pandas, NumPy
- **Visualization**: Plotly
- **Reporting**: ReportLab (PDF generation)
- **Model Persistence**: joblib

---

## How It Works

### Step-by-step workflow

1. **Choose Data Source**
   - Select a **Built-in Dataset**, or
   - Switch to **Upload Your Own CSV**

2. **For Custom CSV**
   - Upload the file
   - Select the **Target Column** (what you want to predict)
   - Choose Task Type: `auto` (recommended), `classification`, or `regression`
   - The app automatically:
     - Handles missing values
     - Encodes categorical columns
     - Detects the problem type

3. **Configure Model**
   - Choose algorithm (Random Forest, XGBoost, etc.)
   - Adjust hyperparameters (n_estimators, max_depth, learning_rate, etc.)
   - Set test size and whether to standardize features

4. **Train Model**
   - Click **Train Model**
   - The app trains the model, computes metrics, and extracts feature importance

5. **Explore Results**
   - **Data Overview** → Distributions, correlation heatmap, sample data
   - **Model Performance** → Metric cards, confusion matrix / residual plot, ROC curve, Cross-Validation scores
   - **Feature Importance** → Ranked interactive chart + table
   - **Live Predictions** → Adjust inputs and get real-time predictions
   - **Automated Report** → Generate and download a polished PDF report

---

## Supported Datasets

| Dataset | Task | Description |
|---------|------|-------------|
| Synthetic Customer Churn | Classification | 2,500 samples of realistic telecom-style churn data |
| Breast Cancer Wisconsin | Classification | Classic medical diagnostic dataset |
| Wine Recognition | Classification | Chemical analysis of 3 wine cultivars |
| California Housing | Regression | Predict median house value from census data |
| **Your Own CSV** | Auto / Classification / Regression | Fully supported with automatic preprocessing |

---

## Supported Algorithms

**Classification**
- Random Forest
- Gradient Boosting
- Logistic Regression
- XGBoost
- SVM
- KNN
- Decision Tree

**Regression**
- Random Forest
- Gradient Boosting
- Ridge Regression
- XGBoost

---

## Key Insights You Can Gain

- **Which features matter most?**  
  Feature importance ranking shows the strongest predictors in your data.

- **How good is the model?**  
  Multiple metrics + Cross-Validation give a realistic view of performance (not just training score).

- **Where does the model fail?**  
  Confusion matrix and residual plots highlight error patterns.

- **What would happen if...?**  
  Live prediction tab lets you perform “what-if” analysis by changing input values.

- **Ready for reporting?**  
  One-click PDF report is perfect for sharing results with stakeholders or including in presentations.
---

## How the App Handles Your Data

When you upload a CSV, the system:

1. Drops rows where the target is missing
2. Fills numeric missing values with the **median**
3. Fills categorical missing values with the **mode**
4. Label-encodes all categorical columns
5. Auto-detects Classification vs Regression based on the number of unique values in the target
6. Optionally standardizes features (recommended for linear models)

---

## Screenshots

- `assets/overview.png` → Data Overview tab
- `assets/performance.png` → Model Performance tab
- `assets/importance.png` → Feature Importance tab
- `assets/predictions.png` → Live Predictions tab
- `assets/report.png` → PDF Report example

- 
If you face any issues or have suggestions, feel free to open an **Issue** on GitHub.

