# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
from datetime import datetime

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Optional libraries - used if installed
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

# ------------------------
# App Config
# ------------------------
st.set_page_config(
    page_title="Zim Smart Credit App",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Small CSS to keep a pleasant look (not too heavy)
st.markdown(
    """
    <style>
    .big-title { font-size:34px; font-weight:700; color: #2b5876; }
    .muted { color: #6c757d; }
    .card { background-color: #ffffff; padding: 14px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.06); }
    .success-box { background: linear-gradient(135deg,#d4edda 0%,#c3e6cb 100%); padding:12px; border-radius:10px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------
# Utilities
# ------------------------
DATA_URL = "https://raw.githubusercontent.com/Mthabisincube/Credit-Smart-project/refs/heads/master/smart_credit_scoring_zimbabwe.csv"
MODEL_PATH = "trained_model.joblib"
PIPELINE_PATH = "pipeline.joblib"

@st.cache_data(show_spinner=False)
def load_data(url=DATA_URL):
    df = pd.read_csv(url)
    # Basic cleaning thought: ensure column names are consistent
    df.columns = [c.strip() for c in df.columns]
    return df

def save_model(obj, path):
    joblib.dump(obj, path)

def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

def make_pdf_report(summary_text: str, prob_df: pd.DataFrame = None) -> bytes:
    """
    Create a simple PDF report using fpdf if available. If not available, raises.
    Returns PDF bytes.
    """
    if not FPDF_AVAILABLE:
        raise RuntimeError("FPDF not installed.")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 8, "Zim Smart Credit Assessment Report", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", size=10)
    for line in summary_text.split("\n"):
        pdf.multi_cell(0, 6, line)
    if prob_df is not None:
        pdf.ln(6)
        pdf.set_font("Arial", style="B", size=11)
        pdf.cell(0, 6, "Prediction Probabilities:", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.ln(2)
        # Table header
        pdf.cell(60, 6, "Class", border=1)
        pdf.cell(60, 6, "Probability (%)", border=1, ln=True)
        for _, row in prob_df.iterrows():
            pdf.cell(60, 6, str(row['Credit Score']), border=1)
            pdf.cell(60, 6, f"{row['Probability (%)']}", border=1, ln=True)
    return pdf.output(dest="S").encode("latin-1")

def align_dummies(df, cols_reference):
    """
    Aligns dataframe's one-hot columns to the reference set of columns.
    Adds missing columns with 0 and drops extras.
    """
    for c in cols_reference:
        if c not in df.columns:
            df[c] = 0
    # Remove columns not in reference
    for c in list(df.columns):
        if c not in cols_reference:
            df.drop(columns=c, inplace=True)
    # Ensure same order
    df = df[cols_reference]
    return df

# ------------------------
# Load data
# ------------------------
with st.spinner("Loading dataset..."):
    df = load_data()

# Quick check
if df is None or df.empty:
    st.error("Failed to load dataset. Check DATA_URL or network.")
    st.stop()

# ------------------------
# Sidebar: Inputs & controls
# ------------------------
st.sidebar.markdown("<div class='card'><h3 class='muted'>📋 Enter details</h3></div>", unsafe_allow_html=True)

# Basic user fields (sourced from dataset unique values where appropriate)
location_options = sorted(df['Location'].dropna().unique())
gender_options = sorted(df['Gender'].dropna().unique())
repayment_options = sorted(df['Loan_Repayment_History'].dropna().unique())

with st.sidebar.form(key="input_form"):
    st.markdown("### 👤 Personal")
    location = st.selectbox("📍 Location", options=location_options)
    gender = st.selectbox("👤 Gender", options=gender_options)
    age = st.slider("🎂 Age", int(df['Age'].min()), int(df['Age'].max()), int(df['Age'].median()))
    st.markdown("### 💰 Financial Behaviour")
    mobile_txns = st.number_input("📱 Mobile Money Transactions (avg/month)", value=float(df['Mobile_Money_Txns'].median()))
    airtime_spend = st.number_input("📞 Airtime Spend (ZWL / month)", value=float(df['Airtime_Spend_ZWL'].median()))
    utility_pay = st.number_input("💡 Utility Payments (ZWL / month)", value=float(df['Utility_Payments_ZWL'].median()))
    repayment_hist = st.selectbox("📊 Loan Repayment History", options=repayment_options)
    submitted = st.form_submit_button("Save inputs")

if submitted:
    st.sidebar.success("Inputs saved ✔️")

# ML controls
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Model Controls")
model_choice = st.sidebar.selectbox("Model", options=["Random Forest", "Logistic Regression"] + (["XGBoost"] if XGBOOST_AVAILABLE else []))
enable_tuning = st.sidebar.checkbox("Enable quick hyperparam tuning (Randomized)", value=False)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=9999, value=42)

# Load saved pipeline/model if exists into session
if "pipeline" not in st.session_state:
    loaded = load_model(PIPELINE_PATH)
    if loaded is not None:
        st.session_state.pipeline = loaded
    else:
        st.session_state.pipeline = None

if "model" not in st.session_state:
    loaded_model = load_model(MODEL_PATH)
    if loaded_model is not None:
        st.session_state.model = loaded_model
    else:
        st.session_state.model = None

# ------------------------
# Main layout
# ------------------------
st.title("💳 Zim Smart Credit App — Finalized")
st.markdown("A modernized Streamlit app for alternative-data credit scoring. Enter your details in the sidebar and explore the tabs below.")

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Assessment", "📊 Dataset & EDA", "🔍 ML / Train", "📥 Export & Model"])

# ------------------------
# Tab 1: Assessment (quick rule-based + saved input display)
# ------------------------
with tab1:
    st.markdown("## 🎯 Instant Credit Assessment (Rule-based)")
    # input summary
    input_summary = {
        "Location": location,
        "Gender": gender,
        "Age": age,
        "Mobile_Money_Txns": mobile_txns,
        "Airtime_Spend_ZWL": airtime_spend,
        "Utility_Payments_ZWL": utility_pay,
        "Loan_Repayment_History": repayment_hist
    }
    st.write(pd.DataFrame([input_summary]).T.rename(columns={0: "Value"}))
    st.markdown("---")

    # Simple rule-based scoring improved from previous
    score = 0
    max_score = 8  # adjust scale

    # Age factor
    if 30 <= age <= 50:
        score += 2
        age_label = "Optimal"
    elif 25 <= age < 30 or 50 < age <= 60:
        score += 1
        age_label = "Moderate"
    else:
        age_label = "Higher risk"

    # Mobile txns factor relative to dataset median
    median_mobile = df['Mobile_Money_Txns'].median()
    if mobile_txns >= median_mobile:
        score += 2
        mobile_label = "Active"
    else:
        mobile_label = "Less active"
    
    # Airtime spend: normalized threshold
    airtime_q1 = df['Airtime_Spend_ZWL'].quantile(0.25)
    airtime_q3 = df['Airtime_Spend_ZWL'].quantile(0.75)
    if airtime_spend >= airtime_q3:
        score += 1
        airtime_label = "High usage"
    elif airtime_q1 <= airtime_spend < airtime_q3:
        score += 1
        airtime_label = "Average"
    else:
        airtime_label = "Low"

    # Utility payments - signal of account activity
    if utility_pay > df['Utility_Payments_ZWL'].median():
        score += 1
        utility_label = "On-time"
    else:
        utility_label = "Lower payments"

    # Repayment history mapping
    rep_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
    rep_score = rep_map.get(repayment_hist, 0)
    score += rep_score

    st.markdown("### 🔢 Subscores")
    st.write({
        "Age": age_label,
        "Mobile Activity": mobile_label,
        "Airtime": airtime_label,
        "Utility Payments": utility_label,
        "Repayment History (points)": rep_score
    })

    # Final
    percentage = (score / max_score) * 100
    st.markdown("---")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Overall Score", f"{score}/{max_score}", delta=f"{percentage:.1f}%")
        st.progress(min(percentage / 100.0, 1.0))
    with col2:
        if score >= 6:
            st.success("✅ Excellent creditworthiness — low risk")
            st.markdown("Recommend standard approval with favorable terms.")
        elif score >= 3:
            st.info("⚠️ Moderate risk — extra verification recommended")
            st.markdown("Recommend standard verification and moderate limits.")
        else:
            st.error("❌ High risk — enhanced verification required")
            st.markdown("Recommend enhanced verification and possibly collateral.")

# ------------------------
# Tab 2: Dataset & EDA
# ------------------------
with tab2:
    st.header("📊 Dataset Overview & EDA")
    st.markdown("Quick dataset preview and simple visualizations.")
    st.write(f"Total records: **{len(df):,}**")
    st.dataframe(df.head(12), use_container_width=True)

    st.markdown("### Credit Score distribution")
    try:
        score_counts = df['Credit_Score'].value_counts().sort_index()
        st.bar_chart(score_counts)
    except Exception:
        st.info("No 'Credit_Score' distribution available in this dataset.")

    st.markdown("### Feature distributions")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    chosen = st.selectbox("Pick numeric to plot", options=numeric_cols)
    if chosen:
        hist_values = np.histogram(df[chosen].dropna(), bins=20)[0]
        st.bar_chart(hist_values)

# ------------------------
# Tab 3: ML / Train
# ------------------------
with tab3:
    st.header("🔍 Machine Learning & Model Training")
    st.markdown("Choose model, (optional) run quick tuning, and train. Trained pipeline persists to disk.")

    # Button to prepare training data
    with st.expander("Dataset selection & preview", expanded=False):
        st.write("Target column assumed to be 'Credit_Score'. If not present, training is disabled.")
        if 'Credit_Score' not in df.columns:
            st.error("Dataset lacks 'Credit_Score' target column. Cannot train.")
        else:
            st.write("Classes:", df['Credit_Score'].unique())
            st.dataframe(df.head())

    training_col, pred_col = None, None
    if 'Credit_Score' in df.columns:
        X = df.drop(columns=["Credit_Score"])
        y = df["Credit_Score"].astype(str)  # ensure string classes
        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

        st.write(f"Numeric features ({len(numeric_features)}): {numeric_features}")
        st.write(f"Categorical features ({len(categorical_features)}): {categorical_features}")

        # Build preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='drop')

        # Choose estimator
        if model_choice == "Random Forest":
            estimator = RandomForestClassifier(random_state=random_state)
            param_dist = {
                "clf__n_estimators": [100, 200, 400],
                "clf__max_depth": [None, 6, 12, 20],
                "clf__min_samples_split": [2, 5, 10]
            }
        elif model_choice == "Logistic Regression":
            estimator = LogisticRegression(max_iter=1000, random_state=random_state)
            param_dist = {
                "clf__C": [0.01, 0.1, 1, 10],
                "clf__penalty": ['l2']
            }
        elif model_choice == "XGBoost" and XGBOOST_AVAILABLE:
            estimator = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
            param_dist = {
                "clf__n_estimators": [100, 200, 400],
                "clf__max_depth": [3, 6, 12],
                "clf__learning_rate": [0.01, 0.05, 0.1]
            }
        else:
            st.error("Selected model not available.")
            estimator = RandomForestClassifier(random_state=random_state)
            param_dist = {}

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('clf', estimator)
        ])

        # Train / tune UI
        do_train = st.button("Train model now")
        if do_train:
            try:
                # Split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
                if enable_tuning and param_dist:
                    st.info("Running RandomizedSearchCV (quick). This will take longer depending on dataset size.")
                    rnd = RandomizedSearchCV(
                        estimator=pipeline,
                        param_distributions=param_dist,
                        n_iter=10,
                        scoring='accuracy',
                        n_jobs=-1,
                        random_state=random_state,
                        verbose=0
                    )
                    rnd.fit(X_train, y_train)
                    best_pipeline = rnd.best_estimator_
                    st.success(f"Tuning completed. Best score: {rnd.best_score_:.3f}")
                else:
                    best_pipeline = pipeline.fit(X_train, y_train)

                # Save pipeline & store into session
                save_model(best_pipeline, PIPELINE_PATH)
                st.session_state.pipeline = best_pipeline

                # predictions
                y_pred = best_pipeline.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"Training successful. Test Accuracy: {acc:.3f}")

                # classification report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).T)

                # confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                st.subheader("Confusion Matrix")
                st.write(cm)

                # feature importance or permutation explainability
                st.subheader("Explainability")
                if SHAP_AVAILABLE:
                    st.info("Computing SHAP values (may take a moment).")
                    # We need the transformed X for SHAP depending on model type
                    try:
                        # Build a small explainer on training set
                        # Get feature names after preprocessing
                        X_trans = best_pipeline.named_steps['preprocessor'].transform(X_train)
                        ohe = None
                        # Attempt to get names for columns (works for OneHotEncoder in sklearn >=1.0)
                        try:
                            cat_cols = best_pipeline.named_steps['preprocessor'].transformers_[1][2]
                            onehot = best_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
                            ohe_feature_names = onehot.get_feature_names_out(cat_cols)
                            feature_names = list(best_pipeline.named_steps['preprocessor'].transformers_[0][2]) + list(ohe_feature_names)
                        except Exception:
                            # fallback
                            feature_names = None

                        explainer = shap.Explainer(best_pipeline.named_steps['clf'], X_trans)
                        shap_values = explainer(X_trans[:100])
                        st.write("SHAP summary (first 100 samples):")
                        st.pyplot(shap.plots.beeswarm(shap_values, show=False))
                    except Exception as e:
                        st.warning(f"SHAP failed or too slow in this environment: {e}")
                else:
                    # Permutation importance fallback
                    st.info("SHAP not available — using permutation importances (approx).")
                    try:
                        from sklearn.inspection import permutation_importance
                        # Use small sample for speed
                        result = permutation_importance(best_pipeline, X_test.sample(min(200, len(X_test)), random_state=1), y_test.sample(min(200, len(y_test)), random_state=1), n_repeats=8, random_state=0, n_jobs=-1)
                        importances = pd.DataFrame({'feature': np.arange(len(result.importances_mean)), 'importance': result.importances_mean})
                        st.dataframe(importances.sort_values('importance', ascending=False).head(10))
                    except Exception as e:
                        st.warning(f"Permutation importance failed: {e}")

            except Exception as e:
                st.error(f"Training failed: {e}")

# ------------------------
# Tab 4: Export & Model
# ------------------------
with tab4:
    st.header("📥 Export & Manage Model")
    st.markdown("You can load/save the trained pipeline and make predictions for the saved inputs.")

    if st.session_state.pipeline is None:
        st.warning("No trained pipeline found in session. Train a model first under the ML tab.")
    else:
        st.success("A trained pipeline is available in session.")
        # Quick prediction using the saved inputs from sidebar
        if submitted:
            st.markdown("### 🔮 Predict for current sidebar inputs")
            user_df = pd.DataFrame([{
                'Location': location,
                'Gender': gender,
                'Mobile_Money_Txns': mobile_txns,
                'Airtime_Spend_ZWL': airtime_spend,
                'Utility_Payments_ZWL': utility_pay,
                'Loan_Repayment_History': repayment_hist,
                'Age': age
            }])

            try:
                preds = st.session_state.pipeline.predict(user_df)
                pred_proba = st.session_state.pipeline.predict_proba(user_df) if hasattr(st.session_state.pipeline.named_steps['clf'], "predict_proba") else None
                st.markdown("#### Prediction")
                st.write("Predicted Class:", preds[0])
                if pred_proba is not None:
                    probs = (pred_proba[0] * 100).round(2)
                    classes = st.session_state.pipeline.named_steps['clf'].classes_
                    prob_df = pd.DataFrame({'Credit Score': classes, 'Probability (%)': probs})
                    st.dataframe(prob_df.sort_values('Probability (%)', ascending=False))

                # Download PDF or CSV
                st.markdown("### 📤 Export Report")
                report_text = f"Zim Smart Credit Assessment Report\nDate: {datetime.utcnow().isoformat()}Z\n\nInputs:\n"
                for k, v in input_summary.items():
                    report_text += f"{k}: {v}\n"
                report_text += f"\nRule-based Score: {score}/{max_score} ({percentage:.1f}%)\n"
                if pred_proba is not None:
                    report_text += "\nPrediction Probabilities:\n"
                    for idx, row in prob_df.iterrows():
                        report_text += f"{row['Credit Score']}: {row['Probability (%)']}%\n"

                # PDF download
                if FPDF_AVAILABLE:
                    try:
                        pdf_bytes = make_pdf_report(report_text, prob_df if pred_proba is not None else None)
                        st.download_button("Download PDF Report", data=pdf_bytes, file_name="credit_report.pdf", mime="application/pdf")
                    except Exception as e:
                        st.warning(f"PDF creation failed, fallback to CSV/ TXT. Error: {e}")
                        csv_buf = io.StringIO()
                        prob_df.to_csv(csv_buf, index=False)
                        st.download_button("Download probabilities CSV", data=csv_buf.getvalue(), file_name="probabilities.csv", mime="text/csv")
                else:
                    st.info("FPDF not installed. Offering CSV download instead.")
                    if pred_proba is not None:
                        csv_buf = io.StringIO()
                        prob_df.to_csv(csv_buf, index=False)
                        st.download_button("Download probabilities CSV", data=csv_buf.getvalue(), file_name="probabilities.csv", mime="text/csv")
                    txt_buf = io.StringIO()
                    txt_buf.write(report_text)
                    st.download_button("Download report (TXT)", data=txt_buf.getvalue(), file_name="credit_report.txt", mime="text/plain")

                # Option to save pipeline/model explicitly
                if st.button("Save pipeline to disk"):
                    try:
                        save_model(st.session_state.pipeline, PIPELINE_PATH)
                        st.success("Pipeline saved to disk.")
                    except Exception as e:
                        st.error(f"Failed to save pipeline: {e}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.markdown("---")
    # Model loading
    if load_model(MODEL_PATH) is None:
        st.info("No standalone model file found on disk (trained_model.joblib). The pipeline is preferred for re-usage.")
    else:
        st.success("Found model artifact on disk.")

    # Option to clear saved pipeline from session (but not disk)
    if st.button("Clear trained pipeline from session"):
        st.session_state.pipeline = None
        st.success("Cleared session pipeline.")

st.markdown("---")
st.caption("App created for demonstration. Consider adding authentication, privacy policies and secure storage for production deployments.")
