import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import uuid
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF
import base64

# Page config
st.set_page_config(
    page_title="Zim Smart Credit",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #f5f7fa; }
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0; }
    .sub-header { font-size: 1rem; color: #4B5563; margin-top: 0; }
    .score-card {
        background: white; border-radius: 20px; padding: 2rem; box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        text-align: center; border: 1px solid #E5E7EB;
    }
    .metric-label { font-size: 0.9rem; color: #6B7280; }
    .metric-value { font-size: 2.5rem; font-weight: 700; color: #1E3A8A; }
    .risk-badge { padding: 0.25rem 1rem; border-radius: 50px; font-weight: 600; display: inline-block; }
    .risk-low { background: #D1FAE5; color: #065F46; }
    .risk-medium { background: #FEF3C7; color: #92400E; }
    .risk-high { background: #FEE2E2; color: #991B1B; }
    .stButton>button { background: #1E3A8A; color: white; border-radius: 12px; font-weight: 600; border: none; padding: 0.5rem 2rem; }
    .stButton>button:hover { background: #2563EB; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'assessments_history' not in st.session_state:
    st.session_state.assessments_history = []
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.label_encoders = {}
    st.session_state.target_encoder = None
    st.session_state.model_metrics = {}
    st.session_state.model_trained = False
    st.session_state.explainer = None
    st.session_state.X_columns = None
    st.session_state.X_train_sample = None  # for SHAP background

# Load data and add synthetic Income Source
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/Mthabisincube/Credit-Smart-project/refs/heads/master/smart_credit_scoring_zimbabwe.csv")
    np.random.seed(42)
    income_sources = ['Formal Employment', 'Informal Business', 'Farming', 'Remittances', 'Other']
    df['Income_Source'] = np.random.choice(income_sources, size=len(df), p=[0.4, 0.25, 0.15, 0.1, 0.1])
    return df

df = load_data()

# Train model on startup
if not st.session_state.model_trained:
    with st.spinner("🚀 Initializing AI credit model..."):
        X = df.drop("Credit_Score", axis=1)  # includes Income_Source
        y = df["Credit_Score"]

        # Encode categorical features
        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le

        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        model = RandomForestClassifier(
            n_estimators=100, max_depth=20, min_samples_split=5,
            min_samples_leaf=2, random_state=42, class_weight='balanced', n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y_encoded, cv=kf, scoring='accuracy') * 100

        st.session_state.model = model
        st.session_state.label_encoders = label_encoders
        st.session_state.target_encoder = target_encoder
        st.session_state.model_trained = True
        st.session_state.model_metrics = {
            'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1,
            'cv_mean': cv_scores.mean(), 'cv_scores': cv_scores.tolist(),
            'feature_importance': dict(zip(X.columns, model.feature_importances_))
        }
        st.session_state.X_columns = X.columns.tolist()
        st.session_state.X_train_sample = X_train.sample(min(100, len(X_train)), random_state=42)

        # SHAP explainer
        try:
            explainer = shap.TreeExplainer(model)
            st.session_state.explainer = explainer
        except Exception as e:
            st.warning("SHAP explainer not available – using feature importance only.")

# Helper functions
def get_risk_level(score_class):
    # score_class: Poor, Fair, Good, Excellent
    if score_class in ['Excellent', 'Good']:
        return 'Low Risk'
    elif score_class == 'Fair':
        return 'Medium Risk'
    else:
        return 'High Risk'

def get_recommendations(score_class, confidence):
    if score_class in ['Excellent', 'Good']:
        return "✅ Auto-approve up to ZWL 50,000 with favourable rates."
    elif score_class == 'Fair':
        return "⚠️ Manual review required. Offer moderate limit (ZWL 10,000–25,000)."
    else:
        return "❌ Enhanced due diligence needed. Consider collateral or guarantor."

def save_assessment(data):
    data['assessment_id'] = str(uuid.uuid4())[:8]
    data['timestamp'] = datetime.now().isoformat()
    data['date'] = datetime.now().strftime('%Y-%m-%d')
    st.session_state.assessments_history.append(data.copy())
    # Keep last 30 days only
    cutoff = datetime.now() - timedelta(days=30)
    st.session_state.assessments_history = [
        a for a in st.session_state.assessments_history
        if datetime.fromisoformat(a['timestamp']) > cutoff
    ]

def generate_pdf_report(assessment, recs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Zim Smart Credit Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, f"ID: {assessment['assessment_id']}", ln=True)
    pdf.cell(200, 10, f"Date: {assessment['date']}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, "Details", ln=True)
    pdf.set_font("Arial", size=12)
    for k, v in assessment.items():
        if k not in ['assessment_id', 'timestamp', 'date', 'max_score', 'predicted_class', 'confidence']:
            pdf.cell(200, 8, f"{k.replace('_',' ').title()}: {v}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, "Result", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 8, f"Credit Score: {assessment['predicted_class']}", ln=True)
    pdf.cell(200, 8, f"Confidence: {assessment['confidence']:.1f}%", ln=True)
    pdf.cell(200, 8, f"Risk Level: {assessment['risk_level']}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, "Recommendations", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, recs)
    return pdf.output(dest='S').encode('latin-1', errors='replace')

def get_pdf_link(pdf_bytes, filename):
    b64 = base64.b64encode(pdf_bytes).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="{filename}" style="text-decoration:none;background:#1E3A8A;color:white;padding:0.5rem 1rem;border-radius:12px;">📄 Download PDF Report</a>'

# -------------------------------------------------------------------
# Sidebar – Input Form
# -------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/credit-card.png", width=80)
    st.markdown("## Applicant Details")
    st.markdown("---")

    Location = st.selectbox("📍 Location", sorted(df['Location'].unique()))
    gender = st.selectbox("👤 Gender", sorted(df['Gender'].unique()))
    Age = st.slider("🎂 Age", int(df['Age'].min()), int(df['Age'].max()), int(df['Age'].mean()))

    st.markdown("### 💰 Financial Behavior")
    Mobile_Money_Txns = st.slider("📱 Mobile Money Transactions", 
                                  float(df['Mobile_Money_Txns'].min()), 
                                  float(df['Mobile_Money_Txns'].max()), 
                                  float(df['Mobile_Money_Txns'].mean()))
    Airtime_Spend_ZWL = st.slider("📞 Airtime Spend (ZWL)", 
                                  float(df['Airtime_Spend_ZWL'].min()), 
                                  float(df['Airtime_Spend_ZWL'].max()), 
                                  float(df['Airtime_Spend_ZWL'].mean()))
    Utility_Payments_ZWL = st.slider("💡 Utility Payments (ZWL)", 
                                     float(df['Utility_Payments_ZWL'].min()), 
                                     float(df['Utility_Payments_ZWL'].max()), 
                                     float(df['Utility_Payments_ZWL'].mean()))
    Loan_Repayment_History = st.selectbox("📊 Loan Repayment History", 
                                          sorted(df['Loan_Repayment_History'].unique()))
    Income_Source = st.selectbox("💰 Source of Income", 
                                 ['Formal Employment', 'Informal Business', 'Farming', 'Remittances', 'Other'])

    assess_button = st.button("🚀 Get Credit Score", type="primary", use_container_width=True)

# -------------------------------------------------------------------
# Main Area
# -------------------------------------------------------------------
st.markdown('<p class="main-header">Zim Smart Credit</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Alternative data credit scoring for financial inclusion</p>', unsafe_allow_html=True)

# Placeholders for results (updated after button click)
score_placeholder = st.empty()
explain_placeholder = st.empty()
rec_placeholder = st.empty()

# Default state – show instructions
if not assess_button:
    with score_placeholder.container():
        st.info("👈 Fill in the applicant's details in the sidebar and click **Get Credit Score** to see the AI-powered assessment.")
    with explain_placeholder:
        st.markdown("### How it works")
        st.markdown("""
        Our model uses alternative data to predict creditworthiness:
        - **Mobile money transactions** – frequency and volume indicate financial activity.
        - **Airtime & utility payments** – regular payments suggest reliability.
        - **Repayment history** – past behaviour is the strongest predictor.
        - **Income source** – stability of income affects risk.
        - **Demographics** – location and age provide context.
        """)

# Process assessment when button clicked
if assess_button:
    # Prepare input for model
    input_dict = {
        'Location': Location,
        'Gender': gender,
        'Age': Age,
        'Mobile_Money_Txns': Mobile_Money_Txns,
        'Airtime_Spend_ZWL': Airtime_Spend_ZWL,
        'Utility_Payments_ZWL': Utility_Payments_ZWL,
        'Loan_Repayment_History': Loan_Repayment_History,
        'Income_Source': Income_Source
    }
    input_df = pd.DataFrame([input_dict])

    # Encode categoricals
    for col, le in st.session_state.label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    # Predict
    pred_encoded = st.session_state.model.predict(input_df)[0]
    pred_class = st.session_state.target_encoder.inverse_transform([pred_encoded])[0]
    proba = st.session_state.model.predict_proba(input_df)[0]
    confidence = np.max(proba) * 100
    risk_level = get_risk_level(pred_class)
    recommendations = get_recommendations(pred_class, confidence)

    # Save assessment
    assessment_data = {
        **input_dict,
        'score': pred_class,
        'confidence': confidence,
        'risk_level': risk_level,
        'max_score': None  # not used
    }
    save_assessment(assessment_data)
    st.session_state.last_assessment = assessment_data

    # Display score gauge
    with score_placeholder.container():
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            # Map class to numeric for gauge (Poor=1, Fair=2, Good=3, Excellent=4)
            class_to_num = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
            num_score = class_to_num[pred_class]
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = num_score,
                number = {'suffix': f" - {pred_class}", 'font': {'size': 40}},
                delta = {'reference': 2.5},
                gauge = {
                    'axis': {'range': [1, 4], 'tickvals': [1,2,3,4], 'ticktext': ['Poor','Fair','Good','Excellent']},
                    'bar': {'color': "#1E3A8A"},
                    'steps': [
                        {'range': [1, 2], 'color': "#FEE2E2"},
                        {'range': [2, 3], 'color': "#FEF3C7"},
                        {'range': [3, 4], 'color': "#D1FAE5"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': num_score
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # Risk badge
            badge_class = f"risk-badge risk-{risk_level.split()[0].lower()}"
            st.markdown(f"<div style='text-align:center'><span class='{badge_class}'>{risk_level}</span></div>", unsafe_allow_html=True)

    # Explainable AI (SHAP waterfall for this prediction)
    with explain_placeholder.container():
        st.markdown("### 🔍 Why this score?")
        if st.session_state.explainer is not None:
            try:
                # Get SHAP values for this instance
                explainer = st.session_state.explainer
                # Ensure input_df has same columns as training
                X_input = input_df[st.session_state.X_columns]
                shap_values = explainer.shap_values(X_input)

                # For multi-class, we need the SHAP values for the predicted class
                class_index = pred_encoded  # numeric class
                if isinstance(shap_values, list):
                    shap_vals = shap_values[class_index][0]  # first (only) instance
                else:
                    shap_vals = shap_values[0]

                # Waterfall plot
                fig, ax = plt.subplots(figsize=(10, 4))
                shap.waterfall_plot(
                    shap.Explanation(values=shap_vals,
                                      base_values=explainer.expected_value[class_index] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                                      data=X_input.iloc[0].values,
                                      feature_names=st.session_state.X_columns),
                    show=False
                )
                st.pyplot(fig)
                plt.close()
                st.caption("The plot shows how each feature pushed the score from the base value (average prediction) to the final prediction.")
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")
                # Fallback to feature importance table
                st.markdown("**Top contributing factors:**")
                importances = st.session_state.model_metrics['feature_importance']
                top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
                for feat, imp in top_features:
                    st.markdown(f"- **{feat}**: {imp:.3f}")
        else:
            st.info("Feature importance (global) shown because SHAP not available.")
            importances = st.session_state.model_metrics['feature_importance']
            top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
            for feat, imp in top_features:
                st.markdown(f"- **{feat}**: {imp:.3f}")

    # Recommendations and PDF
    with rec_placeholder.container():
        col1, col2 = st.columns([3,1])
        with col1:
            st.info(recommendations)
        with col2:
            pdf_bytes = generate_pdf_report(assessment_data, recommendations)
            filename = f"credit_report_{assessment_data['assessment_id']}.pdf"
            st.markdown(get_pdf_link(pdf_bytes, filename), unsafe_allow_html=True)

# -------------------------------------------------------------------
# Tabs for Portfolio and Model Info
# -------------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Portfolio Overview", "🤖 Model Information"])

with tab1:
    st.markdown("### Portfolio Snapshot (Last 30 Days)")
    if st.session_state.assessments_history:
        hist_df = pd.DataFrame(st.session_state.assessments_history)
        hist_df['datetime'] = pd.to_datetime(hist_df['timestamp'])

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Assessments", len(hist_df))
        with col2:
            good_rate = (hist_df['score'].isin(['Good','Excellent']).mean() * 100)
            st.metric("Good/Excellent %", f"{good_rate:.1f}%")
        with col3:
            fair_rate = (hist_df['score'] == 'Fair').mean() * 100
            st.metric("Fair %", f"{fair_rate:.1f}%")
        with col4:
            poor_rate = (hist_df['score'] == 'Poor').mean() * 100
            st.metric("Poor %", f"{poor_rate:.1f}%")

        # Distribution plots
        col1, col2 = st.columns(2)
        with col1:
            score_counts = hist_df['score'].value_counts().reindex(['Poor','Fair','Good','Excellent'], fill_value=0)
            fig = px.bar(x=score_counts.index, y=score_counts.values, color=score_counts.index,
                         color_discrete_map={'Poor':'#FEE2E2','Fair':'#FEF3C7','Good':'#D1FAE5','Excellent':'#A7F3D0'})
            fig.update_layout(title="Credit Score Distribution", xaxis_title="Score", yaxis_title="Count", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            if 'Income_Source' in hist_df.columns:
                inc_counts = hist_df['Income_Source'].value_counts()
                fig = px.pie(values=inc_counts.values, names=inc_counts.index, title="Income Source Breakdown")
                st.plotly_chart(fig, use_container_width=True)

        # Daily trend
        daily = hist_df.groupby(hist_df['datetime'].dt.date).size().reset_index(name='count')
        fig = px.line(daily, x='datetime', y='count', title="Daily Assessment Volume")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No assessments yet. Use the assessment tool to build your portfolio.")

with tab2:
    st.markdown("### Model Performance & Interpretability")
    if st.session_state.model_trained:
        metrics = st.session_state.model_metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.1f}%")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.1f}%")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.1f}%")
        with col4:
            st.metric("F1 Score", f"{metrics['f1_score']:.1f}%")

        st.markdown("#### Cross-Validation Scores")
        cv_df = pd.DataFrame({'Fold': [f'Fold {i+1}' for i in range(5)], 'Accuracy (%)': metrics['cv_scores']})
        st.dataframe(cv_df, use_container_width=True, hide_index=True)
        st.markdown(f"**Mean CV Accuracy:** {metrics['cv_mean']:.1f}%")

        st.markdown("#### Global Feature Importance")
        imp_df = pd.DataFrame(list(metrics['feature_importance'].items()), columns=['Feature','Importance'])
        imp_df = imp_df.sort_values('Importance', ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(imp_df['Feature'], imp_df['Importance'], color='#1E3A8A')
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        st.pyplot(fig)
    else:
        st.warning("Model not trained yet.")
