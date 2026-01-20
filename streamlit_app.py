import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Zim Smart Credit App",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# STYLING (UNCHANGED)
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-image: linear-gradient(rgba(255,255,255,0.92), rgba(255,255,255,0.92)),
    url('https://images.unsplash.com/photo-1554224155-6726b3ff858f');
    background-size: cover;
    background-attachment: fixed;
}
.main-header {
    font-size: 3.5rem;
    text-align: center;
    font-weight: bold;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
}
.card {
    background-color: rgba(248,249,250,0.95);
    padding: 1.5rem;
    border-radius: 15px;
}
.tab-content {
    background-color: rgba(255,255,255,0.9);
    border-radius: 12px;
    padding: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown('<h1 class="main-header">🏦 Zim Smart Credit App</h1>', unsafe_allow_html=True)
st.markdown("### 💳 AI-Powered Credit Scoring Using Alternative Data")
st.markdown("---")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(
        "https://raw.githubusercontent.com/Mthabisincube/Credit-Smart-project/refs/heads/master/smart_credit_scoring_zimbabwe.csv"
    )

df = load_data()

# --------------------------------------------------
# TRAIN ONE GLOBAL MODEL
# --------------------------------------------------
@st.cache_resource
def train_model(df):
    X = df.drop("Credit_Score", axis=1)
    y = df["Credit_Score"]

    encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    target_encoder = LabelEncoder()
    y_enc = target_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100

    return model, accuracy, encoders, target_encoder

model, model_accuracy, encoders, target_encoder = train_model(df)

# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------
with st.sidebar:
    st.header("🔮 Credit Assessment")

    Location = st.selectbox("📍 Location", sorted(df["Location"].unique()))
    Gender = st.selectbox("👤 Gender", sorted(df["Gender"].unique()))
    Age = st.slider("🎂 Age", int(df.Age.min()), int(df.Age.max()), int(df.Age.mean()))

    Mobile_Money_Txns = st.slider(
        "📱 Mobile Money Transactions",
        float(df.Mobile_Money_Txns.min()),
        float(df.Mobile_Money_Txns.max()),
        float(df.Mobile_Money_Txns.mean())
    )

    Airtime_Spend_ZWL = st.slider(
        "📞 Airtime Spend (ZWL)",
        float(df.Airtime_Spend_ZWL.min()),
        float(df.Airtime_Spend_ZWL.max()),
        float(df.Airtime_Spend_ZWL.mean())
    )

    Utility_Payments_ZWL = st.slider(
        "💡 Utility Payments (ZWL)",
        float(df.Utility_Payments_ZWL.min()),
        float(df.Utility_Payments_ZWL.max()),
        float(df.Utility_Payments_ZWL.mean())
    )

    Loan_Repayment_History = st.selectbox(
        "📊 Loan Repayment History",
        sorted(df.Loan_Repayment_History.unique())
    )

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "🔍 Analysis",
    "🎯 Assessment",
    "🤖 AI Model",
    "📄 Reports"
])

# --------------------------------------------------
# TAB 1: DASHBOARD
# --------------------------------------------------
with tab1:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><h3>Records</h3><h2>{len(df)}</h2></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><h3>Features</h3><h2>{df.shape[1]-1}</h2></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><h3>Classes</h3><h2>{df.Credit_Score.nunique()}</h2></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><h3>Model Accuracy</h3><h2>{model_accuracy:.1f}%</h2></div>", unsafe_allow_html=True)

    st.dataframe(df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# TAB 2: ANALYSIS
# --------------------------------------------------
with tab2:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.bar_chart(df.Credit_Score.value_counts())
    st.bar_chart(df.Location.value_counts())
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# TAB 3: RULE-BASED ASSESSMENT
# --------------------------------------------------
with tab3:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    score = 0
    if 30 <= Age <= 50:
        score += 2
    if Mobile_Money_Txns > df.Mobile_Money_Txns.median():
        score += 1

    repay_map = {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3}
    score += repay_map[Loan_Repayment_History]

    st.metric("Rule-Based Score", f"{score}/6")
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# TAB 4: AI MODEL PREDICTION
# --------------------------------------------------
with tab4:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    user_df = pd.DataFrame({
        "Location": [Location],
        "Gender": [Gender],
        "Mobile_Money_Txns": [Mobile_Money_Txns],
        "Airtime_Spend_ZWL": [Airtime_Spend_ZWL],
        "Utility_Payments_ZWL": [Utility_Payments_ZWL],
        "Loan_Repayment_History": [Loan_Repayment_History],
        "Age": [Age]
    })

    for col in user_df.select_dtypes(include="object").columns:
        user_df[col] = encoders[col].transform(user_df[col])

    pred = model.predict(user_df)[0]
    proba = model.predict_proba(user_df)[0].max() * 100
    credit_class = target_encoder.inverse_transform([pred])[0]

    st.metric("AI Credit Score", credit_class)
    st.metric("Confidence", f"{proba:.1f}%")
    st.metric("Model Accuracy", f"{model_accuracy:.2f}%")

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# TAB 5: REPORTS (NO NEW LIBRARIES)
# --------------------------------------------------
with tab5:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    report = user_df.copy()
    report["Predicted_Credit_Score"] = credit_class
    report["Prediction_Confidence_%"] = round(proba, 2)
    report["Model_Accuracy_%"] = round(model_accuracy, 2)

    st.dataframe(report, use_container_width=True)

    st.download_button(
        "⬇️ Download Credit Report (CSV)",
        report.to_csv(index=False),
        file_name="credit_report.csv",
        mime="text/csv"
    )

    st.markdown('</div>', unsafe_allow_html=True)
