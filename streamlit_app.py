import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px

# Page config
st.set_page_config(page_title="Smart Credit App", page_icon="💳", layout="wide")

st.title('💳 Smart Credit App')
st.write('Taking Credit Scoring to the next level with Alternative Data')

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/Mthabisincube/Credit-Smart-project/refs/heads/master/smart_credit_scoring_zimbabwe.csv")

df = load_data()

# Data overview
with st.expander('📊 Data Overview'):
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Number of Features", len(df.columns) - 1)
    col3.metric("Credit Score Classes", df['Credit_Score'].nunique())

    st.write('**Raw Data**')
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.write('**X (Features)**')
        X = df.drop("Credit_Score", axis=1)
        st.dataframe(X, use_container_width=True)
    with col2:
        st.write('**Y (Target)**')
        Y = df["Credit_Score"]
        st.dataframe(Y, use_container_width=True)

# Visualizations
with st.expander('📈 Data Visualizations'):
    tab1, tab2, tab3 = st.tabs(["Credit Score Distribution", "Feature Relationships", "Location Analysis"])

    with tab1:
        fig = px.histogram(df, x='Credit_Score', color='Credit_Score', title='Credit Score Distribution', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            selected_x = st.selectbox("X-axis", numeric_cols, index=0)
            selected_y = st.selectbox("Y-axis", numeric_cols, index=1)
            fig = px.scatter(df, x=selected_x, y=selected_y, color='Credit_Score',
                             title=f'{selected_y} vs {selected_x} by Credit Score')
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        location_counts = df.groupby(['Location', 'Credit_Score']).size().reset_index(name='Count')
        fig = px.bar(location_counts, x='Location', y='Count', color='Credit_Score',
                     title='Credit Scores by Location', barmode='stack')
        st.plotly_chart(fig, use_container_width=True)

# Sidebar inputs
with st.sidebar:
    st.header('**Alternative Data Input**')
    st.markdown("---")

    Location = st.selectbox("📍 Location", df['Location'].unique())
    gender = st.selectbox("👤 Gender", df['Gender'].unique())
    Mobile_Money_Txns = st.slider("📱 Mobile Money Transactions", float(df['Mobile_Money_Txns'].min()), float(df['Mobile_Money_Txns'].max()), float(df['Mobile_Money_Txns'].mean()))
    Airtime_Spend_ZWL = st.slider("📞 Airtime Spend (ZWL)", float(df['Airtime_Spend_ZWL'].min()), float(df['Airtime_Spend_ZWL'].max()), float(df['Airtime_Spend_ZWL'].mean()))
    Utility_Payments_ZWL = st.slider("💡 Utility Payments (ZWL)", float(df['Utility_Payments_ZWL'].min()), float(df['Utility_Payments_ZWL'].max()), float(df['Utility_Payments_ZWL'].mean()))
    Loan_Repayment_History = st.selectbox("📊 Loan Repayment History", df['Loan_Repayment_History'].unique())
    Age = st.slider("🎂 Age", int(df['Age'].min()), int(df['Age'].max()), int(df['Age'].mean()))

# Input summary
input_summary_df = pd.DataFrame({
    "Feature": ["Location", "Gender", "Mobile Money Transactions", "Airtime Spend (ZWL)",
                "Utility Payments (ZWL)", "Loan Repayment History", "Age"],
    "Value": [Location, gender, f"{Mobile_Money_Txns:.2f}", f"{Airtime_Spend_ZWL:.2f}",
              f"{Utility_Payments_ZWL:.2f}", Loan_Repayment_History, Age]
})

summary = f"""
- **Location**: {Location}  
- **Gender**: {gender}  
- **Mobile Money Transactions**: {Mobile_Money_Txns:.2f}  
- **Airtime Spend (ZWL)**: {Airtime_Spend_ZWL:.2f}  
- **Utility Payments (ZWL)**: {Utility_Payments_ZWL:.2f}  
- **Loan Repayment History**: {Loan_Repayment_History}  
- **Age**: {Age}
"""

st.header("🧮 Credit Assessment Inputs")
col1, col2 = st.columns([1, 2])
with col1:
    st.dataframe(input_summary_df, use_container_width=True, hide_index=True)
    with st.expander("📝 Text Summary of Inputs"):
        st.markdown(summary)

with col2:
    st.subheader("Preliminary Assessment")
    score = 0
    if 25 <= Age <= 55: score += 2
    elif Age > 55: score += 1
    if Mobile_Money_Txns > df['Mobile_Money_Txns'].median(): score += 1
    score += {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}[Loan_Repayment_History]

    if score >= 4:
        st.success("✅ **Likely Creditworthy**")
        st.write("This profile shows positive indicators for creditworthiness.")
    elif score >= 2:
        st.warning("⚠️ **Moderate Risk**")
        st.write("Mixed indicators. Additional assessment may be needed.")
    else:
        st.error("❌ **Higher Risk Profile**")
        st.write("This profile may require additional scrutiny or collateral.")

# ML model section
with st.expander("🤖 Machine Learning Model (Sample)"):
    st.write("This section demonstrates how a machine learning model could be integrated.")
    if st.button("Train Sample Model"):
        with st.spinner("Training model..."):
            X_ml = df.drop("Credit_Score", axis=1)
            y_ml = df["Credit_Score"]

            le_dict = {}
            for col in X_ml.select_dtypes(include='object').columns:
                le = LabelEncoder()
                X_ml[col] = le.fit_transform(X_ml[col])
                le_dict[col] = le

            X_train, X_test, y_train, y_test = train_test_split(X_ml, y_ml, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.success(f"Model trained with {accuracy:.2%} accuracy")

            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                               x=sorted(y_ml.unique()), y=sorted(y_ml.unique()),
                               title="Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)

            feature_importance = pd.DataFrame({
                'feature': X_ml.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            fig = px.bar(feature_importance, x='importance', y='feature', title='Feature Importance in Credit Scoring')
            st.plotly_chart(fig, use_container_width=True)

