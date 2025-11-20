import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px

st.set_page_config(page_title="Smart Credit App", page_icon="💳", layout="wide")

st.title('💳 Smart Credit App')
st.write('Taking Credit Scoring to the next level with Alternative Data')

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/Mthabisincube/Credit-Smart-project/refs/heads/master/smart_credit_scoring_zimbabwe.csv")
    return df

df = load_data()

# Display raw data and features
with st.expander('📊 Data Overview'):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Number of Features", len(df.columns) - 1)
    with col3:
        st.metric("Credit Score Classes", df['Credit_Score'].nunique())
    
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

# Data visualization
with st.expander('📈 Data Visualizations'):
    tab1, tab2, tab3 = st.tabs(["Credit Score Distribution", "Feature Relationships", "Location Analysis"])
    
    with tab1:
        fig = px.histogram(df, x='Credit_Score', title='Credit Score Distribution', 
                          color='Credit_Score', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            selected_x = st.selectbox("X-axis", numeric_cols, index=0)
            selected_y = st.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
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
    
    Location = st.selectbox("📍 Location", 
                           ('Bulawayo', 'Chiredzi', 'Masvingo', 'Chinhoyi', 
                            'Mutare', 'Harare', 'Gweru', 'Gokwe'))
    
    gender = st.selectbox("👤 Gender", ("Male", "Female"))
    
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
                                         ('Poor', 'Fair', 'Good', 'Excellent'))
    
    Age = st.slider("🎂 Age", 
                    int(df['Age'].min()), 
                    int(df['Age'].max()), 
                    int(df['Age'].mean()))

# Create input summary
input_summary_df = pd.DataFrame({
    "Feature": ["Location", "Gender", "Mobile Money Transactions", "Airtime Spend (ZWL)",
                "Utility Payments (ZWL)", "Loan Repayment History", "Age"],
    "Value": [Location, gender, f"{Mobile_Money_Txns:.2f}", f"{Airtime_Spend_ZWL:.2f}",
              f"{Utility_Payments_ZWL:.2f}", Loan_Repayment_History, Age]
})

# Display input summary
st.header("🧮 Credit Assessment Inputs")
col1, col2 = st.columns([1, 2])

with col1:
    st.dataframe(input_summary_df, use_container_width=True, hide_index=True)

with col2:
    # Simple scoring logic (you can replace this with your actual model)
    st.subheader("Preliminary Assessment")
    
    # Simple scoring based on inputs (placeholder logic)
    score = 0
    
    # Age factor
    if Age >= 25 and Age <= 55:
        score += 2
    elif Age > 55:
        score += 1
    
    # Transaction volume factor
    if Mobile_Money_Txns > df['Mobile_Money_Txns'].median():
        score += 1
    
    # Repayment history factor
    repayment_scores = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
    score += repayment_scores[Loan_Repayment_History]
    
    # Display assessment
    if score >= 4:
        st.success("✅ **Likely Creditworthy**")
        st.write("Based on the inputs, this profile shows positive indicators for creditworthiness.")
    elif score >= 2:
        st.warning("⚠️ **Moderate Risk**")
        st.write("This profile shows mixed indicators. Additional assessment may be needed.")
    else:
        st.error("❌ **Higher Risk Profile**")
        st.write("This profile may require additional scrutiny or collateral.")

# Machine Learning Section (Optional)
with st.expander("🤖 Machine Learning Model (Sample)"):
    st.write("This section demonstrates how a machine learning model could be integrated.")
    
    if st.button("Train Sample Model"):
        with st.spinner("Training model..."):
            # Prepare data
            X_ml = df.drop("Credit_Score", axis=1)
            y_ml = df["Credit_Score"]
            
            # Encode categorical variables
            le_dict = {}
            for col in X_ml.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X_ml[col] = le.fit_transform(X_ml[col])
                le_dict[col] = le
            
            # Train model
            X_train, X_test, y_train, y_test = train_test_split(X_ml, y_ml, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            st.success(f"Model trained with {accuracy:.2%} accuracy")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X_ml.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig = px.bar(feature_importance, x='importance', y='feature', 
                        title='Feature Importance in Credit Scoring')
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit | Smart Credit Assessment for Zimbabwe</p>
</div>
""", unsafe_allow_html=True)
