import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Smart Credit App", page_icon="💳", layout="wide")

st.title('💳 Zimbabwe Smart Credit App')
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

# Data visualization using Streamlit native charts
with st.expander('📈 Data Visualizations'):
    tab1, tab2, tab3 = st.tabs(["Credit Score Distribution", "Feature Analysis", "Location Analysis"])
    
    with tab1:
        st.write('**Credit Score Distribution**')
        score_counts = df['Credit_Score'].value_counts().sort_index()
        st.bar_chart(score_counts)
        
        # Show distribution as table
        st.write("**Count by Credit Score:**")
        distribution_df = score_counts.reset_index()
        distribution_df.columns = ['Credit Score', 'Count']
        st.dataframe(distribution_df, hide_index=True)
    
    with tab2:
        st.write('**Numerical Features Analysis**')
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            selected_feature = st.selectbox("Select feature to analyze:", numeric_cols)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Distribution of {selected_feature}**")
                # Create histogram using bar chart
                hist_data = np.histogram(df[selected_feature], bins=20)[0]
                st.bar_chart(hist_data)
            
            with col2:
                st.write(f"**Statistics for {selected_feature}**")
                stats_data = {
                    'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '25% Quartile', '75% Quartile'],
                    'Value': [
                        f"{df[selected_feature].mean():.2f}",
                        f"{df[selected_feature].median():.2f}",
                        f"{df[selected_feature].std():.2f}",
                        f"{df[selected_feature].min():.2f}",
                        f"{df[selected_feature].max():.2f}",
                        f"{df[selected_feature].quantile(0.25):.2f}",
                        f"{df[selected_feature].quantile(0.75):.2f}"
                    ]
                }
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, hide_index=True)
    
    with tab3:
        st.write('**Credit Scores by Location**')
        location_summary = df.groupby('Location')['Credit_Score'].value_counts().unstack().fillna(0)
        st.dataframe(location_summary)
        
        st.write("**Record Count by Location**")
        location_counts = df['Location'].value_counts()
        st.bar_chart(location_counts)

# Sidebar inputs
with st.sidebar:
    st.header('**Alternative Data Input**')
    st.markdown("---")
    
    Location = st.selectbox("📍 Location", 
                           sorted(df['Location'].unique()))
    
    gender = st.selectbox("👤 Gender", sorted(df['Gender'].unique()))
    
    # Get min, max, mean from actual data for sliders
    mobile_min = float(df['Mobile_Money_Txns'].min())
    mobile_max = float(df['Mobile_Money_Txns'].max())
    mobile_mean = float(df['Mobile_Money_Txns'].mean())
    
    airtime_min = float(df['Airtime_Spend_ZWL'].min())
    airtime_max = float(df['Airtime_Spend_ZWL'].max())
    airtime_mean = float(df['Airtime_Spend_ZWL'].mean())
    
    utility_min = float(df['Utility_Payments_ZWL'].min())
    utility_max = float(df['Utility_Payments_ZWL'].max())
    utility_mean = float(df['Utility_Payments_ZWL'].mean())
    
    age_min = int(df['Age'].min())
    age_max = int(df['Age'].max())
    age_mean = int(df['Age'].mean())
    
    Mobile_Money_Txns = st.slider("📱 Mobile Money Transactions", 
                                  mobile_min, mobile_max, mobile_mean)
    
    Airtime_Spend_ZWL = st.slider("📞 Airtime Spend (ZWL)", 
                                  airtime_min, airtime_max, airtime_mean)
    
    Utility_Payments_ZWL = st.slider("💡 Utility Payments (ZWL)", 
                                     utility_min, utility_max, utility_mean)
    
    Loan_Repayment_History = st.selectbox("📊 Loan Repayment History", 
                                         sorted(df['Loan_Repayment_History'].unique()))
    
    Age = st.slider("🎂 Age", age_min, age_max, age_mean)

# Create input summary
input_summary_df = pd.DataFrame({
    "Feature": ["Location", "Gender", "Mobile Money Transactions", "Airtime Spend (ZWL)",
                "Utility Payments (ZWL)", "Loan Repayment History", "Age"],
    "Value": [Location, gender, f"{Mobile_Money_Txns:.2f}", f"{Airtime_Spend_ZWL:.2f}",
              f"{Utility_Payments_ZWL:.2f}", Loan_Repayment_History, Age]
})

# Display input summary and assessment
st.header("🧮 Credit Assessment Inputs")
col1, col2 = st.columns([1, 2])

with col1:
    st.write("**Your Input Summary**")
    st.dataframe(input_summary_df, use_container_width=True, hide_index=True)

with col2:
    # Simple scoring logic
    st.subheader("Preliminary Assessment")
    
    # Simple scoring based on inputs
    score = 0
    max_score = 6
    
    # Age factor
    if 30 <= Age <= 50:
        score += 2
        age_feedback = "✅ Optimal age range (30-50)"
    elif 25 <= Age < 30 or 50 < Age <= 60:
        score += 1
        age_feedback = "⚠️ Moderate age range"
    else:
        age_feedback = "⚠️ Extreme age range"
    
    # Transaction volume factor
    mobile_median = df['Mobile_Money_Txns'].median()
    if Mobile_Money_Txns > mobile_median:
        score += 1
        mobile_feedback = f"✅ Above average transactions (> {mobile_median:.1f})"
    else:
        mobile_feedback = f"⚠️ Below average transactions"
    
    # Repayment history factor
    repayment_scores = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
    score += repayment_scores[Loan_Repayment_History]
    repayment_feedback = f"Repayment history: {Loan_Repayment_History}"
    
    # Calculate percentage
    percentage = (score / max_score) * 100
    
    # Display progress bar and assessment
    st.write(f"**Credit Score: {score}/{max_score} ({percentage:.1f}%)**")
    st.progress(percentage / 100)
    
    # Display assessment
    if score >= 5:
        st.success("✅ **Likely Creditworthy**")
        st.write("Strong candidate for credit approval based on alternative data.")
    elif score >= 3:
        st.warning("⚠️ **Moderate Risk**")
        st.write("May require additional verification or lower credit limit.")
    else:
        st.error("❌ **Higher Risk Profile**")
        st.write("Recommend thorough verification and possibly collateral.")
    
    # Show reasoning
    with st.expander("📋 Assessment Details"):
        st.write(f"- **Age**: {age_feedback}")
        st.write(f"- **Mobile Transactions**: {mobile_feedback}")
        st.write(f"- **Repayment History**: {repayment_feedback}")

# Machine Learning Model Section using only scikit-learn
with st.expander("🤖 Machine Learning Credit Scoring Model"):
    st.write("### Train a Random Forest Classifier")
    
    if st.button("Train Model"):
        with st.spinner("Training model... This may take a few seconds."):
            try:
                # Prepare data
                X = df.drop("Credit_Score", axis=1)
                y = df["Credit_Score"]
                
                # Encode categorical variables
                label_encoders = {}
                for column in X.select_dtypes(include=['object']).columns:
                    le = LabelEncoder()
                    X[column] = le.fit_transform(X[column])
                    label_encoders[column] = le
                
                # Encode target
                target_encoder = LabelEncoder()
                y_encoded = target_encoder.fit_transform(y)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                st.success(f"✅ Model trained successfully!")
                
                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.2%}")
                with col2:
                    st.metric("Training Samples", len(X_train))
                with col3:
                    st.metric("Test Samples", len(X_test))
                
                # Feature importance
                st.write("### Feature Importance")
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.dataframe(feature_importance, hide_index=True)
                
                # Make prediction for user input
                st.write("### Predict Your Credit Score")
                if st.button("Predict with Current Inputs"):
                    # Prepare user input
                    user_data = pd.DataFrame({
                        'Location': [Location],
                        'Gender': [gender],
                        'Mobile_Money_Txns': [Mobile_Money_Txns],
                        'Airtime_Spend_ZWL': [Airtime_Spend_ZWL],
                        'Utility_Payments_ZWL': [Utility_Payments_ZWL],
                        'Loan_Repayment_History': [Loan_Repayment_History],
                        'Age': [Age]
                    })
                    
                    # Encode user input
                    for column in user_data.select_dtypes(include=['object']).columns:
                        if column in label_encoders:
                            # Handle unseen labels
                            if user_data[column].iloc[0] in label_encoders[column].classes_:
                                user_data[column] = label_encoders[column].transform(user_data[column])
                            else:
                                user_data[column] = -1  # Handle unseen labels
                    
                    # Scale and predict
                    user_data_scaled = scaler.transform(user_data)
                    prediction_encoded = model.predict(user_data_scaled)
                    prediction_proba = model.predict_proba(user_data_scaled)
                    
                    # Decode prediction
                    predicted_class = target_encoder.inverse_transform(prediction_encoded)[0]
                    confidence = np.max(prediction_proba) * 100
                    
                    st.success(f"**Predicted Credit Score: {predicted_class}**")
                    st.info(f"**Confidence: {confidence:.1f}%**")
                    
                    # Show probabilities for all classes
                    prob_df = pd.DataFrame({
                        'Credit Score': target_encoder.classes_,
                        'Probability': prediction_proba[0] * 100
                    }).sort_values('Probability', ascending=False)
                    
                    st.write("**Probability Distribution:**")
                    st.dataframe(prob_df, hide_index=True)
                
            except Exception as e:
                st.error(f"Error training model: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit & Scikit-learn | Smart Credit Assessment for Zimbabwe</p>
</div>
""", unsafe_allow_html=True)
