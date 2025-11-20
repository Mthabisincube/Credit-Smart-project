import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# Page configuration with beautiful theme
st.set_page_config(
    page_title="Smart Credit App",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .feature-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #b8d4f0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-header">💳 Smart Credit App</h1>', unsafe_allow_html=True)
    st.markdown("### Revolutionizing Credit Scoring with Alternative Data")
    st.markdown("---")

# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/Mthabisincube/Credit-Smart-project/refs/heads/master/smart_credit_scoring_zimbabwe.csv")
    return df

df = load_data()

# Beautiful sidebar with gradient background
with st.sidebar:
    st.markdown("""
    <style>
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    <div class="sidebar-header">
        <h2>🔮 Credit Assessment</h2>
        <p>Enter your details below</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📋 Personal Information")
    
    col1, col2 = st.columns(2)
    with col1:
        Location = st.selectbox(
            "📍 Location", 
            sorted(df['Location'].unique()),
            help="Select your current location"
        )
    with col2:
        gender = st.selectbox(
            "👤 Gender", 
            sorted(df['Gender'].unique()),
            help="Select your gender"
        )
    
    Age = st.slider(
        "🎂 Age", 
        int(df['Age'].min()), 
        int(df['Age'].max()), 
        int(df['Age'].mean()),
        help="Select your age"
    )
    
    st.markdown("### 💰 Financial Behavior")
    
    Mobile_Money_Txns = st.slider(
        "📱 Mobile Money Transactions", 
        float(df['Mobile_Money_Txns'].min()), 
        float(df['Mobile_Money_Txns'].max()), 
        float(df['Mobile_Money_Txns'].mean()),
        help="Average monthly mobile money transactions"
    )
    
    Airtime_Spend_ZWL = st.slider(
        "📞 Airtime Spend (ZWL)", 
        float(df['Airtime_Spend_ZWL'].min()), 
        float(df['Airtime_Spend_ZWL'].max()), 
        float(df['Airtime_Spend_ZWL'].mean()),
        help="Monthly airtime expenditure in ZWL"
    )
    
    Utility_Payments_ZWL = st.slider(
        "💡 Utility Payments (ZWL)", 
        float(df['Utility_Payments_ZWL'].min()), 
        float(df['Utility_Payments_ZWL'].max()), 
        float(df['Utility_Payments_ZWL'].mean()),
        help="Monthly utility payments in ZWL"
    )
    
    Loan_Repayment_History = st.selectbox(
        "📊 Loan Repayment History", 
        sorted(df['Loan_Repayment_History'].unique()),
        help="Select your loan repayment history"
    )

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔍 Analysis", "🎯 Assessment", "🤖 AI Model"])

with tab1:
    st.markdown('<div class="sub-header">Dataset Overview</div>', unsafe_allow_html=True)
    
    # Metrics in a beautiful grid
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">Total Records<br><h2>{}</h2></div>'.format(len(df)), unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">Features<br><h2>{}</h2></div>'.format(len(df.columns) - 1), unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">Credit Classes<br><h2>{}</h2></div>'.format(df['Credit_Score'].nunique()), unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">Data Quality<br><h2>100%</h2></div>'.format(df.isnull().sum().sum()), unsafe_allow_html=True)
    
    # Data preview with expandable sections
    with st.expander("📋 View Raw Data", expanded=False):
        st.dataframe(df, use_container_width=True, height=300)
    
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🔧 Features (X)", expanded=False):
            X = df.drop("Credit_Score", axis=1)
            st.dataframe(X, use_container_width=True, height=250)
    with col2:
        with st.expander("🎯 Target (Y)", expanded=False):
            Y = df["Credit_Score"]
            st.dataframe(Y, use_container_width=True, height=250)

with tab2:
    st.markdown('<div class="sub-header">Data Analysis & Insights</div>', unsafe_allow_html=True)
    
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["📈 Distributions", "🔢 Statistics", "🌍 Geographic"])
    
    with analysis_tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Credit Score Distribution")
            score_counts = df['Credit_Score'].value_counts().sort_index()
            st.bar_chart(score_counts, use_container_width=True)
        with col2:
            st.markdown("#### Location Distribution")
            location_counts = df['Location'].value_counts()
            st.bar_chart(location_counts, use_container_width=True)
    
    with analysis_tab2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        selected_feature = st.selectbox("Select feature for detailed analysis:", numeric_cols)
        
        if selected_feature:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"#### {selected_feature} Distribution")
                hist_values = np.histogram(df[selected_feature], bins=20)[0]
                st.bar_chart(hist_values, use_container_width=True)
            
            with col2:
                st.markdown(f"#### Statistics for {selected_feature}")
                stats = {
                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th %ile', '75th %ile'],
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
                stats_df = pd.DataFrame(stats)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with analysis_tab3:
        st.markdown("#### Credit Scores by Location")
        location_summary = df.groupby('Location')['Credit_Score'].value_counts().unstack().fillna(0)
        st.dataframe(location_summary.style.background_gradient(cmap='Blues'), use_container_width=True)

with tab3:
    st.markdown('<div class="sub-header">Credit Assessment Results</div>', unsafe_allow_html=True)
    
    # Input summary in a beautiful card
    st.markdown("### 📋 Your Input Summary")
    input_data = {
        "Feature": ["Location", "Gender", "Age", "Mobile Transactions", "Airtime Spend", "Utility Payments", "Repayment History"],
        "Value": [Location, gender, Age, f"{Mobile_Money_Txns:.1f}", f"{Airtime_Spend_ZWL:.1f} ZWL", 
                 f"{Utility_Payments_ZWL:.1f} ZWL", Loan_Repayment_History]
    }
    input_df = pd.DataFrame(input_data)
    st.dataframe(input_df, use_container_width=True, hide_index=True)
    
    # Assessment calculation
    st.markdown("### 🎯 Credit Assessment")
    
    # Scoring logic with beautiful progress bars
    score = 0
    max_score = 6
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Age Factor")
        if 30 <= Age <= 50:
            score += 2
            st.success("✅ Optimal (30-50)")
            st.progress(1.0)
        elif 25 <= Age < 30 or 50 < Age <= 60:
            score += 1
            st.warning("⚠️ Moderate")
            st.progress(0.5)
        else:
            st.error("❌ Higher Risk")
            st.progress(0.2)
    
    with col2:
        st.markdown("#### Transaction Activity")
        mobile_median = df['Mobile_Money_Txns'].median()
        if Mobile_Money_Txns > mobile_median:
            score += 1
            st.success(f"✅ Above Avg ({mobile_median:.1f})")
            st.progress(1.0)
        else:
            st.warning("⚠️ Below Average")
            st.progress(0.3)
    
    with col3:
        st.markdown("#### Repayment History")
        repayment_scores = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
        rep_score = repayment_scores[Loan_Repayment_History]
        score += rep_score
        progress_map = {'Poor': 0.2, 'Fair': 0.4, 'Good': 0.7, 'Excellent': 1.0}
        st.info(f"📊 {Loan_Repayment_History}")
        st.progress(progress_map[Loan_Repayment_History])
    
    # Final assessment with beautiful styling
    st.markdown("---")
    percentage = (score / max_score) * 100
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"### Overall Score: {score}/{max_score}")
        st.markdown(f"### {percentage:.1f}%")
        st.progress(percentage / 100)
    
    with col2:
        if score >= 5:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("### ✅ **EXCELLENT CREDITWORTHINESS**")
            st.markdown("**Recommendation:** Strong candidate for credit approval with favorable terms")
            st.markdown('</div>', unsafe_allow_html=True)
        elif score >= 3:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown("### ⚠️ **MODERATE RISK PROFILE**")
            st.markdown("**Recommendation:** Standard verification with moderate credit limits")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="danger-box">', unsafe_allow_html=True)
            st.markdown("### ❌ **HIGHER RISK PROFILE**")
            st.markdown("**Recommendation:** Enhanced verification and possible collateral required")
            st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="sub-header">AI-Powered Credit Scoring</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    <h3>🚀 Machine Learning Model</h3>
    <p>Our advanced Random Forest classifier analyzes patterns in your financial behavior to predict creditworthiness with high accuracy.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🎯 Train AI Model", type="primary", use_container_width=True):
        with st.spinner("🤖 Training model... This may take a few moments."):
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
                
                # Split and scale data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                )
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                st.success("✅ Model trained successfully!")
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.2%}")
                with col2:
                    st.metric("Training Samples", len(X_train))
                with col3:
                    st.metric("Test Samples", len(X_test))
                with col4:
                    st.metric("Features Used", len(X.columns))
                
                # Feature importance
                st.markdown("### 🔍 Feature Importance")
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Display as bar chart
                st.bar_chart(feature_importance.set_index('Feature')['Importance'])
                
                # Real-time prediction
                st.markdown("### 🎯 Predict Your Credit Score")
                
                if st.button("🔮 Get AI Prediction", type="secondary", use_container_width=True):
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
                            if user_data[column].iloc[0] in label_encoders[column].classes_:
                                user_data[column] = label_encoders[column].transform(user_data[column])
                            else:
                                user_data[column] = -1
                    
                    # Scale and predict
                    user_data_scaled = scaler.transform(user_data)
                    prediction_encoded = model.predict(user_data_scaled)
                    prediction_proba = model.predict_proba(user_data_scaled)
                    
                    predicted_class = target_encoder.inverse_transform(prediction_encoded)[0]
                    confidence = np.max(prediction_proba) * 100
                    
                    # Beautiful prediction display
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f'<div class="success-box"><h3>AI Prediction: {predicted_class}</h3></div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'<div class="card"><h3>Confidence: {confidence:.1f}%</h3></div>', unsafe_allow_html=True)
                    
                    # Probability distribution
                    st.markdown("#### Probability Distribution")
                    prob_df = pd.DataFrame({
                        'Credit Score': target_encoder.classes_,
                        'Probability (%)': (prediction_proba[0] * 100).round(2)
                    }).sort_values('Probability (%)', ascending=False)
                    
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"❌ Error training model: {str(e)}")
