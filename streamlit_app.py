import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
from datetime import datetime
import json

# Page configuration with beautiful theme
st.set_page_config(
    page_title="Zim Smart Credit App",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling with background image
st.markdown("""
<style>
    .stApp {
        background-image: linear-gradient(rgba(255,255,255,0.92), rgba(255,255,255,0.92)), 
                          url('https://images.unsplash.com/photo-1554224155-6726b3ff858f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1911&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }
    
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        padding: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .report-card {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 2px solid #e9ecef;
    }
    
    .download-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem;
    }
    
    .model-stats {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    
    .card {
        background-color: rgba(248, 249, 250, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        backdrop-filter: blur(5px);
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(212, 237, 218, 0.95) 0%, rgba(195, 230, 203, 0.95) 100%);
        border: 2px solid #28a745;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #155724;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 243, 205, 0.95) 0%, rgba(255, 234, 167, 0.95) 100%);
        border: 2px solid #ffc107;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #856404;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .danger-box {
        background: linear-gradient(135deg, rgba(248, 215, 218, 0.95) 0%, rgba(245, 198, 203, 0.95) 100%);
        border: 2px solid #dc3545;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #721c24;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .tab-content {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<h1 class="main-header glowing-text">🏦 Zim Smart Credit App</h1>', unsafe_allow_html=True)
st.markdown("### 💳 Revolutionizing Credit Scoring with Alternative Data in Zimbabwe")
st.markdown("---")

# Global variables for model and encoders
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.label_encoders = {}
    st.session_state.target_encoder = None
    st.session_state.model_metrics = {}
    st.session_state.model_trained = False

# Assessment results in session state
if 'assessment_results' not in st.session_state:
    st.session_state.assessment_results = {
        'score': 0,
        'max_score': 6,
        'predicted_class': None,
        'confidence': None,
        'probabilities': None,
        'risk_level': 'Medium'
    }

# Load data with caching
@st.cache_data
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/Mthabisincube/Credit-Smart-project/refs/heads/master/smart_credit_scoring_zimbabwe.csv")

df = load_data()

# Helper functions
def get_risk_level(score):
    """Get risk level based on score"""
    if score >= 5:
        return "Low"
    elif score >= 3:
        return "Medium"
    else:
        return "High"

def get_recommendations(score, ai_prediction=None):
    """Generate recommendations based on score and AI prediction"""
    recommendations = []
    
    if score >= 5:
        recommendations.append("✓ Strong candidate for credit approval")
        recommendations.append("✓ Eligible for higher credit limits")
        recommendations.append("✓ Favorable interest rates applicable")
    elif score >= 3:
        recommendations.append("✓ Standard credit verification required")
        recommendations.append("✓ Moderate credit limits recommended")
        recommendations.append("✓ Regular monitoring suggested")
    else:
        recommendations.append("✗ Enhanced verification required")
        recommendations.append("✗ Collateral might be necessary")
        recommendations.append("✗ Lower credit limits recommended")
    
    if ai_prediction and ai_prediction in ['Good', 'Excellent']:
        recommendations.append("✓ AI model confirms creditworthiness")
    elif ai_prediction and ai_prediction in ['Poor', 'Fair']:
        recommendations.append("⚠ AI model suggests careful review")
    
    return "\n".join(recommendations)

def generate_pdf_report():
    """Generate a PDF report (simulated)"""
    results = st.session_state.assessment_results
    
    # Format confidence if available
    confidence_display = f"{results['confidence']:.1f}%" if results['confidence'] else 'Not available'
    
    report = f"""
    ZIM SMART CREDIT APP - CREDIT ASSESSMENT REPORT
    Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    ===================================================
    
    I. CLIENT INFORMATION
    - Location: {Location}
    - Gender: {gender}
    - Age: {Age}
    
    II. FINANCIAL BEHAVIOR
    - Mobile Money Transactions: {Mobile_Money_Txns:.2f}
    - Airtime Spend: {Airtime_Spend_ZWL:.2f} ZWL
    - Utility Payments: {Utility_Payments_ZWL:.2f} ZWL
    - Loan Repayment History: {Loan_Repayment_History}
    
    III. CREDIT ASSESSMENT
    - Manual Assessment Score: {results['score']}/{results['max_score']}
    - Risk Level: {results['risk_level']}
    - AI Prediction: {results['predicted_class'] if results['predicted_class'] else 'Not available'}
    - Confidence Level: {confidence_display}
    
    IV. MODEL PERFORMANCE METRICS
    - Accuracy: {st.session_state.model_metrics.get('accuracy', 0)*100:.1f}%
    - Precision: {st.session_state.model_metrics.get('precision', 0)*100:.1f}%
    - Recall: {st.session_state.model_metrics.get('recall', 0)*100:.1f}%
    - F1-Score: {st.session_state.model_metrics.get('f1_score', 0)*100:.1f}%
    
    V. RECOMMENDATIONS
    {get_recommendations(results['score'], results['predicted_class'])}
    
    ===================================================
    
    This report is generated by Zim Smart Credit App.
    For inquiries, contact: support@zimcredit.co.zw
    """
    return report.encode()

def generate_csv_report():
    """Generate CSV report"""
    results = st.session_state.assessment_results
    
    report_data = {
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'location': [Location],
        'gender': [gender],
        'age': [Age],
        'mobile_money_txns': [Mobile_Money_Txns],
        'airtime_spend': [Airtime_Spend_ZWL],
        'utility_payments': [Utility_Payments_ZWL],
        'repayment_history': [Loan_Repayment_History],
        'manual_score': [f"{results['score']}/{results['max_score']}"],
        'risk_level': [results['risk_level']],
        'ai_prediction': [results['predicted_class'] if results['predicted_class'] else 'N/A'],
        'confidence': [f"{results['confidence']:.1f}%" if results['confidence'] else 'N/A'],
        'model_accuracy': [f"{st.session_state.model_metrics.get('accuracy', 0)*100:.1f}%"]
    }
    df_report = pd.DataFrame(report_data)
    return df_report.to_csv(index=False).encode('utf-8')

# Train unified model function
def train_unified_model():
    """Train the unified ML model and store it in session state"""
    with st.spinner("🤖 Training unified model... This may take a few moments."):
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
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X, y_encoded, cv=5)
            cv_mean = cv_scores.mean()
            
            # Store in session state
            st.session_state.model = model
            st.session_state.label_encoders = label_encoders
            st.session_state.target_encoder = target_encoder
            st.session_state.model_trained = True
            
            # Store metrics
            st.session_state.model_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_mean,
                'cv_scores': cv_scores,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'feature_importance': dict(zip(X.columns, model.feature_importances_)),
                'test_size': len(X_test),
                'train_size': len(X_train)
            }
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error training model: {str(e)}")
            return False

# Prediction function using unified model
def predict_credit_score(user_input):
    """Predict credit score using the unified model"""
    if not st.session_state.model_trained:
        return None, None, None
    
    try:
        # Prepare user data
        user_data = pd.DataFrame([user_input])
        
        # Encode categorical features
        for column in user_data.select_dtypes(include=['object']).columns:
            if column in st.session_state.label_encoders:
                le = st.session_state.label_encoders[column]
                if user_data[column].iloc[0] in le.classes_:
                    user_data[column] = le.transform(user_data[column])
                else:
                    user_data[column] = -1
        
        # Predict
        prediction_encoded = st.session_state.model.predict(user_data)
        prediction_proba = st.session_state.model.predict_proba(user_data)
        
        predicted_class = st.session_state.target_encoder.inverse_transform(prediction_encoded)[0]
        confidence = np.max(prediction_proba) * 100
        
        return predicted_class, confidence, prediction_proba[0]
        
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        return None, None, None

# Beautiful sidebar
with st.sidebar:
    st.markdown("""
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
            sorted(df['Location'].unique())
        )
    with col2:
        gender = st.selectbox(
            "👤 Gender", 
            sorted(df['Gender'].unique())
        )
    
    Age = st.slider(
        "🎂 Age", 
        int(df['Age'].min()), 
        int(df['Age'].max()), 
        int(df['Age'].mean())
    )
    
    st.markdown("### 💰 Financial Behavior")
    
    Mobile_Money_Txns = st.slider(
        "📱 Mobile Money Transactions", 
        float(df['Mobile_Money_Txns'].min()), 
        float(df['Mobile_Money_Txns'].max()), 
        float(df['Mobile_Money_Txns'].mean())
    )
    
    Airtime_Spend_ZWL = st.slider(
        "📞 Airtime Spend (ZWL)", 
        float(df['Airtime_Spend_ZWL'].min()), 
        float(df['Airtime_Spend_ZWL'].max()), 
        float(df['Airtime_Spend_ZWL'].mean())
    )
    
    Utility_Payments_ZWL = st.slider(
        "💡 Utility Payments (ZWL)", 
        float(df['Utility_Payments_ZWL'].min()), 
        float(df['Utility_Payments_ZWL'].max()), 
        float(df['Utility_Payments_ZWL'].mean())
    )
    
    Loan_Repayment_History = st.selectbox(
        "📊 Loan Repayment History", 
        sorted(df['Loan_Repayment_History'].unique())
    )
    
    # Unified model training button in sidebar
    st.markdown("---")
    if st.button("🚀 Train Unified Model", type="primary", use_container_width=True):
        if train_unified_model():
            st.success("✅ Model trained successfully!")
            st.rerun()

# Main content with tabs - ADDED REPORTS TAB
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🔍 Analysis", "🎯 Assessment", "🤖 AI Model", "📈 Reports"])

with tab1:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("### 📈 Dataset Overview")
    
    # Beautiful metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Total Records</h3>
            <h2>{len(df):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔧 Features</h3>
            <h2>{len(df.columns) - 1}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Credit Classes</h3>
            <h2>{df['Credit_Score'].nunique()}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>✅ Data Quality</h3>
            <h2>100%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Data preview sections
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("📋 Raw Data Preview", expanded=True):
            st.dataframe(df, use_container_width=True, height=300)
    
    with col2:
        with st.expander("🔍 Features & Target", expanded=True):
            st.write("**Features (X):**")
            X = df.drop("Credit_Score", axis=1)
            st.dataframe(X.head(8), use_container_width=True, height=200)
            
            st.write("**Target (Y):**")
            Y = df["Credit_Score"]
            st.dataframe(Y.head(8), use_container_width=True, height=150)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("### 🔍 Data Analysis & Insights")
    
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["📊 Distributions", "📈 Statistics", "🌍 Geographic"])
    
    with analysis_tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Credit Score Distribution")
            score_counts = df['Credit_Score'].value_counts().sort_index()
            st.bar_chart(score_counts)
            
            # Show as table
            st.markdown("**Count by Credit Score:**")
            dist_df = score_counts.reset_index()
            dist_df.columns = ['Credit Score', 'Count']
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### Location Distribution")
            location_counts = df['Location'].value_counts()
            st.bar_chart(location_counts)
            
            st.markdown("**Count by Location:**")
            loc_df = location_counts.reset_index()
            loc_df.columns = ['Location', 'Count']
            st.dataframe(loc_df, use_container_width=True, hide_index=True)
    
    with analysis_tab2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        selected_feature = st.selectbox("Select feature for detailed analysis:", numeric_cols)
        
        if selected_feature:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### {selected_feature} Distribution")
                hist_values = np.histogram(df[selected_feature], bins=20)[0]
                st.bar_chart(hist_values)
            
            with col2:
                st.markdown(f"#### 📊 Statistics for {selected_feature}")
                stats_data = {
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
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with analysis_tab3:
        st.markdown("#### Credit Scores by Location")
        location_summary = df.groupby('Location')['Credit_Score'].value_counts().unstack().fillna(0)
        st.dataframe(location_summary, use_container_width=True)
        
        st.markdown("#### Location Performance Summary")
        location_stats = df.groupby('Location').agg({
            'Credit_Score': lambda x: (x == 'Good').mean()  # Example metric
        }).round(3)
        st.dataframe(location_stats, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("### 🎯 Credit Assessment Results")
    
    # Input summary in beautiful cards
    st.markdown("#### 📋 Your Input Summary")
    input_data = {
        "Feature": ["📍 Location", "👤 Gender", "🎂 Age", "📱 Mobile Transactions", 
                   "📞 Airtime Spend", "💡 Utility Payments", "📊 Repayment History"],
        "Value": [Location, gender, f"{Age} years", f"{Mobile_Money_Txns:.1f}", 
                 f"{Airtime_Spend_ZWL:.1f} ZWL", f"{Utility_Payments_ZWL:.1f} ZWL", Loan_Repayment_History]
    }
    input_df = pd.DataFrame(input_data)
    
    # Display as styled dataframe
    st.dataframe(
        input_df, 
        use_container_width=True, 
        hide_index=True,
        height=280
    )
    
    # Assessment calculation with beautiful progress bars
    st.markdown("#### 📊 Assessment Factors")
    
    score = 0
    max_score = 6
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 🎂 Age Factor")
        if 30 <= Age <= 50:
            score += 2
            st.success("✅ Optimal (30-50 years)")
            st.progress(1.0)
        elif 25 <= Age < 30 or 50 < Age <= 60:
            score += 1
            st.warning("⚠️ Moderate")
            st.progress(0.5)
        else:
            st.error("❌ Higher Risk")
            st.progress(0.2)
    
    with col2:
        st.markdown("##### 💰 Transaction Activity")
        mobile_median = df['Mobile_Money_Txns'].median()
        if Mobile_Money_Txns > mobile_median:
            score += 1
            st.success(f"✅ Above Average")
            st.progress(1.0)
        else:
            st.warning("⚠️ Below Average")
            st.progress(0.3)
    
    with col3:
        st.markdown("##### 📈 Repayment History")
        repayment_scores = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
        rep_score = repayment_scores[Loan_Repayment_History]
        score += rep_score
        progress_map = {'Poor': 0.2, 'Fair': 0.4, 'Good': 0.7, 'Excellent': 1.0}
        st.info(f"📊 {Loan_Repayment_History}")
        st.progress(progress_map[Loan_Repayment_History])
    
    # Store assessment results in session state
    st.session_state.assessment_results['score'] = score
    st.session_state.assessment_results['max_score'] = max_score
    st.session_state.assessment_results['risk_level'] = get_risk_level(score)
    
    # Final assessment
    st.markdown("---")
    percentage = (score / max_score) * 100
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 📈 Overall Score")
        st.markdown(f"# {score}/{max_score}")
        st.markdown(f"### {percentage:.1f}%")
        st.progress(percentage / 100)
        
        # Score interpretation
        if score >= 5:
            st.success("🎉 Excellent Score!")
        elif score >= 3:
            st.info("📊 Good Score")
        else:
            st.warning("📝 Needs Improvement")
    
    with col2:
        st.markdown("#### 🎯 Final Assessment")
        if score >= 5:
            st.markdown("""
            <div class="success-box">
                <h3>✅ EXCELLENT CREDITWORTHINESS</h3>
                <p><strong>Recommendation:</strong> Strong candidate for credit approval with favorable terms and higher limits</p>
                <p><strong>Risk Level:</strong> Low</p>
            </div>
            """, unsafe_allow_html=True)
        elif score >= 3:
            st.markdown("""
            <div class="warning-box">
                <h3>⚠️ MODERATE RISK PROFILE</h3>
                <p><strong>Recommendation:</strong> Standard verification process with moderate credit limits</p>
                <p><strong>Risk Level:</strong> Medium</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="danger-box">
                <h3>❌ HIGHER RISK PROFILE</h3>
                <p><strong>Recommendation:</strong> Enhanced verification and possible collateral required</p>
                <p><strong>Risk Level:</strong> High</p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("### 🤖 AI-Powered Credit Scoring")
    
    st.markdown("""
    <div class="card">
        <h3>🚀 Unified Machine Learning Model</h3>
        <p>Our unified Random Forest classifier analyzes patterns in your financial behavior to predict creditworthiness with high accuracy using alternative data sources.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model training status and metrics
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.model_trained:
            st.success("✅ Model is trained and ready for predictions")
        else:
            st.warning("⚠️ Model not trained yet. Click the button in sidebar to train.")
    
    with col2:
        if st.button("🔄 Refresh Model Status", use_container_width=True):
            st.rerun()
    
    # Display model metrics if trained
    if st.session_state.model_trained:
        st.markdown("#### 📊 Model Performance Metrics")
        
        metrics = st.session_state.model_metrics
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Accuracy</h3>
                <h2>{metrics['accuracy']*100:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📈 Precision</h3>
                <h2>{metrics['precision']*100:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Recall</h3>
                <h2>{metrics['recall']*100:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>⚖️ F1-Score</h3>
                <h2>{metrics['f1_score']*100:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Cross-validation scores
        st.markdown("#### 🔄 Cross-Validation Results")
        cv_df = pd.DataFrame({
            'Fold': [f'Fold {i+1}' for i in range(len(metrics['cv_scores']))],
            'Accuracy': [f"{score*100:.1f}%" for score in metrics['cv_scores']]
        })
        st.dataframe(cv_df, use_container_width=True, hide_index=True)
        
        # Feature importance
        st.markdown("#### 🔍 Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': list(metrics['feature_importance'].keys()),
            'Importance': list(metrics['feature_importance'].values())
        }).sort_values('Importance', ascending=False)
        
        st.bar_chart(feature_importance.set_index('Feature')['Importance'])
        st.dataframe(feature_importance, use_container_width=True, hide_index=True)
        
        # Real-time prediction
        st.markdown("#### 🎯 Get Your AI Prediction")
        
        if st.button("🔮 Predict My Credit Score", type="primary", use_container_width=True):
            user_input = {
                'Location': Location,
                'Gender': gender,
                'Age': Age,
                'Mobile_Money_Txns': Mobile_Money_Txns,
                'Airtime_Spend_ZWL': Airtime_Spend_ZWL,
                'Utility_Payments_ZWL': Utility_Payments_ZWL,
                'Loan_Repayment_History': Loan_Repayment_History
            }
            
            predicted_class, confidence, probabilities = predict_credit_score(user_input)
            
            if predicted_class:
                # Store prediction results in session state
                st.session_state.assessment_results['predicted_class'] = predicted_class
                st.session_state.assessment_results['confidence'] = confidence
                st.session_state.assessment_results['probabilities'] = probabilities
                
                # Beautiful prediction display
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="success-box">
                        <h3>AI Prediction</h3>
                        <h1>{predicted_class}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="card">
                        <h3>Confidence Level</h3>
                        <h1>{confidence:.1f}%</h1>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability distribution
                st.markdown("#### 📊 Probability Distribution")
                classes = st.session_state.target_encoder.classes_
                prob_df = pd.DataFrame({
                    'Credit Score': classes,
                    'Probability (%)': (probabilities * 100).round(2)
                }).sort_values('Probability (%)', ascending=False)
                
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
                
                st.success("✅ Prediction stored for report generation!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# NEW REPORTS TAB
with tab5:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("### 📈 Comprehensive Reports")
    
    st.markdown("""
    <div class="card">
        <h3>📋 Report Generation Center</h3>
        <p>Generate comprehensive reports for your credit assessment. These reports include:</p>
        <ul>
            <li>Client information and financial behavior</li>
            <li>Manual assessment results</li>
            <li>AI prediction and confidence levels</li>
            <li>Model performance metrics</li>
            <li>Personalized recommendations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Report configuration
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "📄 Select Report Type",
            ["Comprehensive Report", "Executive Summary", "Technical Analysis", "Client Brief"]
        )
    
    with col2:
        include_ai = st.checkbox("🤖 Include AI Prediction", 
                                value=st.session_state.assessment_results['predicted_class'] is not None,
                                disabled=st.session_state.assessment_results['predicted_class'] is None)
        include_metrics = st.checkbox("📊 Include Model Metrics", 
                                     value=st.session_state.model_trained,
                                     disabled=not st.session_state.model_trained)
        include_recommendations = st.checkbox("💡 Include Recommendations", value=True)
    
    # Generate report button
    if st.button("📄 Generate Report", type="primary", use_container_width=True):
        # Create report content
        st.markdown("#### 📋 Generated Report Preview")
        
        # Report header
        st.markdown(f"""
        <div class="report-card">
            <h2>ZIM SMART CREDIT APP</h2>
            <h3>{report_type}</h3>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Report ID:</strong> CR-{datetime.now().strftime('%Y%m%d-%H%M%S')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Client Information
        st.markdown("#### 👤 Client Information")
        client_info = pd.DataFrame({
            'Field': ['Location', 'Gender', 'Age'],
            'Value': [Location, gender, f"{Age} years"]
        })
        st.dataframe(client_info, use_container_width=True, hide_index=True)
        
        # Financial Behavior
        st.markdown("#### 💰 Financial Behavior Analysis")
        financial_info = pd.DataFrame({
            'Metric': ['Mobile Money Transactions', 'Airtime Spend (ZWL)', 'Utility Payments (ZWL)', 'Loan Repayment History'],
            'Value': [
                f"{Mobile_Money_Txns:.2f}",
                f"{Airtime_Spend_ZWL:.2f}",
                f"{Utility_Payments_ZWL:.2f}",
                Loan_Repayment_History
            ],
            'Assessment': [
                "Above Average" if Mobile_Money_Txns > df['Mobile_Money_Txns'].median() else "Below Average",
                "High" if Airtime_Spend_ZWL > df['Airtime_Spend_ZWL'].median() else "Low",
                "Consistent" if Utility_Payments_ZWL > df['Utility_Payments_ZWL'].median() else "Irregular",
                Loan_Repayment_History
            ]
        })
        st.dataframe(financial_info, use_container_width=True, hide_index=True)
        
        # Assessment Results
        st.markdown("#### 🎯 Assessment Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="model-stats">
                <h4>Manual Assessment</h4>
                <h2>{st.session_state.assessment_results['score']}/{st.session_state.assessment_results['max_score']}</h2>
                <p>Risk Level: {st.session_state.assessment_results['risk_level']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if include_ai and st.session_state.assessment_results['predicted_class']:
            with col2:
                st.markdown(f"""
                <div class="model-stats">
                    <h4>AI Prediction</h4>
                    <h2>{st.session_state.assessment_results['predicted_class']}</h2>
                    <p>Confidence: {st.session_state.assessment_results['confidence']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Model Metrics
        if include_metrics and st.session_state.model_trained:
            st.markdown("#### 📊 Model Performance")
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Cross-Validation Mean'],
                'Value': [
                    f"{st.session_state.model_metrics['accuracy']*100:.1f}%",
                    f"{st.session_state.model_metrics['precision']*100:.1f}%",
                    f"{st.session_state.model_metrics['recall']*100:.1f}%",
                    f"{st.session_state.model_metrics['f1_score']*100:.1f}%",
                    f"{st.session_state.model_metrics['cv_mean']*100:.1f}%"
                ]
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Recommendations
        if include_recommendations:
            st.markdown("#### 💡 Recommendations")
            recommendations = get_recommendations(
                st.session_state.assessment_results['score'], 
                st.session_state.assessment_results['predicted_class']
            )
            
            for rec in recommendations.split('\n'):
                if rec.startswith('✓'):
                    st.success(rec)
                elif rec.startswith('⚠'):
                    st.warning(rec)
                elif rec.startswith('✗'):
                    st.error(rec)
                else:
                    st.info(rec)
        
        # Download options
        st.markdown("---")
        st.markdown("#### 💾 Download Report")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # PDF Report
            pdf_report = generate_pdf_report()
            st.download_button(
                label="📄 Download PDF",
                data=pdf_report,
                file_name=f"credit_report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            # CSV Report
            csv_report = generate_csv_report()
            st.download_button(
                label="📊 Download CSV",
                data=csv_report,
                file_name=f"credit_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            # JSON Report
            results = st.session_state.assessment_results
            json_report = {
                'timestamp': datetime.now().isoformat(),
                'report_type': report_type,
                'client_info': {
                    'location': Location,
                    'gender': gender,
                    'age': Age
                },
                'financial_behavior': {
                    'mobile_money_txns': Mobile_Money_Txns,
                    'airtime_spend': Airtime_Spend_ZWL,
                    'utility_payments': Utility_Payments_ZWL,
                    'repayment_history': Loan_Repayment_History
                },
                'assessment': {
                    'manual_score': f"{results['score']}/{results['max_score']}",
                    'risk_level': results['risk_level'],
                    'ai_prediction': results['predicted_class'],
                    'confidence': results['confidence'],
                    'probabilities': results['probabilities'].tolist() if results['probabilities'] is not None else None
                },
                'model_metrics': st.session_state.model_metrics if st.session_state.model_trained else None,
                'recommendations': get_recommendations(results['score'], results['predicted_class'])
            }
            
            json_str = json.dumps(json_report, indent=2)
            
            st.download_button(
                label="🔤 Download JSON",
                data=json_str,
                file_name=f"credit_report_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    # Show warning if trying to generate AI report without prediction
    if not st.session_state.assessment_results['predicted_class'] and include_ai:
        st.warning("⚠️ No AI prediction available. Please generate a prediction in the AI Model tab first.")
    
    st.markdown('</div>', unsafe_allow_html=True)
