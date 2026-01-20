import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date
import json
import base64
import io

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
    
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
        font-weight: 600;
        text-align: center;
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
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
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
    
    .report-box {
        background: linear-gradient(135deg, rgba(220, 237, 255, 0.95) 0%, rgba(195, 220, 255, 0.95) 100%);
        border: 3px solid #1f77b4;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #0d47a1;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
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
    
    .accuracy-display {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin: 20px 0;
    }
    
    .accuracy-label {
        font-size: 16px;
        color: #666;
        text-align: center;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== ASSESSMENT HISTORY MANAGEMENT ====================

def initialize_assessment_history():
    """Initialize session state for storing assessment history"""
    if 'assessment_history' not in st.session_state:
        st.session_state.assessment_history = []
    if 'current_assessment' not in st.session_state:
        st.session_state.current_assessment = None
    # Initialize model storage
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'label_encoders' not in st.session_state:
        st.session_state.label_encoders = None
    if 'target_encoder' not in st.session_state:
        st.session_state.target_encoder = None
    if 'model_accuracy' not in st.session_state:
        st.session_state.model_accuracy = None

def save_assessment(assessment_data):
    """Save an assessment to history"""
    assessment_data['timestamp'] = datetime.now().isoformat()
    assessment_data['date'] = date.today().isoformat()
    st.session_state.assessment_history.append(assessment_data)
    st.session_state.current_assessment = assessment_data
    
    # Keep only last 100 assessments to prevent memory issues
    if len(st.session_state.assessment_history) > 100:
        st.session_state.assessment_history = st.session_state.assessment_history[-100:]

def get_last_30_days_assessments():
    """Get assessments from the last 30 days"""
    thirty_days_ago = datetime.now() - timedelta(days=30)
    
    recent_assessments = []
    for assessment in st.session_state.assessment_history:
        assessment_date = datetime.fromisoformat(assessment['timestamp'])
        if assessment_date >= thirty_days_ago:
            recent_assessments.append(assessment)
    
    return recent_assessments

def calculate_30_day_summary():
    """Calculate summary statistics for last 30 days"""
    recent_assessments = get_last_30_days_assessments()
    
    if not recent_assessments:
        return None
    
    # Convert to DataFrame for easier analysis
    df_assessments = pd.DataFrame(recent_assessments)
    
    # Calculate summary statistics
    summary = {
        'total_assessments': len(recent_assessments),
        'average_score': df_assessments['final_score'].mean() if 'final_score' in df_assessments.columns else 0,
        'risk_distribution': df_assessments['risk_level'].value_counts().to_dict() if 'risk_level' in df_assessments.columns else {},
        'date_range': {
            'start': min(df_assessments['date']) if 'date' in df_assessments.columns else '',
            'end': max(df_assessments['date']) if 'date' in df_assessments.columns else ''
        }
    }
    
    return summary

# ==================== MODEL EVALUATION FUNCTIONS ====================

def evaluate_random_forest(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=10):
    """Evaluate Random Forest model and return metrics"""
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'feature_importance': feature_importance
    }

# ==================== MAIN APP CODE ====================

# Initialize assessment history
initialize_assessment_history()

# Header Section
st.markdown('<h1 class="main-header glowing-text">🏦 Zim Smart Credit App</h1>', unsafe_allow_html=True)
st.markdown("### 💳 AI-Powered Credit Scoring with Random Forest")
st.markdown("---")

# Load data with caching
@st.cache_data
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/Mthabisincube/Credit-Smart-project/refs/heads/master/smart_credit_scoring_zimbabwe.csv")

df = load_data()

# Beautiful sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2>🔮 AI Credit Assessment</h2>
        <p>Enter your details for AI analysis</p>
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
    
    # User name for reporting
    st.markdown("---")
    user_name = st.text_input("👤 Your Name (for reports)", "Valued Customer")

# Main content with tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🎯 AI Assessment", "🤖 Model Accuracy", "📈 Reports"])

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
        recent_assessments = get_last_30_days_assessments()
        st.markdown(f"""
        <div class="metric-card">
            <h3>📅 Recent Assessments</h3>
            <h2>{len(recent_assessments)}</h2>
            <small>(Last 30 days)</small>
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
    st.markdown("""
    <div class="card">
        <h2>🎯 AI Credit Assessment</h2>
        <p>Get instant credit scoring powered by Random Forest</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input summary
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
    
    # Check if model has been trained
    if 'trained_model' in st.session_state and 'label_encoders' in st.session_state and 'target_encoder' in st.session_state:
        st.markdown("#### 🤖 AI Assessment in Progress...")
        
        with st.spinner("Analyzing your financial profile with AI..."):
            try:
                # Prepare user data for prediction
                user_data = pd.DataFrame({
                    'Location': [Location],
                    'Gender': [gender],
                    'Age': [Age],
                    'Mobile_Money_Txns': [Mobile_Money_Txns],
                    'Airtime_Spend_ZWL': [Airtime_Spend_ZWL],
                    'Utility_Payments_ZWL': [Utility_Payments_ZWL],
                    'Loan_Repayment_History': [Loan_Repayment_History]
                })
                
                # Encode user input using saved encoders
                for column in user_data.select_dtypes(include=['object']).columns:
                    if column in st.session_state.label_encoders:
                        le = st.session_state.label_encoders[column]
                        if user_data[column].iloc[0] in le.classes_:
                            user_data[column] = le.transform(user_data[column])
                        else:
                            # Handle unseen labels
                            user_data[column] = -1
                
                # Predict with trained model
                model = st.session_state.trained_model
                prediction_encoded = model.predict(user_data)
                prediction_proba = model.predict_proba(user_data)
                
                predicted_class = st.session_state.target_encoder.inverse_transform(prediction_encoded)[0]
                confidence = np.max(prediction_proba) * 100
                
                # Map prediction to risk levels
                risk_mapping = {
                    'Good': 'Low',
                    'Fair': 'Medium', 
                    'Poor': 'High'
                }
                
                risk_level = risk_mapping.get(predicted_class, 'Medium')
                score_percentage = confidence
                
                # Display AI assessment results
                st.markdown("---")
                st.markdown("#### 📊 AI Assessment Results")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("#### 🎯 AI Prediction")
                    st.markdown(f"# {predicted_class}")
                    st.markdown(f"### {confidence:.1f}% Confidence")
                    st.progress(confidence / 100)
                    
                    # Score interpretation
                    if predicted_class == 'Good':
                        st.success("✅ Excellent Credit Score!")
                    elif predicted_class == 'Fair':
                        st.info("📊 Moderate Credit Score")
                    else:
                        st.warning("📝 Needs Improvement")
                
                with col2:
                    # Display appropriate risk box
                    if predicted_class == 'Good':
                        st.markdown(f"""
                        <div class="success-box">
                            <h3>✅ EXCELLENT CREDITWORTHINESS</h3>
                            <p><strong>Recommendation:</strong> Strong candidate for credit approval with favorable terms and higher limits</p>
                            <p><strong>Risk Level:</strong> Low</p>
                            <p><strong>AI Confidence:</strong> {confidence:.1f}%</p>
                            <p><strong>Model Accuracy:</strong> {st.session_state.model_accuracy:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif predicted_class == 'Fair':
                        st.markdown(f"""
                        <div class="warning-box">
                            <h3>⚠️ MODERATE RISK PROFILE</h3>
                            <p><strong>Recommendation:</strong> Standard verification process with moderate credit limits</p>
                            <p><strong>Risk Level:</strong> Medium</p>
                            <p><strong>AI Confidence:</strong> {confidence:.1f}%</p>
                            <p><strong>Model Accuracy:</strong> {st.session_state.model_accuracy:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="danger-box">
                            <h3>❌ HIGHER RISK PROFILE</h3>
                            <p><strong>Recommendation:</strong> Enhanced verification and possible collateral required</p>
                            <p><strong>Risk Level:</strong> High</p>
                            <p><strong>AI Confidence:</strong> {confidence:.1f}%</p>
                            <p><strong>Model Accuracy:</strong> {st.session_state.model_accuracy:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Probability distribution
                st.markdown("#### 📊 AI Probability Distribution")
                prob_df = pd.DataFrame({
                    'Credit Score': st.session_state.target_encoder.classes_,
                    'Probability (%)': (prediction_proba[0] * 100).round(2)
                }).sort_values('Probability (%)', ascending=False)
                
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
                
                # Save assessment button
                st.markdown("---")
                if st.button("💾 Save AI Assessment", type="primary", use_container_width=True):
                    assessment_data = {
                        'user_name': user_name,
                        'location': Location,
                        'gender': gender,
                        'age': Age,
                        'mobile_money_txns': Mobile_Money_Txns,
                        'airtime_spend': Airtime_Spend_ZWL,
                        'utility_payments': Utility_Payments_ZWL,
                        'repayment_history': Loan_Repayment_History,
                        'final_score': score_percentage,
                        'risk_level': risk_level,
                        'predicted_class': predicted_class,
                        'ai_confidence': confidence,
                        'model_accuracy': st.session_state.model_accuracy,
                        'assessment_type': 'AI-Powered'
                    }
                    
                    save_assessment(assessment_data)
                    st.success(f"✅ AI assessment saved successfully! Total assessments: {len(st.session_state.assessment_history)}")
                
            except Exception as e:
                st.error(f"❌ Error in AI assessment: {str(e)}")
                st.info("Please ensure the AI model has been evaluated in the Model Accuracy tab first.")
    else:
        st.error("""
        ⚠️ **AI Model Not Ready**
        
        To use AI-powered credit assessment:
        
        1. Go to the **🤖 Model Accuracy** tab
        2. Click **"Evaluate Random Forest Model"**
        3. Once evaluated, return here for AI-powered predictions
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:  # Model Accuracy tab
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("### 🤖 Random Forest Model Accuracy")
    
    # Model explanation
    st.markdown("""
    <div class="card">
        <h4>🌳 About Random Forest Model</h4>
        <p>This is the machine learning model used for credit scoring in this app.</p>
        <p><strong>Model Type:</strong> Random Forest Classifier</p>
        <p><strong>Algorithm:</strong> Ensemble of decision trees</p>
        <p><strong>Purpose:</strong> Predicts credit score (Good/Fair/Poor) based on financial behavior</p>
        <p><em>Click the button below to evaluate the model's accuracy</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model parameters
    st.markdown("#### ⚙️ Model Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider("Number of Trees", 10, 500, 100, 10,
                                help="Number of decision trees in the forest")
    
    with col2:
        max_depth = st.slider("Max Tree Depth", 2, 20, 10, 1,
                             help="Maximum depth of each decision tree")
    
    # Test size
    test_size = st.slider("Test Size %", 10, 40, 20, 5,
                         help="Percentage of data to use for testing")
    
    # Button to evaluate model
    if st.button("📊 Evaluate Random Forest Model", type="primary", use_container_width=True):
        with st.spinner("🤖 Evaluating Random Forest model..."):
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
                    X, y_encoded, 
                    test_size=test_size/100, 
                    random_state=42, 
                    stratify=y_encoded
                )
                
                # Evaluate model
                results = evaluate_random_forest(X_train, X_test, y_train, y_test, n_estimators, max_depth)
                
                # Save model and results to session state
                st.session_state.trained_model = results['model']
                st.session_state.label_encoders = label_encoders
                st.session_state.target_encoder = target_encoder
                st.session_state.model_accuracy = results['accuracy']
                
                st.success("✅ Random Forest model evaluated successfully!")
                
                # ============= DISPLAY ACCURACY RESULTS =============
                st.markdown("---")
                st.markdown("### 🎯 Model Accuracy Results")
                
                # Display accuracy percentage in a big, clear way
                accuracy_percentage = results['accuracy'] * 100
                
                st.markdown(f"""
                <div style="text-align: center; margin: 30px 0;">
                    <div class="accuracy-label">RANDOM FOREST MODEL ACCURACY</div>
                    <div class="accuracy-display">{results['accuracy']:.1%}</div>
                    <div style="color: #666; font-size: 18px;">
                        {accuracy_percentage:.1f} out of 100 correct predictions
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Progress bar for accuracy
                st.progress(results['accuracy'])
                
                # Interpretation
                if results['accuracy'] >= 0.85:
                    st.success(f"**Excellent Performance**: {results['accuracy']:.1%} accuracy exceeds industry standards for credit scoring.")
                elif results['accuracy'] >= 0.75:
                    st.warning(f"**Good Performance**: {results['accuracy']:.1%} accuracy is acceptable for deployment.")
                else:
                    st.error(f"**Needs Improvement**: {results['accuracy']:.1%} accuracy suggests the model needs optimization.")
                
                # ============= DETAILED METRICS =============
                st.markdown("---")
                st.markdown("#### 📊 Detailed Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Precision", f"{results['precision']:.1%}")
                
                with col2:
                    st.metric("Recall", f"{results['recall']:.1%}")
                
                with col3:
                    st.metric("F1-Score", f"{results['f1']:.1%}")
                
                with col4:
                    st.metric("Cross-Validation", f"{results['cv_mean']:.1%}")
                
                # ============= FEATURE IMPORTANCE =============
                st.markdown("---")
                st.markdown("#### 🔍 Feature Importance")
                
                feature_importance = results['feature_importance']
                
                # Create visualization
                fig = go.Figure(data=[
                    go.Bar(
                        x=feature_importance['Importance'],
                        y=feature_importance['Feature'],
                        orientation='h',
                        marker_color='lightblue'
                    )
                ])
                
                fig.update_layout(
                    title='Feature Importance Scores',
                    xaxis_title='Importance',
                    yaxis_title='Feature',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display as table
                st.dataframe(feature_importance, use_container_width=True, hide_index=True)
                
                # ============= MODEL TESTING =============
                st.markdown("---")
                st.markdown("#### 🎯 Test Model with Your Data")
                
                if st.button("🔮 Test Model Prediction", type="secondary", use_container_width=True):
                    user_data = pd.DataFrame({
                        'Location': [Location],
                        'Gender': [gender],
                        'Age': [Age],
                        'Mobile_Money_Txns': [Mobile_Money_Txns],
                        'Airtime_Spend_ZWL': [Airtime_Spend_ZWL],
                        'Utility_Payments_ZWL': [Utility_Payments_ZWL],
                        'Loan_Repayment_History': [Loan_Repayment_History]
                    })
                    
                    # Encode user input
                    for column in user_data.select_dtypes(include=['object']).columns:
                        if column in label_encoders:
                            le = label_encoders[column]
                            if user_data[column].iloc[0] in le.classes_:
                                user_data[column] = le.transform(user_data[column])
                            else:
                                user_data[column] = -1
                    
                    # Predict
                    prediction_encoded = results['model'].predict(user_data)
                    prediction_proba = results['model'].predict_proba(user_data)
                    
                    predicted_class = target_encoder.inverse_transform(prediction_encoded)[0]
                    confidence = np.max(prediction_proba) * 100
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="success-box">
                            <h3>Test Result</h3>
                            <h1>{predicted_class}</h1>
                            <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                            <p><strong>Model Accuracy:</strong> {results['accuracy']:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("#### 📊 Probability Distribution")
                        prob_df = pd.DataFrame({
                            'Credit Score': target_encoder.classes_,
                            'Probability (%)': (prediction_proba[0] * 100).round(2)
                        }).sort_values('Probability (%)', ascending=False)
                        st.dataframe(prob_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"❌ Error evaluating model: {str(e)}")
    
    # If model is already evaluated
    elif 'model_accuracy' in st.session_state and st.session_state.model_accuracy is not None:
        st.markdown("---")
        st.markdown(f"""
        <div class="success-box">
            <h3>✅ Model Already Evaluated</h3>
            <p><strong>Model Accuracy:</strong> {st.session_state.model_accuracy:.1%}</p>
            <p><strong>Model Type:</strong> Random Forest</p>
            <p><em>This model is ready for use in the AI Assessment tab.</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show accuracy percentage
        accuracy_percentage = st.session_state.model_accuracy * 100
        
        st.markdown(f"""
        <div style="text-align: center; margin: 30px 0;">
            <div class="accuracy-label">CURRENT MODEL ACCURACY</div>
            <div class="accuracy-display">{st.session_state.model_accuracy:.1%}</div>
            <div style="color: #666; font-size: 18px;">
                {accuracy_percentage:.1f} out of 100 correct predictions
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.progress(st.session_state.model_accuracy)
        
        # Option to re-evaluate
        if st.button("🔄 Re-evaluate Model", type="secondary"):
            st.session_state.pop('trained_model', None)
            st.session_state.pop('model_accuracy', None)
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("### 📈 Assessment Reports")
    
    # Check if we have assessment history
    recent_assessments = get_last_30_days_assessments()
    
    if not recent_assessments:
        st.warning("📭 No assessment history found.")
        st.info("Complete an assessment in the '🎯 AI Assessment' tab first.")
    else:
        # Calculate summary
        summary = calculate_30_day_summary()
        
        # Display summary statistics
        st.markdown("#### 📊 Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Assessments", summary['total_assessments'])
        with col2:
            st.metric("Average Score", f"{summary['average_score']:.1f}/100")
        with col3:
            if summary['risk_distribution']:
                most_common_risk = max(summary['risk_distribution'].items(), key=lambda x: x[1])[0]
                st.metric("Most Common Risk", most_common_risk)
            else:
                st.metric("Most Common Risk", "N/A")
        
        # Display assessments
        st.markdown("#### 📋 Recent Assessments")
        
        history_df = pd.DataFrame(recent_assessments)
        
        # Select columns to display
        display_columns = ['date', 'user_name', 'final_score', 'risk_level', 'predicted_class']
        
        # Filter to available columns
        available_columns = [col for col in display_columns if col in history_df.columns]
        
        if available_columns:
            # Format the DataFrame
            display_df = history_df[available_columns].copy()
            display_df = display_df.sort_values('date', ascending=False)
            
            # Rename columns for better display
            column_names = {
                'date': 'Date',
                'user_name': 'User',
                'final_score': 'Score',
                'risk_level': 'Risk Level',
                'predicted_class': 'Prediction'
            }
            
            display_df.rename(columns=column_names, inplace=True)
            
            # Display with pagination
            st.dataframe(
                display_df,
                use_container_width=True,
                height=300
            )
        else:
            st.info("No assessment data available")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 20px;">
    <p>Zim Smart Credit App | AI-Powered Credit Scoring | © 2024</p>
    <p><small>Model Accuracy: {}</small></p>
</div>
""".format(f"{st.session_state.model_accuracy:.1%}" if st.session_state.model_accuracy else "Not evaluated"), unsafe_allow_html=True)
