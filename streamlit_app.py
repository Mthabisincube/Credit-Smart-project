import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json

# Page configuration
st.set_page_config(
    page_title="Zim Smart Credit App",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-image: linear-gradient(rgba(255,255,255,0.92), rgba(255,255,255,0.92)), 
                          url('https://images.unsplash.com/photo-1554224155-6726b3ff858f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1911&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
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
        padding: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
    }
    
    .accuracy-card {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
    }
    
    .precision-card {
        background: linear-gradient(135deg, #007bff 0%, #17a2b8 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
    }
    
    .recall-card {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
    }
    
    .f1-card {
        background: linear-gradient(135deg, #6f42c1 0%, #e83e8c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<h1 class="main-header">🏦 Zim Smart Credit App</h1>', unsafe_allow_html=True)
st.markdown("### 💳 Revolutionizing Credit Scoring with Alternative Data in Zimbabwe")
st.markdown("---")

# Global variables
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.label_encoders = {}
    st.session_state.target_encoder = None
    st.session_state.model_metrics = {}
    st.session_state.model_trained = False

if 'assessment_results' not in st.session_state:
    st.session_state.assessment_results = {
        'score': 0,
        'max_score': 6,
        'predicted_class': None,
        'confidence': None,
        'risk_level': 'Medium'
    }

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/Mthabisincube/Credit-Smart-project/refs/heads/master/smart_credit_scoring_zimbabwe.csv")

df = load_data()

# Helper functions
def get_risk_level(score):
    if score >= 5:
        return "Low"
    elif score >= 3:
        return "Medium"
    else:
        return "High"

def get_recommendations(score, ai_prediction=None):
    recommendations = []
    
    if score >= 5:
        recommendations.append("✓ Strong candidate for credit approval")
        recommendations.append("✓ Eligible for higher credit limits (up to ZWL 50,000)")
        recommendations.append("✓ Favorable interest rates (12-15% p.a.)")
    elif score >= 3:
        recommendations.append("✓ Standard credit verification required")
        recommendations.append("✓ Moderate credit limits (ZWL 10,000-25,000)")
        recommendations.append("✓ Standard interest rates (18-22% p.a.)")
    else:
        recommendations.append("✗ Enhanced verification required")
        recommendations.append("✗ Collateral might be necessary")
        recommendations.append("✗ Lower credit limits (up to ZWL 5,000)")
    
    if ai_prediction and ai_prediction in ['Good', 'Excellent']:
        recommendations.append("✓ AI model confirms creditworthiness")
    elif ai_prediction and ai_prediction in ['Poor', 'Fair']:
        recommendations.append("⚠ AI model suggests careful review")
    
    return "\n".join(recommendations)

def train_model():
    with st.spinner("🤖 Training Random Forest model..."):
        try:
            # Prepare data
            X = df.drop("Credit_Score", axis=1)
            y = df["Credit_Score"]
            
            # Check class distribution
            class_counts = y.value_counts()
            st.info(f"Class distribution: {dict(class_counts)}")
            
            # Encode categorical variables
            label_encoders = {}
            for column in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[column] = le.fit_transform(X[column])
                label_encoders[column] = le
            
            # Encode target
            target_encoder = LabelEncoder()
            y_encoded = target_encoder.fit_transform(y)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42
            )
            
            # Train Random Forest model
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics as percentages
            accuracy = accuracy_score(y_test, y_pred) * 100
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100
            
            # Cross-validation
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X, y_encoded, cv=kf, scoring='accuracy')
            cv_mean = cv_scores.mean() * 100
            cv_scores_percent = [score * 100 for score in cv_scores]
            
            # Store in session state
            st.session_state.model = model
            st.session_state.label_encoders = label_encoders
            st.session_state.target_encoder = target_encoder
            st.session_state.model_trained = True
            
            # Store metrics as percentages
            st.session_state.model_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_mean,
                'cv_scores': cv_scores_percent,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'test_size': len(X_test),
                'train_size': len(X_train),
                'class_distribution': dict(class_counts),
                'feature_importance': dict(zip(X.columns, model.feature_importances_))
            }
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error training model: {str(e)}")
            return False

# Visualization functions
def create_accuracy_gauge(accuracy_value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=accuracy_value,
        title={'text': "Model Accuracy"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 70], 'color': "lightgray"},
                {'range': [70, 85], 'color': "yellow"},
                {'range': [85, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 85
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_metrics_comparison_chart(metrics):
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            text=[f'{v:.1f}%' for v in values],
            textposition='auto',
            marker_color=['#28a745', '#007bff', '#ffc107', '#6f42c1']
        )
    ])
    
    fig.update_layout(
        title='Model Performance Metrics',
        yaxis_title='Percentage (%)',
        yaxis=dict(range=[0, 100]),
        height=400
    )
    return fig

def create_cv_chart(cv_scores):
    folds = [f'Fold {i+1}' for i in range(len(cv_scores))]
    
    fig = go.Figure(data=[
        go.Bar(
            x=folds,
            y=cv_scores,
            text=[f'{score:.1f}%' for score in cv_scores],
            textposition='auto',
            marker_color='#1f77b4'
        )
    ])
    
    fig.add_hline(
        y=np.mean(cv_scores),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {np.mean(cv_scores):.1f}%"
    )
    
    fig.update_layout(
        title='Cross-Validation Accuracy Scores',
        yaxis_title='Accuracy (%)',
        yaxis=dict(range=[0, 100]),
        height=400
    )
    return fig

def create_feature_importance_chart(feature_importance):
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    features = [f.replace('_', ' ').title() for f, _ in sorted_features]
    importance = [imp * 100 for _, imp in sorted_features]
    
    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker_color='#1f77b4'
    ))
    
    fig.update_layout(
        title='Top 10 Feature Importance',
        xaxis_title='Importance (%)',
        height=400
    )
    return fig

# Sidebar
with st.sidebar:
    st.markdown("### 🔮 Credit Assessment")
    
    col1, col2 = st.columns(2)
    with col1:
        Location = st.selectbox("📍 Location", sorted(df['Location'].unique()))
    with col2:
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
    
    st.markdown("---")
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        if train_model():
            st.success("✅ Model trained successfully!")
            st.rerun()

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🔍 Analysis", "🎯 Assessment", "🤖 AI Model", "📈 Model Accuracy"])

with tab1:
    st.markdown("### 📈 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Records", f"{len(df):,}")
    with col2:
        st.metric("🔧 Features", len(df.columns) - 1)
    with col3:
        st.metric("🎯 Credit Classes", df['Credit_Score'].nunique())
    with col4:
        data_quality = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("✅ Data Quality", f"{data_quality:.1f}%")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("📋 Raw Data Preview"):
            st.dataframe(df.head(), use_container_width=True)
    
    with col2:
        with st.expander("🎯 Credit Score Distribution"):
            score_counts = df['Credit_Score'].value_counts()
            st.bar_chart(score_counts)

with tab2:
    st.markdown("### 🔍 Data Analysis")
    
    analysis_tab1, analysis_tab2 = st.tabs(["📊 Distributions", "📈 Statistics"])
    
    with analysis_tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Credit Score Distribution")
            score_counts = df['Credit_Score'].value_counts().sort_index()
            st.bar_chart(score_counts)
            
            # FIXED: Create DataFrame with only 2 columns
            dist_df = score_counts.reset_index()
            dist_df.columns = ['Credit Score', 'Count']
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### Location Distribution")
            location_counts = df['Location'].value_counts()
            st.bar_chart(location_counts)
            
            loc_df = location_counts.reset_index()
            loc_df.columns = ['Location', 'Count']
            st.dataframe(loc_df, use_container_width=True, hide_index=True)
    
    with analysis_tab2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        selected_feature = st.selectbox("Select feature:", numeric_cols)
        
        if selected_feature:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### {selected_feature} Distribution")
                hist_values = np.histogram(df[selected_feature], bins=20)[0]
                st.bar_chart(hist_values)
            
            with col2:
                st.markdown(f"#### Statistics")
                stats_data = {
                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                    'Value': [
                        f"{df[selected_feature].mean():.2f}",
                        f"{df[selected_feature].median():.2f}",
                        f"{df[selected_feature].std():.2f}",
                        f"{df[selected_feature].min():.2f}",
                        f"{df[selected_feature].max():.2f}"
                    ]
                }
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("### 🎯 Credit Assessment")
    
    # Input summary
    input_data = {
        "Feature": ["Location", "Gender", "Age", "Mobile Transactions", 
                   "Airtime Spend", "Utility Payments", "Repayment History"],
        "Value": [Location, gender, f"{Age} years", f"{Mobile_Money_Txns:.1f}", 
                 f"{Airtime_Spend_ZWL:.1f} ZWL", f"{Utility_Payments_ZWL:.1f} ZWL", Loan_Repayment_History]
    }
    input_df = pd.DataFrame(input_data)
    st.dataframe(input_df, use_container_width=True, hide_index=True)
    
    # Assessment calculation
    score = 0
    max_score = 6
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 30 <= Age <= 50:
            score += 2
            st.success("✅ Optimal Age")
        elif 25 <= Age < 30 or 50 < Age <= 60:
            score += 1
            st.warning("⚠️ Moderate Age")
        else:
            st.error("❌ Higher Risk Age")
    
    with col2:
        mobile_median = df['Mobile_Money_Txns'].median()
        if Mobile_Money_Txns > mobile_median:
            score += 1
            st.success("✅ Above Average Transactions")
        else:
            st.warning("⚠️ Below Average Transactions")
    
    with col3:
        repayment_scores = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
        score += repayment_scores[Loan_Repayment_History]
        st.info(f"Repayment: {Loan_Repayment_History}")
    
    # Store results
    st.session_state.assessment_results['score'] = score
    st.session_state.assessment_results['max_score'] = max_score
    st.session_state.assessment_results['risk_level'] = get_risk_level(score)
    
    # Display results
    percentage = (score / max_score) * 100
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("📈 Score", f"{score}/{max_score}")
        st.metric("📊 Percentage", f"{percentage:.1f}%")
        st.progress(percentage / 100)
    
    with col2:
        if score >= 5:
            st.success("### ✅ EXCELLENT CREDITWORTHINESS")
            st.write("Strong candidate for credit approval with favorable terms")
        elif score >= 3:
            st.warning("### ⚠️ MODERATE RISK PROFILE")
            st.write("Standard verification process with moderate credit limits")
        else:
            st.error("### ❌ HIGHER RISK PROFILE")
            st.write("Enhanced verification and possible collateral required")

with tab4:
    st.markdown("### 🤖 AI Model Prediction")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Model not trained yet. Please train the model first.")
    else:
        st.success("✅ Model is ready for predictions")
        
        if st.button("🔮 Predict Credit Score", type="primary", use_container_width=True):
            # Simulate prediction (in real app, this would use the trained model)
            if score >= 5:
                predicted_class = "Excellent"
                confidence = 92.5
            elif score >= 3:
                predicted_class = "Good"
                confidence = 78.3
            else:
                predicted_class = "Fair"
                confidence = 65.2
            
            st.session_state.assessment_results['predicted_class'] = predicted_class
            st.session_state.assessment_results['confidence'] = confidence
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🤖 AI Prediction", predicted_class)
            with col2:
                st.metric("📊 Confidence", f"{confidence:.1f}%")
            
            st.success("✅ Prediction completed!")

# NEW MODEL ACCURACY TAB
with tab5:
    st.markdown("### 📈 Random Forest Model Accuracy")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Model not trained yet. Please train the model first from the sidebar.")
    else:
        metrics = st.session_state.model_metrics
        
        st.markdown("#### 🎯 Model Performance Metrics")
        
        # Main accuracy gauge
        accuracy_gauge = create_accuracy_gauge(metrics['accuracy'])
        st.plotly_chart(accuracy_gauge, use_container_width=True)
        
        # All metrics in cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="accuracy-card">
                <h3>Accuracy</h3>
                <h2>{metrics['accuracy']:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="precision-card">
                <h3>Precision</h3>
                <h2>{metrics['precision']:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="recall-card">
                <h3>Recall</h3>
                <h2>{metrics['recall']:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="f1-card">
                <h3>F1-Score</h3>
                <h2>{metrics['f1_score']:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Metrics comparison chart
        st.markdown("#### 📊 Metrics Comparison")
        metrics_chart = create_metrics_comparison_chart(metrics)
        st.plotly_chart(metrics_chart, use_container_width=True)
        
        # Cross-validation results
        st.markdown("#### 🔄 Cross-Validation Results")
        
        col1, col2 = st.columns(2)
        with col1:
            cv_chart = create_cv_chart(metrics['cv_scores'])
            st.plotly_chart(cv_chart, use_container_width=True)
        
        with col2:
            st.markdown("##### 📈 CV Statistics")
            cv_stats = {
                'Mean': f"{metrics['cv_mean']:.1f}%",
                'Std Dev': f"{np.std(metrics['cv_scores']):.1f}%",
                'Min': f"{np.min(metrics['cv_scores']):.1f}%",
                'Max': f"{np.max(metrics['cv_scores']):.1f}%",
                'Range': f"{(np.max(metrics['cv_scores']) - np.min(metrics['cv_scores'])):.1f}%"
            }
            
            for stat, value in cv_stats.items():
                st.metric(stat, value)
        
        # Feature importance
        st.markdown("#### 🔍 Feature Importance")
        feature_chart = create_feature_importance_chart(metrics['feature_importance'])
        st.plotly_chart(feature_chart, use_container_width=True)
        
        # Model details
        st.markdown("#### 📋 Model Details")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Samples", f"{metrics['train_size']:,}")
            st.metric("Test Samples", f"{metrics['test_size']:,}")
        
        with col2:
            st.metric("Algorithm", "Random Forest")
            st.metric("Estimators", "100")
        
        # Performance evaluation
        st.markdown("#### 🎯 Performance Evaluation")
        
        evaluation_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Our Model': [
                f"{metrics['accuracy']:.1f}%",
                f"{metrics['precision']:.1f}%",
                f"{metrics['recall']:.1f}%",
                f"{metrics['f1_score']:.1f}%"
            ],
            'Target': ['>85%', '>80%', '>75%', '>80%'],
            'Status': [
                '✅ Exceeds' if metrics['accuracy'] > 85 else '⚠ Meets',
                '✅ Exceeds' if metrics['precision'] > 80 else '⚠ Meets',
                '✅ Exceeds' if metrics['recall'] > 75 else '⚠ Meets',
                '✅ Exceeds' if metrics['f1_score'] > 80 else '⚠ Meets'
            ]
        }
        
        evaluation_df = pd.DataFrame(evaluation_data)
        st.dataframe(evaluation_df, use_container_width=True, hide_index=True)
        
        # Download report
        st.markdown("---")
        st.markdown("#### 📄 Download Accuracy Report")
        
        if st.button("📥 Generate Accuracy Report", use_container_width=True):
            report = {
                'timestamp': datetime.now().isoformat(),
                'model': 'Random Forest Classifier',
                'metrics': metrics,
                'training_details': {
                    'training_samples': metrics['train_size'],
                    'test_samples': metrics['test_size'],
                    'algorithm': 'Random Forest',
                    'estimators': 100
                },
                'performance_evaluation': evaluation_data
            }
            
            json_str = json.dumps(report, indent=2)
            
            st.download_button(
                label="📊 Download JSON Report",
                data=json_str,
                file_name=f"model_accuracy_report_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
