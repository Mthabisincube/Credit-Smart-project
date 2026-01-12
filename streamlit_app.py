import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_score, recall_score, 
                           f1_score, precision_recall_curve, auc)
from sklearn.impute import SimpleImputer, KNNImputer

# ADDED: Random Forest imports
from sklearn.ensemble import RandomForestClassifier

# XGBoost and SHAP
import xgboost as xgb
import shap

# Plotting
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
    
    .phase-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
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
    
    .shap-box {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #4CAF50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<h1 class="main-header">🏦 Zim Smart Credit App</h1>', unsafe_allow_html=True)
st.markdown("### 💳 Revolutionizing Credit Scoring with Alternative Data in Zimbabwe")
st.markdown("---")

# ============================================================
# PHASE 1: DATA LOADING & INITIAL PROCESSING
# ============================================================
@st.cache_data
def load_data():
    """Load and preprocess initial dataset"""
    df = pd.read_csv("https://raw.githubusercontent.com/Mthabisincube/Credit-Smart-project/refs/heads/master/smart_credit_scoring_zimbabwe.csv")
    
    # Ensure proper data types
    numeric_cols = ['Mobile_Money_Txns', 'Airtime_Spend_ZWL', 'Utility_Payments_ZWL', 'Age']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# Load data
df = load_data()

# ============================================================
# PHASE 2: FEATURE ENGINEERING AND DATA PREPROCESSING
# ============================================================
def engineer_features(df):
    """Perform feature engineering as described in Phase 2"""
    df_engineered = df.copy()
    
    # 1. Aggregation and Summarisation Features
    st.info("🔄 Creating aggregation features...")
    
    # Total monthly spending (assuming data is monthly)
    df_engineered['Total_Monthly_Spending'] = df_engineered['Airtime_Spend_ZWL'] + \
                                              df_engineered['Utility_Payments_ZWL']
    
    # Average transaction size for mobile money
    df_engineered['Avg_Mobile_Transaction'] = df_engineered['Mobile_Money_Txns'].apply(
        lambda x: x if pd.isna(x) else x / 30  # Assuming 30 transactions per month average
    )
    
    # 2. Temporal Feature Extraction (simulated)
    st.info("⏰ Creating temporal features...")
    
    # Spending patterns - creating simulated temporal features
    # In real scenario, you would have date columns
    df_engineered['Spending_Last30_Days'] = df_engineered['Total_Monthly_Spending'] * 1.1  # Simulated increase
    df_engineered['Spending_Last60_Days'] = df_engineered['Total_Monthly_Spending'] * 2.2
    df_engineered['Spending_Last90_Days'] = df_engineered['Total_Monthly_Spending'] * 3.3
    
    # 3. Behavioural Indicators
    st.info("📊 Creating behavioral indicators...")
    
    # Coefficient of Variation (simulated)
    # In real scenario, calculate from transaction history
    df_engineered['Spending_CV'] = np.random.uniform(0.1, 0.5, len(df_engineered))
    
    # Transaction frequency stability
    df_engineered['Transaction_Consistency'] = df_engineered['Mobile_Money_Txns'] / \
                                               (df_engineered['Mobile_Money_Txns'].mean() + 1)
    
    # 4. Ratio and Interaction Features
    st.info("📈 Creating ratio features...")
    
    # Balance-to-spending ratio (simulated)
    df_engineered['Balance_To_Spending_Ratio'] = np.random.uniform(0.5, 5, len(df_engineered))
    
    # Income-to-debt ratio (simulated)
    df_engineered['Income_To_Debt_Ratio'] = np.random.uniform(1, 10, len(df_engineered))
    
    # Age to spending ratio
    df_engineered['Age_Spending_Ratio'] = df_engineered['Age'] / (df_engineered['Total_Monthly_Spending'] + 1)
    
    # 5. Missing Data Handling
    st.info("🔍 Handling missing data...")
    
    # Check for missing values
    missing_data = df_engineered.isnull().sum()
    
    # Impute numerical features
    numeric_features = df_engineered.select_dtypes(include=[np.number]).columns
    
    # Use median imputation for robustness
    imputer = SimpleImputer(strategy='median')
    df_engineered[numeric_features] = imputer.fit_transform(df_engineered[numeric_features])
    
    # Create binary indicators for originally missing values
    for col in numeric_features:
        if df[col].isnull().sum() > 0:
            df_engineered[f'{col}_was_missing'] = df[col].isnull().astype(int)
    
    return df_engineered

# ============================================================
# PHASE 3: MODEL TRAINING AND EVALUATION
# ============================================================
# ADDED: Random Forest training function
def train_random_forest_model(X_train, y_train, X_test, y_test):
    """Train Random Forest model with comprehensive evaluation"""
    
    # Define Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    # Train model
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)
    
    return rf_model, y_pred, y_pred_proba

def train_xgboost_model(X_train, y_train, X_test, y_test):
    """Train XGBoost model with comprehensive evaluation"""
    
    # Define XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    return model, y_pred, y_pred_proba

def evaluate_model(y_test, y_pred, y_pred_proba, class_names):
    """Comprehensive model evaluation"""
    
    results = {}
    
    # 1. ROC-AUC
    if len(class_names) == 2:
        results['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        results['roc_curve'] = (fpr, tpr)
    else:
        # Multi-class ROC-AUC
        results['roc_auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    
    # ADDED: Accuracy
    results['accuracy'] = accuracy_score(y_test, y_pred)
    
    # 2. Precision, Recall, F1-Score
    results['precision'] = precision_score(y_test, y_pred, average='weighted')
    results['recall'] = recall_score(y_test, y_pred, average='weighted')
    results['f1'] = f1_score(y_test, y_pred, average='weighted')
    
    # 3. Confusion Matrix
    results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    
    # 4. Classification Report
    results['classification_report'] = classification_report(y_test, y_pred, 
                                                            target_names=class_names, 
                                                            output_dict=True)
    
    # 5. Calibration (simplified)
    # In practice, use CalibrationDisplay or probability calibration
    
    return results

def plot_shap_summary(model, X_train, feature_names, model_type='xgb'):
    """Generate SHAP summary plot"""
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_train)
    
    # Create summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, 
                      show=False, max_display=10)
    
    # Customize title based on model type
    if model_type == 'rf':
        plt.title('Random Forest - SHAP Feature Importance', fontsize=14)
    else:
        plt.title('XGBoost - SHAP Feature Importance', fontsize=14)
    
    plt.tight_layout()
    return fig

# ADDED: Compare models function
def compare_models(rf_results, xgb_results):
    """Compare Random Forest and XGBoost results"""
    comparison = {
        'Metric': ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Random Forest': [
            rf_results['roc_auc'],
            rf_results['accuracy'],
            rf_results['precision'],
            rf_results['recall'],
            rf_results['f1']
        ],
        'XGBoost': [
            xgb_results['roc_auc'],
            xgb_results['accuracy'],
            xgb_results['precision'],
            xgb_results['recall'],
            xgb_results['f1']
        ]
    }
    
    return pd.DataFrame(comparison)

# ============================================================
# STREAMLIT INTERFACE
# ============================================================

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
    
    # ADDED: Model selection
    model_choice = st.selectbox(
        "🤖 Select Model for Prediction",
        ["Random Forest", "XGBoost", "Both"]
    )

# Main content with tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard", 
    "🔬 Feature Engineering", 
    "🌲 Random Forest", 
    "🤖 XGBoost", 
    "📊 Model Comparison",
    "🎯 Assessment"
])

with tab1:
    st.markdown('<div class="phase-header"><h2>Phase 1: Data Overview</h2></div>', unsafe_allow_html=True)
    
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
            <h3>🔧 Original Features</h3>
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
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>✅ Data Quality</h3>
            <h2>{100 - missing_pct:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Data preview
    with st.expander("📋 Raw Data Preview", expanded=True):
        st.dataframe(df, use_container_width=True, height=300)

with tab2:
    st.markdown('<div class="phase-header"><h2>Phase 2: Feature Engineering</h2></div>', unsafe_allow_html=True)
    
    if st.button("🚀 Engineer Features", type="primary"):
        with st.spinner("🔄 Performing feature engineering..."):
            # Apply feature engineering
            df_engineered = engineer_features(df)
            
            st.success(f"✅ Feature engineering complete! Added {len(df_engineered.columns) - len(df.columns)} new features")
            
            # Show engineered features
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔍 Engineered Features Overview")
                new_features = [col for col in df_engineered.columns if col not in df.columns]
                st.write(f"**New features created:** {len(new_features)}")
                for feature in new_features:
                    st.markdown(f"• {feature}")
            
            with col2:
                st.markdown("#### 📊 Feature Statistics")
                st.dataframe(df_engineered[new_features].describe().round(2), use_container_width=True)
            
            # Store in session state
            st.session_state['df_engineered'] = df_engineered
            st.session_state['features_engineered'] = True

with tab3:
    st.markdown('<div class="phase-header"><h2>Phase 3: Random Forest Model</h2></div>', unsafe_allow_html=True)
    
    if 'df_engineered' in st.session_state:
        df_engineered = st.session_state['df_engineered']
        
        # Prepare data for modeling
        X = df_engineered.drop("Credit_Score", axis=1)
        y = df_engineered["Credit_Score"]
        
        # Encode categorical variables
        label_encoders = {}
        for column in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column].astype(str))
            label_encoders[column] = le
        
        # Encode target
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        class_names = target_encoder.classes_
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if st.button("🌲 Train Random Forest Model", type="primary"):
            with st.spinner("🌲 Training Random Forest model..."):
                # Train Random Forest model
                rf_model, y_pred_rf, y_pred_proba_rf = train_random_forest_model(
                    X_train_scaled, y_train, X_test_scaled, y_test
                )
                
                # Evaluate model
                rf_results = evaluate_model(y_test, y_pred_rf, y_pred_proba_rf, class_names)
                
                st.success("✅ Random Forest model trained successfully!")
                
                # Display metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("🎯 ROC-AUC", f"{rf_results['roc_auc']:.3f}")
                with col2:
                    st.metric("🎯 Accuracy", f"{rf_results['accuracy']:.3f}")
                with col3:
                    st.metric("📊 Precision", f"{rf_results['precision']:.3f}")
                with col4:
                    st.metric("🔍 Recall", f"{rf_results['recall']:.3f}")
                with col5:
                    st.metric("⚖️ F1-Score", f"{rf_results['f1']:.3f}")
                
                # Confusion Matrix
                st.markdown("#### 📈 Confusion Matrix")
                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                sns.heatmap(rf_results['confusion_matrix'], annot=True, fmt='d', 
                           cmap='Blues', xticklabels=class_names, yticklabels=class_names)
                plt.title('Random Forest - Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                st.pyplot(fig_cm)
                
                # Feature Importance
                st.markdown("#### 🔍 Feature Importance (Random Forest)")
                rf_feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Plot top 20 features
                fig_fi, ax_fi = plt.subplots(figsize=(10, 8))
                top_features = rf_feature_importance.head(20)
                ax_fi.barh(range(len(top_features)), top_features['Importance'], color='#36D1DC')
                ax_fi.set_yticks(range(len(top_features)))
                ax_fi.set_yticklabels(top_features['Feature'])
                ax_fi.invert_yaxis()
                ax_fi.set_xlabel('Importance')
                ax_fi.set_title('Random Forest - Top 20 Feature Importance')
                plt.tight_layout()
                st.pyplot(fig_fi)
                
                # SHAP Analysis for Random Forest
                st.markdown("#### 📊 SHAP Analysis for Model Interpretability")
                if st.button("Generate SHAP Analysis (RF)"):
                    with st.spinner("Calculating SHAP values for Random Forest..."):
                        # Sample for SHAP (computationally intensive)
                        X_sample = X_train_scaled[:100]
                        shap_fig = plot_shap_summary(rf_model, X_sample, X.columns.tolist(), model_type='rf')
                        st.pyplot(shap_fig)
                        st.markdown("""
                        <div class="shap-box">
                            <h4>SHAP Interpretation (Random Forest)</h4>
                            <p>• Each point represents a customer<br>
                            • Color shows feature value (red=high, blue=low)<br>
                            • Position shows impact on prediction<br>
                            • Features sorted by overall importance</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Store model in session state
                st.session_state['rf_model'] = rf_model
                st.session_state['rf_scaler'] = scaler
                st.session_state['rf_label_encoders'] = label_encoders
                st.session_state['rf_target_encoder'] = target_encoder
                st.session_state['rf_X_columns'] = X.columns
                st.session_state['rf_results'] = rf_results
                
    else:
        st.warning("⚠️ Please engineer features first in Tab 2")

with tab4:
    st.markdown('<div class="phase-header"><h2>Phase 3: XGBoost Model</h2></div>', unsafe_allow_html=True)
    
    if 'df_engineered' in st.session_state:
        df_engineered = st.session_state['df_engineered']
        
        # Prepare data for modeling
        X = df_engineered.drop("Credit_Score", axis=1)
        y = df_engineered["Credit_Score"]
        
        # Encode categorical variables
        label_encoders = {}
        for column in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column].astype(str))
            label_encoders[column] = le
        
        # Encode target
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        class_names = target_encoder.classes_
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if st.button("🎯 Train XGBoost Model", type="primary"):
            with st.spinner("🤖 Training XGBoost model..."):
                # Train model
                xgb_model, y_pred_xgb, y_pred_proba_xgb = train_xgboost_model(
                    X_train_scaled, y_train, X_test_scaled, y_test
                )
                
                # Evaluate model
                xgb_results = evaluate_model(y_test, y_pred_xgb, y_pred_proba_xgb, class_names)
                
                st.success("✅ XGBoost model trained successfully!")
                
                # Display metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("🎯 ROC-AUC", f"{xgb_results['roc_auc']:.3f}")
                with col2:
                    st.metric("🎯 Accuracy", f"{xgb_results['accuracy']:.3f}")
                with col3:
                    st.metric("📊 Precision", f"{xgb_results['precision']:.3f}")
                with col4:
                    st.metric("🔍 Recall", f"{xgb_results['recall']:.3f}")
                with col5:
                    st.metric("⚖️ F1-Score", f"{xgb_results['f1']:.3f}")
                
                # Confusion Matrix
                st.markdown("#### 📈 Confusion Matrix")
                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                sns.heatmap(xgb_results['confusion_matrix'], annot=True, fmt='d', 
                           cmap='Reds', xticklabels=class_names, yticklabels=class_names)
                plt.title('XGBoost - Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                st.pyplot(fig_cm)
                
                # Feature Importance
                st.markdown("#### 🔍 Feature Importance (XGBoost)")
                xgb_feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': xgb_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Plot top 20 features
                fig_fi, ax_fi = plt.subplots(figsize=(10, 8))
                top_features = xgb_feature_importance.head(20)
                ax_fi.barh(range(len(top_features)), top_features['Importance'], color='#FF416C')
                ax_fi.set_yticks(range(len(top_features)))
                ax_fi.set_yticklabels(top_features['Feature'])
                ax_fi.invert_yaxis()
                ax_fi.set_xlabel('Importance')
                ax_fi.set_title('XGBoost - Top 20 Feature Importance')
                plt.tight_layout()
                st.pyplot(fig_fi)
                
                # SHAP Analysis
                st.markdown("#### 📊 SHAP Analysis for Model Interpretability")
                if st.button("Generate SHAP Analysis (XGBoost)"):
                    with st.spinner("Calculating SHAP values..."):
                        # Sample for SHAP (computationally intensive)
                        X_sample = X_train_scaled[:100]
                        shap_fig = plot_shap_summary(xgb_model, X_sample, X.columns.tolist())
                        st.pyplot(shap_fig)
                        st.markdown("""
                        <div class="shap-box">
                            <h4>SHAP Interpretation (XGBoost)</h4>
                            <p>• Each point represents a customer<br>
                            • Color shows feature value (red=high, blue=low)<br>
                            • Position shows impact on prediction<br>
                            • Features sorted by overall importance</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Store model in session state
                st.session_state['xgb_model'] = xgb_model
                st.session_state['xgb_scaler'] = scaler
                st.session_state['xgb_label_encoders'] = label_encoders
                st.session_state['xgb_target_encoder'] = target_encoder
                st.session_state['xgb_X_columns'] = X.columns
                st.session_state['xgb_results'] = xgb_results
                
    else:
        st.warning("⚠️ Please engineer features first in Tab 2")

with tab5:
    st.markdown('<div class="phase-header"><h2>Model Comparison Dashboard</h2></div>', unsafe_allow_html=True)
    
    if 'rf_results' in st.session_state and 'xgb_results' in st.session_state:
        # Compare models
        comparison_df = compare_models(
            st.session_state['rf_results'], 
            st.session_state['xgb_results']
        )
        
        st.markdown("### 📊 Performance Comparison")
        
        # Display comparison table with styling
        st.dataframe(comparison_df.style.format({
            'Random Forest': '{:.3f}',
            'XGBoost': '{:.3f}'
        }).highlight_max(axis=1, color='lightgreen'), 
        use_container_width=True)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(comparison_df['Metric']))
        width = 0.35
        
        ax.bar(x - width/2, comparison_df['Random Forest'], width, label='Random Forest', color='#36D1DC')
        ax.bar(x + width/2, comparison_df['XGBoost'], width, label='XGBoost', color='#FF416C')
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Random Forest vs XGBoost Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Metric'], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (rf_val, xgb_val) in enumerate(zip(comparison_df['Random Forest'], comparison_df['XGBoost'])):
            ax.text(i - width/2, rf_val + 0.01, f'{rf_val:.3f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, xgb_val + 0.01, f'{xgb_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Feature importance comparison
        if 'rf_model' in st.session_state and 'xgb_model' in st.session_state:
            st.markdown("### 🔍 Feature Importance Comparison")
            
            # Get feature importance from both models
            rf_fi = pd.DataFrame({
                'Feature': st.session_state['rf_X_columns'],
                'RF_Importance': st.session_state['rf_model'].feature_importances_
            })
            
            xgb_fi = pd.DataFrame({
                'Feature': st.session_state['xgb_X_columns'],
                'XGB_Importance': st.session_state['xgb_model'].feature_importances_
            })
            
            # Merge and get top 10 features by average importance
            fi_comparison = pd.merge(rf_fi, xgb_fi, on='Feature', how='outer').fillna(0)
            fi_comparison['Avg_Importance'] = (fi_comparison['RF_Importance'] + fi_comparison['XGB_Importance']) / 2
            top_fi = fi_comparison.nlargest(10, 'Avg_Importance')
            
            # Plot comparison
            fig, ax = plt.subplots(figsize=(12, 8))
            x = np.arange(len(top_fi))
            width = 0.35
            
            ax.bar(x - width/2, top_fi['RF_Importance'], width, label='Random Forest', color='#36D1DC')
            ax.bar(x + width/2, top_fi['XGB_Importance'], width, label='XGBoost', color='#FF416C')
            
            ax.set_xlabel('Features')
            ax.set_ylabel('Importance')
            ax.set_title('Top 10 Features - Importance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(top_fi['Feature'], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
    else:
        st.info("ℹ️ Train both Random Forest and XGBoost models to see comparison")

with tab6:
    st.markdown('<div class="phase-header"><h2>Phase 4: Credit Assessment</h2></div>', unsafe_allow_html=True)
    
    # User input
    user_data = pd.DataFrame({
        'Location': [Location],
        'Gender': [gender],
        'Age': [Age],
        'Mobile_Money_Txns': [Mobile_Money_Txns],
        'Airtime_Spend_ZWL': [Airtime_Spend_ZWL],
        'Utility_Payments_ZWL': [Utility_Payments_ZWL],
        'Loan_Repayment_History': [Loan_Repayment_History]
    })
    
    # Display input summary
    st.markdown("### 📋 Your Input Summary")
    st.dataframe(user_data.T.rename(columns={0: 'Value'}), use_container_width=True)
    
    def make_prediction(user_data, model_type='rf'):
        """Make prediction using specified model"""
        try:
            # Apply same feature engineering
            user_engineered = engineer_features(user_data)
            
            # Get appropriate model and preprocessing objects
            if model_type == 'rf':
                if 'rf_model' not in st.session_state:
                    return None, "Random Forest model not trained yet"
                
                model = st.session_state['rf_model']
                scaler = st.session_state['rf_scaler']
                label_encoders = st.session_state['rf_label_encoders']
                target_encoder = st.session_state['rf_target_encoder']
                X_columns = st.session_state['rf_X_columns']
                model_name = "Random Forest"
            else:
                if 'xgb_model' not in st.session_state:
                    return None, "XGBoost model not trained yet"
                
                model = st.session_state['xgb_model']
                scaler = st.session_state['xgb_scaler']
                label_encoders = st.session_state['xgb_label_encoders']
                target_encoder = st.session_state['xgb_target_encoder']
                X_columns = st.session_state['xgb_X_columns']
                model
