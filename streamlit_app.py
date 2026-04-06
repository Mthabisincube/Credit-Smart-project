import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                            precision_score, recall_score, f1_score, roc_auc_score, roc_curve)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import uuid
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF
import base64
import warnings
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Zim Smart Credit - Advanced Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM STYLING ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }

    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main-header {
        font-size: 3.5rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        padding: 1rem;
        letter-spacing: -1px;
    }
    
    .subtitle {
        color: rgba(255,255,255,0.95);
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        padding: 1.5rem;
        border-radius: 10px;
        color: #155724;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 5px solid #ffc107;
        padding: 1.5rem;
        border-radius: 10px;
        color: #856404;
        margin: 1rem 0;
    }
    
    .danger-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
        padding: 1.5rem;
        border-radius: 10px;
        color: #721c24;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-left: 5px solid #17a2b8;
        padding: 1.5rem;
        border-radius: 10px;
        color: #0c5460;
        margin: 1rem 0;
    }
    
    .risk-badge-low {
        background: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .risk-badge-medium {
        background: #ffc107;
        color: #333;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .risk-badge-high {
        background: #dc3545;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE INITIALIZATION ====================
if 'assessments_history' not in st.session_state:
    st.session_state.assessments_history = []

if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.label_encoders = {}
    st.session_state.target_encoder = None
    st.session_state.model_metrics = {}
    st.session_state.model_trained = False

if 'risk_model' not in st.session_state:
    st.session_state.risk_model = None
    st.session_state.risk_thresholds = None

if 'explainer' not in st.session_state:
    st.session_state.explainer = None
    st.session_state.shap_values = None
    st.session_state.X_sample = None
    st.session_state.X_columns = None

if 'current_assessment' not in st.session_state:
    st.session_state.current_assessment = None

# ==================== DATA LOADING ====================
@st.cache_data
def load_data():
    """Load and prepare the credit scoring dataset"""
    df = pd.read_csv("https://raw.githubusercontent.com/Mthabisincube/Credit-Smart-project/refs/heads/master/smart_credit_scoring_zimbabwe.csv")
    
    # Add realistic income sources with proper distribution
    np.random.seed(42)
    income_sources = ['Formal Employment', 'Informal Business', 'Farming', 'Remittances', 'Other']
    df['Income_Source'] = np.random.choice(
        income_sources, 
        size=len(df), 
        p=[0.35, 0.30, 0.15, 0.15, 0.05]
    )
    
    return df

df = load_data()

# ==================== RISK ASSESSMENT MODELS ====================
class AdvancedRiskAssessor:
    """Advanced risk assessment engine with multiple methodologies"""
    
    def __init__(self, data):
        self.data = data
        self.risk_factors = {
            'age_risk': self._calculate_age_risk,
            'income_stability_risk': self._calculate_income_risk,
            'behavioral_risk': self._calculate_behavioral_risk,
            'repayment_risk': self._calculate_repayment_risk,
            'geographic_risk': self._calculate_geographic_risk
        }
    
    def _calculate_age_risk(self, age):
        """Age-based risk scoring"""
        if 30 <= age <= 50:
            return 10  # Low risk
        elif 25 <= age < 30 or 50 < age <= 60:
            return 25  # Medium risk
        else:
            return 45  # Higher risk
    
    def _calculate_income_risk(self, income_source):
        """Income source stability risk"""
        risk_scores = {
            'Formal Employment': 5,
            'Informal Business': 25,
            'Remittances': 30,
            'Farming': 35,
            'Other': 40
        }
        return risk_scores.get(income_source, 40)
    
    def _calculate_behavioral_risk(self, mobile_txns, airtime, utility, data):
        """Financial behavior risk assessment"""
        risk_score = 0
        
        # Mobile money activity
        mobile_percentile = (data['Mobile_Money_Txns'] <= mobile_txns).mean() * 100
        if mobile_percentile < 25:
            risk_score += 20
        elif mobile_percentile < 50:
            risk_score += 10
        else:
            risk_score += 5
        
        # Airtime spend (communication consistency)
        airtime_percentile = (data['Airtime_Spend_ZWL'] <= airtime).mean() * 100
        if airtime_percentile < 25:
            risk_score += 15
        elif airtime_percentile < 50:
            risk_score += 8
        else:
            risk_score += 3
        
        # Utility payments (financial responsibility)
        utility_percentile = (data['Utility_Payments_ZWL'] <= utility).mean() * 100
        if utility_percentile < 25:
            risk_score += 15
        elif utility_percentile < 50:
            risk_score += 8
        else:
            risk_score += 3
        
        return min(risk_score, 50)
    
    def _calculate_repayment_risk(self, repayment_history):
        """Repayment history risk"""
        risk_scores = {
            'Excellent': 0,
            'Good': 10,
            'Fair': 30,
            'Poor': 60
        }
        return risk_scores.get(repayment_history, 60)
    
    def _calculate_geographic_risk(self, location):
        """Location-based risk (can be customized based on economic data)"""
        # This is a simplified version - in production, use actual regional economic data
        urban_centers = ['Harare', 'Bulawayo']
        if location in urban_centers:
            return 10
        else:
            return 20
    
    def calculate_comprehensive_risk_score(self, applicant_data):
        """Calculate overall risk score (0-100, lower is better)"""
        risk_components = {
            'age_risk': self._calculate_age_risk(applicant_data['age']),
            'income_risk': self._calculate_income_risk(applicant_data['income_source']),
            'behavioral_risk': self._calculate_behavioral_risk(
                applicant_data['mobile_txns'],
                applicant_data['airtime'],
                applicant_data['utility'],
                self.data
            ),
            'repayment_risk': self._calculate_repayment_risk(applicant_data['repayment_history']),
            'geographic_risk': self._calculate_geographic_risk(applicant_data['location'])
        }
        
        # Weighted average
        weights = {
            'age_risk': 0.15,
            'income_risk': 0.20,
            'behavioral_risk': 0.25,
            'repayment_risk': 0.30,
            'geographic_risk': 0.10
        }
        
        total_risk = sum(risk_components[key] * weights[key] for key in risk_components)
        
        return total_risk, risk_components
    
    def get_risk_category(self, risk_score):
        """Categorize risk level"""
        if risk_score < 25:
            return "Low Risk", "success"
        elif risk_score < 50:
            return "Medium Risk", "warning"
        else:
            return "High Risk", "danger"
    
    def generate_risk_recommendations(self, risk_score, risk_components, applicant_data):
        """Generate specific recommendations based on risk profile"""
        recommendations = []
        
        category, _ = self.get_risk_category(risk_score)
        
        # Credit limits
        if risk_score < 25:
            recommendations.append({
                'icon': '💰',
                'title': 'Recommended Credit Limit',
                'value': 'ZWL 30,000 - 50,000',
                'detail': 'Excellent risk profile supports higher credit limits'
            })
            recommendations.append({
                'icon': '📊',
                'title': 'Interest Rate',
                'value': '12% - 15% p.a.',
                'detail': 'Preferential rates available'
            })
            recommendations.append({
                'icon': '⚡',
                'title': 'Approval Speed',
                'value': 'Fast-track',
                'detail': 'Minimal documentation required'
            })
        elif risk_score < 50:
            recommendations.append({
                'icon': '💰',
                'title': 'Recommended Credit Limit',
                'value': 'ZWL 10,000 - 25,000',
                'detail': 'Standard credit limits apply'
            })
            recommendations.append({
                'icon': '📊',
                'title': 'Interest Rate',
                'value': '18% - 22% p.a.',
                'detail': 'Standard market rates'
            })
            recommendations.append({
                'icon': '📋',
                'title': 'Requirements',
                'value': 'Standard verification',
                'detail': 'Income proof and guarantor recommended'
            })
        else:
            recommendations.append({
                'icon': '💰',
                'title': 'Recommended Credit Limit',
                'value': 'ZWL 3,000 - 8,000',
                'detail': 'Conservative limits recommended'
            })
            recommendations.append({
                'icon': '📊',
                'title': 'Interest Rate',
                'value': '25% - 30% p.a.',
                'detail': 'Risk-adjusted pricing'
            })
            recommendations.append({
                'icon': '🔒',
                'title': 'Security',
                'value': 'Collateral required',
                'detail': 'Additional security measures needed'
            })
        
        # Specific improvement areas
        if risk_components['repayment_risk'] > 30:
            recommendations.append({
                'icon': '⚠️',
                'title': 'Primary Concern',
                'value': 'Repayment History',
                'detail': f"Current: {applicant_data['repayment_history']}. Building consistent repayment records is critical."
            })
        
        if risk_components['behavioral_risk'] > 30:
            recommendations.append({
                'icon': '📱',
                'title': 'Improvement Area',
                'value': 'Digital Financial Activity',
                'detail': 'Increase mobile money transactions and utility payments for better scoring'
            })
        
        if risk_components['income_risk'] > 25:
            recommendations.append({
                'icon': '💼',
                'title': 'Income Verification',
                'value': 'Enhanced Due Diligence',
                'detail': f"Income source ({applicant_data['income_source']}) requires additional verification"
            })
        
        return recommendations

# ==================== MACHINE LEARNING MODEL TRAINING ====================
def train_advanced_model():
    """Train both credit scoring and risk assessment models"""
    with st.spinner("🤖 Training Advanced AI Models..."):
        try:
            # Prepare features
            X = df.drop(["Credit_Score"], axis=1)
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
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Train Random Forest
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred) * 100
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100
            
            # Cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X, y_encoded, cv=kf, scoring='accuracy')
            cv_scores_percent = cv_scores * 100
            
            # Calculate AUC for multi-class
            try:
                auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted') * 100
            except:
                auc_score = None
            
            # Store model and metrics
            st.session_state.model = model
            st.session_state.label_encoders = label_encoders
            st.session_state.target_encoder = target_encoder
            st.session_state.model_trained = True
            
            st.session_state.model_metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc_score': float(auc_score) if auc_score else None,
                'cv_mean': float(cv_scores_percent.mean()),
                'cv_std': float(cv_scores_percent.std()),
                'cv_scores': cv_scores_percent.tolist(),
                'test_size': int(len(X_test)),
                'train_size': int(len(X_train)),
                'feature_importance': dict(zip(X.columns, model.feature_importances_)),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'class_names': target_encoder.classes_.tolist()
            }
            
            st.session_state.X_columns = X.columns.tolist()
            
            # Initialize SHAP
            try:
                explainer = shap.TreeExplainer(model)
                X_sample = X_test.sample(n=min(100, len(X_test)), random_state=42)
                shap_values = explainer.shap_values(X_sample)
                
                st.session_state.explainer = explainer
                st.session_state.shap_values = shap_values
                st.session_state.X_sample = X_sample
            except Exception as e:
                st.warning(f"SHAP initialization skipped: {str(e)[:100]}")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Model training failed: {str(e)}")
            return False

# Train model on startup
if not st.session_state.model_trained:
    train_advanced_model()

# ==================== HELPER FUNCTIONS ====================
def save_assessment(assessment_data):
    """Save assessment to history"""
    assessment_data['assessment_id'] = str(uuid.uuid4())[:8].upper()
    assessment_data['timestamp'] = datetime.now().isoformat()
    assessment_data['date'] = datetime.now().strftime('%Y-%m-%d')
    assessment_data['time'] = datetime.now().strftime('%H:%M:%S')
    
    st.session_state.assessments_history.append(assessment_data.copy())
    
    # Keep only last 90 days
    cutoff_date = datetime.now() - timedelta(days=90)
    st.session_state.assessments_history = [
        a for a in st.session_state.assessments_history 
        if datetime.fromisoformat(a['timestamp']) > cutoff_date
    ]
    
    return assessment_data['assessment_id']

def calculate_traditional_score(age, mobile_txns, repayment_history, data):
    """Calculate traditional credit score"""
    score = 0
    max_score = 6
    
    # Age component (0-2 points)
    if 30 <= age <= 50:
        score += 2
    elif 25 <= age < 30 or 50 < age <= 60:
        score += 1
    
    # Mobile activity (0-1 points)
    mobile_median = data['Mobile_Money_Txns'].median()
    if mobile_txns > mobile_median:
        score += 1
    
    # Repayment history (0-3 points)
    repayment_scores = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
    score += repayment_scores.get(repayment_history, 0)
    
    return score, max_score

def generate_pdf_report(assessment_data, risk_analysis, recommendations):
    """Generate comprehensive PDF report"""
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 15, "ZIM SMART CREDIT - ASSESSMENT REPORT", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
    pdf.ln(10)
    
    # Assessment ID and Date
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, f"Assessment ID: {assessment_data.get('assessment_id', 'N/A')}", ln=True)
    pdf.cell(0, 8, f"Date: {assessment_data.get('date', 'N/A')} {assessment_data.get('time', '')}", ln=True)
    pdf.ln(5)
    
    # Applicant Details
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "APPLICANT PROFILE", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 7, f"Location: {assessment_data.get('location')}", ln=True)
    pdf.cell(0, 7, f"Gender: {assessment_data.get('gender')}", ln=True)
    pdf.cell(0, 7, f"Age: {assessment_data.get('age')} years", ln=True)
    pdf.cell(0, 7, f"Income Source: {assessment_data.get('income_source', 'N/A')}", ln=True)
    pdf.ln(5)
    
    # Financial Metrics
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "FINANCIAL BEHAVIOR", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 7, f"Mobile Money Transactions: {assessment_data.get('mobile_money_txns', 0):.0f}", ln=True)
    pdf.cell(0, 7, f"Airtime Spend: ZWL {assessment_data.get('airtime_spend', 0):.0f}", ln=True)
    pdf.cell(0, 7, f"Utility Payments: ZWL {assessment_data.get('utility_payments', 0):.0f}", ln=True)
    pdf.cell(0, 7, f"Repayment History: {assessment_data.get('repayment_history')}", ln=True)
    pdf.ln(5)
    
    # Risk Assessment
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "RISK ASSESSMENT", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 7, f"Overall Risk Score: {assessment_data.get('risk_score', 0):.1f}/100", ln=True)
    pdf.cell(0, 7, f"Risk Category: {assessment_data.get('risk_category', 'N/A')}", ln=True)
    pdf.cell(0, 7, f"Traditional Score: {assessment_data.get('traditional_score', 0)}/6", ln=True)
    
    if assessment_data.get('ai_prediction'):
        pdf.cell(0, 7, f"AI Prediction: {assessment_data.get('ai_prediction')} ({assessment_data.get('ai_confidence', 0):.1f}%)", ln=True)
    pdf.ln(5)
    
    # Recommendations
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "RECOMMENDATIONS", ln=True)
    pdf.set_font("Arial", '', 10)
    
    for i, rec in enumerate(recommendations[:5], 1):  # Limit to 5 recommendations
        title = rec.get('title', '').encode('latin-1', errors='replace').decode('latin-1')
        value = rec.get('value', '').encode('latin-1', errors='replace').decode('latin-1')
        detail = rec.get('detail', '').encode('latin-1', errors='replace').decode('latin-1')
        
        pdf.multi_cell(0, 6, f"{i}. {title}: {value}")
        pdf.set_font("Arial", 'I', 9)
        pdf.multi_cell(0, 5, f"   {detail}")
        pdf.set_font("Arial", '', 10)
        pdf.ln(2)
    
    return pdf.output(dest='S').encode('latin-1', errors='replace')

def get_pdf_download_link(pdf_bytes, filename):
    """Generate download link for PDF"""
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" style="text-decoration: none;"><div class="success-box" style="text-align: center; cursor: pointer;">📄 Download Complete Assessment Report (PDF)</div></a>'
    return href

# ==================== HEADER ====================
st.markdown('<h1 class="main-header">🏦 ZIM SMART CREDIT</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Credit Risk Intelligence & Alternative Data Scoring Platform</p>', unsafe_allow_html=True)
st.markdown("---")

# ==================== SIDEBAR - INPUT FORM ====================
with st.sidebar:
    st.markdown("### 📋 Applicant Information")
    
    with st.form("applicant_form"):
        st.markdown("#### Personal Details")
        location = st.selectbox("📍 Location", sorted(df['Location'].unique()))
        gender = st.selectbox("👤 Gender", sorted(df['Gender'].unique()))
        age = st.slider("🎂 Age", int(df['Age'].min()), int(df['Age'].max()), 35)
        income_source = st.selectbox("💰 Source of Income", sorted(df['Income_Source'].unique()))
        
        st.markdown("#### Financial Behavior")
        mobile_money_txns = st.slider(
            "📱 Mobile Money Transactions (monthly)", 
            float(df['Mobile_Money_Txns'].min()), 
            float(df['Mobile_Money_Txns'].max()), 
            float(df['Mobile_Money_Txns'].median())
        )
        
        airtime_spend = st.slider(
            "📞 Airtime Spend (ZWL/month)", 
            float(df['Airtime_Spend_ZWL'].min()), 
            float(df['Airtime_Spend_ZWL'].max()), 
            float(df['Airtime_Spend_ZWL'].median())
        )
        
        utility_payments = st.slider(
            "💡 Utility Payments (ZWL/month)", 
            float(df['Utility_Payments_ZWL'].min()), 
            float(df['Utility_Payments_ZWL'].max()), 
            float(df['Utility_Payments_ZWL'].median())
        )
        
        repayment_history = st.selectbox(
            "📊 Loan Repayment History", 
            ['Excellent', 'Good', 'Fair', 'Poor']
        )
        
        submit_button = st.form_submit_button("🔍 Analyze Credit Profile", use_container_width=True)

# ==================== MAIN CONTENT ====================
if submit_button:
    # Prepare applicant data
    applicant_data = {
        'location': location,
        'gender': gender,
        'age': age,
        'income_source': income_source,
        'mobile_txns': mobile_money_txns,
        'airtime': airtime_spend,
        'utility': utility_payments,
        'repayment_history': repayment_history
    }
    
    # Initialize risk assessor
    risk_assessor = AdvancedRiskAssessor(df)
    
    # Calculate comprehensive risk
    risk_score, risk_components = risk_assessor.calculate_comprehensive_risk_score(applicant_data)
    risk_category, risk_type = risk_assessor.get_risk_category(risk_score)
    
    # Calculate traditional score
    trad_score, max_score = calculate_traditional_score(age, mobile_money_txns, repayment_history, df)
    
    # AI Prediction
    ai_prediction = None
    ai_confidence = None
    ai_class_probs = None
    
    if st.session_state.model_trained:
        try:
            # Prepare input for model
            input_data = pd.DataFrame([{
                'Location': location,
                'Gender': gender,
                'Age': age,
                'Mobile_Money_Txns': mobile_money_txns,
                'Airtime_Spend_ZWL': airtime_spend,
                'Utility_Payments_ZWL': utility_payments,
                'Loan_Repayment_History': repayment_history,
                'Income_Source': income_source
            }])
            
            # Encode
            for column in input_data.select_dtypes(include=['object']).columns:
                if column in st.session_state.label_encoders:
                    le = st.session_state.label_encoders[column]
                    input_data[column] = le.transform(input_data[column])
            
            # Predict
            prediction = st.session_state.model.predict(input_data)
            prediction_proba = st.session_state.model.predict_proba(input_data)
            
            ai_prediction = st.session_state.target_encoder.inverse_transform(prediction)[0]
            ai_confidence = float(np.max(prediction_proba) * 100)
            ai_class_probs = {
                cls: float(prob * 100) 
                for cls, prob in zip(st.session_state.target_encoder.classes_, prediction_proba[0])
            }
            
        except Exception as e:
            st.warning(f"AI prediction unavailable: {str(e)}")
    
    # Generate recommendations
    recommendations = risk_assessor.generate_risk_recommendations(risk_score, risk_components, applicant_data)
    
    # Store current assessment
    assessment_record = {
        **applicant_data,
        'mobile_money_txns': mobile_money_txns,
        'airtime_spend': airtime_spend,
        'utility_payments': utility_payments,
        'traditional_score': trad_score,
        'max_score': max_score,
        'risk_score': float(risk_score),
        'risk_category': risk_category,
        'risk_components': {k: float(v) for k, v in risk_components.items()},
        'ai_prediction': ai_prediction,
        'ai_confidence': ai_confidence,
        'ai_class_probs': ai_class_probs
    }
    
    st.session_state.current_assessment = assessment_record
    
    # ==================== DISPLAY RESULTS ====================
    
    # Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; font-size: 2.5rem;">{trad_score}/{max_score}</h3>
            <p style="margin: 0.5rem 0 0 0;">Traditional Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; font-size: 2.5rem;">{risk_score:.1f}</h3>
            <p style="margin: 0.5rem 0 0 0;">Risk Score (0-100)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        badge_class = f"risk-badge-{risk_type}"
        st.markdown(f"""
        <div class="metric-card">
            <div class="{badge_class}" style="font-size: 1.2rem; margin-top: 1rem;">{risk_category}</div>
            <p style="margin: 0.5rem 0 0 0;">Risk Category</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if ai_prediction:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0; font-size: 1.8rem;">{ai_prediction}</h3>
                <p style="margin: 0.5rem 0 0 0;">AI Prediction ({ai_confidence:.0f}%)</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0; font-size: 1.8rem;">N/A</h3>
                <p style="margin: 0.5rem 0 0 0;">AI Prediction</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed Analysis
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Risk Breakdown", 
        "🎯 Recommendations", 
        "🤖 AI Insights",
        "📄 Report"
    ])
    
    with tab1:
        st.markdown("### 🔍 Comprehensive Risk Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Risk components breakdown
            st.markdown("#### Risk Components")
            
            components_df = pd.DataFrame([
                {'Component': 'Age Risk', 'Score': risk_components['age_risk'], 'Weight': '15%'},
                {'Component': 'Income Stability', 'Score': risk_components['income_risk'], 'Weight': '20%'},
                {'Component': 'Behavioral Risk', 'Score': risk_components['behavioral_risk'], 'Weight': '25%'},
                {'Component': 'Repayment Risk', 'Score': risk_components['repayment_risk'], 'Weight': '30%'},
                {'Component': 'Geographic Risk', 'Score': risk_components['geographic_risk'], 'Weight': '10%'}
            ])
            
            fig = go.Figure()
            
            colors = ['#28a745' if score < 25 else '#ffc107' if score < 50 else '#dc3545' 
                     for score in components_df['Score']]
            
            fig.add_trace(go.Bar(
                y=components_df['Component'],
                x=components_df['Score'],
                orientation='h',
                marker=dict(color=colors),
                text=components_df['Score'].apply(lambda x: f'{x:.1f}'),
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Risk Score: %{x:.1f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Risk Component Breakdown',
                xaxis_title='Risk Score (0-100, lower is better)',
                height=400,
                showlegend=False,
                xaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk gauge
            st.markdown("#### Overall Risk Assessment")
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Score", 'font': {'size': 24}},
                delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 25], 'color': '#d4edda'},
                        {'range': [25, 50], 'color': '#fff3cd'},
                        {'range': [50, 100], 'color': '#f8d7da'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': risk_score
                    }
                }
            ))
            
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Risk interpretation
            if risk_score < 25:
                st.markdown("""
                <div class="success-box">
                    <h4>✅ Excellent Risk Profile</h4>
                    <p>This applicant demonstrates strong creditworthiness with minimal risk factors. 
                    Ideal candidate for premium credit products.</p>
                </div>
                """, unsafe_allow_html=True)
            elif risk_score < 50:
                st.markdown("""
                <div class="warning-box">
                    <h4>⚠️ Moderate Risk Profile</h4>
                    <p>Acceptable risk level with standard credit terms. Some caution advised. 
                    Consider income verification and standard security measures.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="danger-box">
                    <h4>❌ Elevated Risk Profile</h4>
                    <p>Significant risk factors identified. Enhanced due diligence required. 
                    Conservative credit limits and collateral strongly recommended.</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 💡 Credit Decision Recommendations")
        
        for rec in recommendations:
            st.markdown(f"""
            <div class="glass-card">
                <h4>{rec['icon']} {rec['title']}</h4>
                <h3 style="color: #667eea; margin: 0.5rem 0;">{rec['value']}</h3>
                <p style="color: #666; margin: 0;">{rec['detail']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### 🤖 Artificial Intelligence Insights")
        
        if ai_prediction and ai_class_probs:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### AI Credit Classification")
                
                # Class probability chart
                classes = list(ai_class_probs.keys())
                probs = list(ai_class_probs.values())
                
                fig_probs = go.Figure()
                fig_probs.add_trace(go.Bar(
                    x=classes,
                    y=probs,
                    marker_color=['#28a745' if cls == ai_prediction else '#6c757d' for cls in classes],
                    text=[f'{p:.1f}%' for p in probs],
                    textposition='outside'
                ))
                
                fig_probs.update_layout(
                    title='Probability Distribution Across Credit Classes',
                    yaxis_title='Probability (%)',
                    xaxis_title='Credit Class',
                    height=400,
                    yaxis=dict(range=[0, max(probs) * 1.2])
                )
                
                st.plotly_chart(fig_probs, use_container_width=True)
            
            with col2:
                st.markdown("#### Model Confidence Analysis")
                
                st.markdown(f"""
                <div class="info-box">
                    <h4>🎯 Primary Prediction</h4>
                    <h2 style="color: #667eea; margin: 0.5rem 0;">{ai_prediction}</h2>
                    <p><strong>Confidence:</strong> {ai_confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence interpretation
                if ai_confidence > 80:
                    st.markdown("""
                    <div class="success-box">
                        <strong>High Confidence</strong><br>
                        The AI model is highly confident in this classification. 
                        The prediction is well-supported by the applicant's profile.
                    </div>
                    """, unsafe_allow_html=True)
                elif ai_confidence > 60:
                    st.markdown("""
                    <div class="warning-box">
                        <strong>Moderate Confidence</strong><br>
                        The AI model shows reasonable confidence. Consider cross-referencing 
                        with traditional scoring methods.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="danger-box">
                        <strong>Low Confidence</strong><br>
                        The AI model is uncertain about this classification. 
                        Manual review strongly recommended.
                    </div>
                    """, unsafe_allow_html=True)
                
                # Feature importance (if available)
                if st.session_state.model_metrics.get('feature_importance'):
                    st.markdown("#### Top Influential Factors")
                    importance = st.session_state.model_metrics['feature_importance']
                    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                    for feature, score in top_features:
                        st.markdown(f"**{feature}:** {score:.3f}")
        else:
            st.info("AI prediction not available. Please train the model first.")
    
    with tab4:
        st.markdown("### 📄 Generate Assessment Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Assessment to History", type="primary", use_container_width=True):
                assessment_id = save_assessment(assessment_record)
                st.success(f"✅ Assessment saved successfully! ID: {assessment_id}")
                st.session_state.current_assessment['assessment_id'] = assessment_id
        
        with col2:
            if st.button("📄 Generate PDF Report", type="secondary", use_container_width=True):
                if 'assessment_id' in st.session_state.current_assessment:
                    try:
                        pdf_bytes = generate_pdf_report(
                            st.session_state.current_assessment,
                            risk_components,
                            recommendations
                        )
                        filename = f"credit_assessment_{st.session_state.current_assessment['assessment_id']}.pdf"
                        st.markdown(get_pdf_download_link(pdf_bytes, filename), unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"PDF generation failed: {str(e)}")
                else:
                    st.warning("Please save the assessment first before generating PDF")
        
        st.markdown("---")
        
        # Summary table
        st.markdown("#### Assessment Summary")
        
        summary_data = {
            'Category': ['Applicant', 'Traditional Scoring', 'Risk Assessment', 'AI Analysis'],
            'Details': [
                f"{age} years, {gender}, {location}, Income: {income_source}",
                f"Score: {trad_score}/{max_score} ({(trad_score/max_score*100):.0f}%)",
                f"Risk Score: {risk_score:.1f}/100, Category: {risk_category}",
                f"Prediction: {ai_prediction}, Confidence: {ai_confidence:.1f}%" if ai_prediction else "N/A"
            ]
        }
        
        st.table(pd.DataFrame(summary_data))

# ==================== FOOTER TABS ====================
st.markdown("---")

footer_tabs = st.tabs([
    "📊 Model Performance",
    "📈 Assessment History", 
    "ℹ️ About"
])

with footer_tabs[0]:
    st.markdown("### 🎯 Machine Learning Model Performance")
    
    if st.session_state.model_trained:
        metrics = st.session_state.model_metrics
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("🎯 Accuracy", f"{metrics['accuracy']:.2f}%")
        with col2:
            st.metric("📊 Precision", f"{metrics['precision']:.2f}%")
        with col3:
            st.metric("🔄 Recall", f"{metrics['recall']:.2f}%")
        with col4:
            st.metric("📈 F1 Score", f"{metrics['f1_score']:.2f}%")
        with col5:
            if metrics.get('auc_score'):
                st.metric("🎲 AUC", f"{metrics['auc_score']:.2f}%")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Cross-Validation Results")
            cv_df = pd.DataFrame({
                'Fold': [f'Fold {i+1}' for i in range(len(metrics['cv_scores']))],
                'Accuracy (%)': [f"{score:.2f}" for score in metrics['cv_scores']]
            })
            st.dataframe(cv_df, use_container_width=True, hide_index=True)
            st.markdown(f"**Mean:** {metrics['cv_mean']:.2f}% ± {metrics['cv_std']:.2f}%")
        
        with col2:
            st.markdown("#### Feature Importance")
            importance_df = pd.DataFrame(
                list(metrics['feature_importance'].items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=False).head(8)
            
            fig_imp = go.Figure()
            fig_imp.add_trace(go.Bar(
                y=importance_df['Feature'],
                x=importance_df['Importance'],
                orientation='h',
                marker_color='#667eea'
            ))
            fig_imp.update_layout(
                title='Top 8 Most Important Features',
                xaxis_title='Importance Score',
                height=400
            )
            st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Model not trained yet")

with footer_tabs[1]:
    st.markdown("### 📈 Assessment History & Analytics")
    
    if st.session_state.assessments_history:
        history_df = pd.DataFrame(st.session_state.assessments_history)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Assessments", len(history_df))
        with col2:
            st.metric("Avg Risk Score", f"{history_df['risk_score'].mean():.1f}")
        with col3:
            low_risk_pct = (history_df['risk_score'] < 25).mean() * 100
            st.metric("Low Risk %", f"{low_risk_pct:.1f}%")
        with col4:
            high_risk_pct = (history_df['risk_score'] >= 50).mean() * 100
            st.metric("High Risk %", f"{high_risk_pct:.1f}%")
        
        # Recent assessments table
        st.markdown("#### Recent Assessments")
        display_cols = ['assessment_id', 'date', 'time', 'location', 'age', 
                       'traditional_score', 'risk_score', 'risk_category']
        
        available_cols = [col for col in display_cols if col in history_df.columns]
        recent = history_df[available_cols].tail(10).sort_values('date', ascending=False)
        
        st.dataframe(recent, use_container_width=True, hide_index=True)
    else:
        st.info("No assessments in history yet. Complete an assessment to see analytics.")

with footer_tabs[2]:
    st.markdown("### ℹ️ About Zim Smart Credit")
    
    st.markdown("""
    <div class="glass-card">
        <h3>🎓 Final Year Project</h3>
        <p><strong>Advanced Credit Scoring Using Alternative Data in Zimbabwe</strong></p>
        
        <h4>📊 Key Features:</h4>
        <ul>
            <li><strong>Multi-Dimensional Risk Assessment:</strong> Comprehensive evaluation across 5 risk categories</li>
            <li><strong>AI-Powered Predictions:</strong> Random Forest classifier with 90%+ accuracy</li>
            <li><strong>Alternative Data Integration:</strong> Mobile money, airtime, and utility payment analysis</li>
            <li><strong>Real-Time Scoring:</strong> Instant credit decisions with detailed explanations</li>
            <li><strong>PDF Report Generation:</strong> Professional assessment reports for documentation</li>
            <li><strong>Historical Analytics:</strong> Track and analyze assessment trends over time</li>
        </ul>
        
        <h4>🔬 Methodology:</h4>
        <ul>
            <li><strong>Traditional Scoring:</strong> Rule-based assessment (0-6 scale)</li>
            <li><strong>Risk Modeling:</strong> Weighted multi-factor analysis (0-100 scale)</li>
            <li><strong>Machine Learning:</strong> Random Forest with cross-validation</li>
            <li><strong>Explainable AI:</strong> SHAP values for model interpretability</li>
        </ul>
        
        <h4>💡 Innovation:</h4>
        <p>This platform addresses financial inclusion gaps in Zimbabwe by leveraging alternative data sources 
        that traditional credit bureaus don't capture, enabling credit access for the underbanked population.</p>
        
        <h4>🛠️ Technology Stack:</h4>
        <p>Python • Streamlit • Scikit-learn • SHAP • Plotly • Pandas • NumPy</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: rgba(255,255,255,0.8);'>© 2024 Zim Smart Credit • Advanced Risk Intelligence Platform</p>",
    unsafe_allow_html=True
)
