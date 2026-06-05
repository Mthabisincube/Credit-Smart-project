import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import uuid

# Page configuration (unchanged)
st.set_page_config(
    page_title="Zim Smart Credit App",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= LIGHT UI CSS (unchanged) =================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4ecf7 50%, #d6e4f0 100%);
        background-attachment: fixed;
    }
    
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main-header {
        font-size: 2.8rem;
        text-align: center;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #5a6c7e;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .metric-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
        border-left: 4px solid;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #1a1a2e;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        font-weight: 500;
    }
    
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        padding-left: 0.8rem;
        border-left: 4px solid #3498db;
    }
    
    .glass-panel {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.5);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-right: 1px solid rgba(0,0,0,0.05);
    }
    
    [data-testid="stSidebar"] label {
        color: #2c3e50 !important;
        font-weight: 500;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(52,152,219,0.4);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(255,255,255,0.5);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        color: #2c3e50;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white !important;
    }
    
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    
    .badge-success { background: #d4edda; color: #155724; }
    .badge-warning { background: #fff3cd; color: #856404; }
    .badge-danger { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🏦 Zim Smart Credit App</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">🚀 AI-Powered Credit Scoring | Alternative Data Intelligence | Financial Inclusion for Zimbabwe</p>', unsafe_allow_html=True)

# Session state initialization (unchanged except for score scale note)
if 'assessments_history' not in st.session_state:
    st.session_state.assessments_history = []

if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.label_encoders = {}
    st.session_state.target_encoder = None
    st.session_state.model_metrics = {}
    st.session_state.model_trained = False
    st.session_state.feature_columns = None

if 'assessment_results' not in st.session_state:
    st.session_state.assessment_results = {
        'score': 0,          # will now be 0-100
        'max_score': 100,
        'predicted_class': None,
        'confidence': None,
        'risk_level': 'Medium',
        'assessment_id': None,
        'timestamp': None
    }

# Load data (unchanged)
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/Mthabisincube/Credit-Smart-project/refs/heads/master/smart_credit_scoring_zimbabwe.csv")
    np.random.seed(42)
    income_sources = ['Formal Employment', 'Informal Business', 'Farming', 'Remittances', 'Other']
    df['Income_Source'] = np.random.choice(income_sources, size=len(df), p=[0.4, 0.25, 0.15, 0.1, 0.1])
    df.rename(columns={'Airtime_Spend_ZWL': 'Airtime_Spend_USD', 'Utility_Payments_ZWL': 'Utility_Payments_USD'}, inplace=True)
    return df

df = load_data()

# Map locations to provinces (unchanged)
location_to_province = {
    'Harare': 'Harare', 'Bulawayo': 'Bulawayo', 'Mutare': 'Manicaland',
    'Marondera': 'Mashonaland East', 'Chinhoyi': 'Mashonaland West',
    'Bindura': 'Mashonaland Central', 'Masvingo': 'Masvingo',
    'Gweru': 'Midlands', 'Kwekwe': 'Midlands', 'Hwange': 'Matabeleland North',
    'Victoria Falls': 'Matabeleland North', 'Gwanda': 'Matabeleland South'
}
df['Province'] = df['Location'].map(location_to_province).fillna('Other')
df = df[df['Province'] != 'Other']

# Province metrics (unchanged)
province_metrics = df.groupby('Province').agg({
    'Credit_Score': ['mean', 'count', lambda x: (x < 3).mean() * 100]
}).round(2)
province_metrics.columns = ['avg_score', 'count', 'high_risk_pct']
province_metrics = province_metrics.reset_index()

# -------------------------------------------------------------------
#  NEW TRANSPARENT CREDIT SCORING LOGIC (0–100)
# -------------------------------------------------------------------
def calculate_transparent_score(age, mobile_money, airtime, utility, repayment_history, income_source):
    """
    Transparent, rule‑based credit score on a 0‑100 scale.
    Each feature contributes a maximum, determined by domain logic and data‑driven percentiles.
    """
    score = 0.0
    max_score = 0.0
    details = {}   # for display later

    # 1. Loan Repayment History (0‑40 points)
    rep_map = {'Poor': 0, 'Fair': 10, 'Good': 25, 'Excellent': 40}
    pts = rep_map[repayment_history]
    score += pts
    max_score += 40
    details['Loan Repayment History'] = pts

    # 2. Mobile Money Transactions (0‑25 points) – proxy for income / digital footprint
    # Use quantiles from the dataset
    p25 = df['Mobile_Money_Txns'].quantile(0.25)
    p50 = df['Mobile_Money_Txns'].quantile(0.50)
    p75 = df['Mobile_Money_Txns'].quantile(0.75)
    if mobile_money >= p75:
        pts_mm = 25
    elif mobile_money >= p50:
        pts_mm = 15
    elif mobile_money >= p25:
        pts_mm = 5
    else:
        pts_mm = 0
    score += pts_mm
    max_score += 25
    details['Mobile Money Transactions'] = pts_mm

    # 3. Airtime Spend (0‑10 points) – communication behaviour
    p25_air = df['Airtime_Spend_USD'].quantile(0.25)
    p50_air = df['Airtime_Spend_USD'].quantile(0.50)
    p75_air = df['Airtime_Spend_USD'].quantile(0.75)
    if airtime >= p75_air:
        pts_air = 10
    elif airtime >= p50_air:
        pts_air = 6
    elif airtime >= p25_air:
        pts_air = 2
    else:
        pts_air = 0
    score += pts_air
    max_score += 10
    details['Airtime Spend'] = pts_air

    # 4. Utility Payments (0‑10 points) – bill payment reliability
    p25_util = df['Utility_Payments_USD'].quantile(0.25)
    p50_util = df['Utility_Payments_USD'].quantile(0.50)
    p75_util = df['Utility_Payments_USD'].quantile(0.75)
    if utility >= p75_util:
        pts_util = 10
    elif utility >= p50_util:
        pts_util = 6
    elif utility >= p25_util:
        pts_util = 2
    else:
        pts_util = 0
    score += pts_util
    max_score += 10
    details['Utility Payments'] = pts_util

    # 5. Age (0‑10 points) – reasonable range
    if 30 <= age <= 45:
        pts_age = 10
    elif (25 <= age < 30) or (45 < age <= 55):
        pts_age = 6
    else:
        pts_age = 2   # 18‑24 or 56+ still gets a small base
    score += pts_age
    max_score += 10
    details['Age'] = pts_age

    # 6. Income Source (0‑5 points) – stability indicator
    inc_map = {
        'Formal Employment': 5,
        'Remittances': 3,
        'Informal Business': 2,
        'Farming': 1,
        'Other': 0
    }
    pts_inc = inc_map.get(income_source, 0)
    score += pts_inc
    max_score += 5
    details['Income Source'] = pts_inc

    # Normalise to 0‑100 (in case max_score is not 100 exactly)
    final_score = round((score / max_score) * 100) if max_score > 0 else 0
    return final_score, details
# -------------------------------------------------------------------

# Retrain the AI model WITHOUT gender, and include Income_Source
def train_model():
    try:
        # UPDATED feature list – removed Gender, added Income_Source
        feature_cols = [
            'Location', 'Age', 'Mobile_Money_Txns',
            'Airtime_Spend_USD', 'Utility_Payments_USD',
            'Loan_Repayment_History', 'Income_Source'
        ]
        
        X = df[feature_cols].copy()
        
        # Convert original Credit_Score (1-6) to categories for classification
        def score_to_category(score):
            if score <= 2:
                return 'Poor'
            elif score == 3:
                return 'Fair'
            elif score == 4:
                return 'Good'
            else:
                return 'Excellent'
        
        y_categorical = df['Credit_Score'].apply(score_to_category)
        
        # Encode categorical features (Location, Loan_Repayment_History, Income_Source)
        label_encoders = {}
        categorical_cols = ['Location', 'Loan_Repayment_History', 'Income_Source']
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Encode target
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y_categorical)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(
            n_estimators=100, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, class_weight='balanced'
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred, average='weighted') * 100
        recall = recall_score(y_test, y_pred, average='weighted') * 100
        f1 = f1_score(y_test, y_pred, average='weighted') * 100
        
        st.session_state.model = model
        st.session_state.label_encoders = label_encoders
        st.session_state.target_encoder = target_encoder
        st.session_state.model_trained = True
        st.session_state.feature_columns = feature_cols
        
        # Store metrics and feature importance
        importances = dict(zip(feature_cols, model.feature_importances_))
        st.session_state.model_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'feature_importance': importances
        }
        return True
    except Exception as e:
        st.error(f"Training error: {str(e)}")
        return False

# Train model if needed
if not st.session_state.model_trained:
    with st.spinner("🤖 Training AI Model (without gender)..."):
        train_model()

# Updated predict function (no gender, includes income_source)
def predict_credit(input_data):
    if not st.session_state.model_trained:
        return "Unknown", 0
    try:
        feature_cols = st.session_state.feature_columns
        X_input = pd.DataFrame([[
            input_data['Location'],
            input_data['Age'],
            input_data['Mobile_Money_Txns'],
            input_data['Airtime_Spend_USD'],
            input_data['Utility_Payments_USD'],
            input_data['Loan_Repayment_History'],
            input_data['Income_Source']
        ]], columns=feature_cols)
        
        for col in ['Location', 'Loan_Repayment_History', 'Income_Source']:
            if col in st.session_state.label_encoders:
                le = st.session_state.label_encoders[col]
                X_input[col] = le.transform(X_input[col].astype(str))
        
        prediction = st.session_state.model.predict(X_input)[0]
        proba = st.session_state.model.predict_proba(X_input)[0]
        confidence = max(proba) * 100
        predicted_class = st.session_state.target_encoder.inverse_transform([prediction])[0]
        return predicted_class, confidence
    except Exception as e:
        return "Unknown", 0

# Helpers (updated risk level to use new score scale)
def get_risk_level(score):
    # score now out of 100
    if score >= 75:
        return "Low"
    elif score >= 50:
        return "Medium"
    else:
        return "High"

def save_assessment(assessment_data):
    assessment_data['assessment_id'] = str(uuid.uuid4())[:8]
    assessment_data['timestamp'] = datetime.now().isoformat()
    assessment_data['date'] = datetime.now().strftime('%Y-%m-%d')
    st.session_state.assessments_history.append(assessment_data.copy())
    cutoff = datetime.now() - timedelta(days=30)
    st.session_state.assessments_history = [
        a for a in st.session_state.assessments_history 
        if datetime.fromisoformat(a['timestamp']) > cutoff
    ]
    return assessment_data['assessment_id']

def get_monthly_stats():
    if not st.session_state.assessments_history:
        return None
    df_assess = pd.DataFrame(st.session_state.assessments_history)
    cutoff = datetime.now() - timedelta(days=30)
    df_assess['datetime'] = pd.to_datetime(df_assess['timestamp'])
    monthly = df_assess[df_assess['datetime'] >= cutoff]
    if len(monthly) == 0:
        return None
    return {
        'total': len(monthly),
        'avg_score': monthly['score'].mean(),          # now 0-100
        'approval_rate': (monthly['score'] >= 50).mean() * 100,
        'high_risk': (monthly['score'] < 50).mean() * 100
    }

# ---------------- SIDEBAR (gender removed) ----------------
with st.sidebar:
    st.markdown("### 🎯 Applicant Information")
    st.markdown("---")
    
    Location = st.selectbox("📍 Location", sorted(df['Location'].unique()))
    # GENDER FIELD REMOVED – no longer asked
    Age = st.slider("🎂 Age", 18, 80, 35)
    
    st.markdown("### 💰 Financial Behavior")
    
    Mobile_Money_Txns = st.slider("📱 Mobile Money Transactions", 0.0, 300.0, 75.0)
    Airtime_Spend_USD = st.slider("📞 Airtime Spend (USD)", 0.0, 300.0, 50.0)
    Utility_Payments_USD = st.slider("💡 Utility Payments (USD)", 0.0, 300.0, 80.0)
    Loan_Repayment_History = st.selectbox("📊 Loan Repayment History", ['Poor', 'Fair', 'Good', 'Excellent'])
    Income_Source = st.selectbox("💰 Source of Income", ['Informal Business', 'Farming', 'Remittances', 'Formal Employment', 'Other'])
# -----------------------------------------------------------

# Calculate the new transparent score (0-100)
transparent_score, score_details = calculate_transparent_score(
    Age, Mobile_Money_Txns, Airtime_Spend_USD, Utility_Payments_USD,
    Loan_Repayment_History, Income_Source
)

risk_level = get_risk_level(transparent_score)

# Get AI prediction (without gender)
predicted_class, confidence = predict_credit({
    'Location': Location,
    'Age': Age,
    'Mobile_Money_Txns': Mobile_Money_Txns,
    'Airtime_Spend_USD': Airtime_Spend_USD,
    'Utility_Payments_USD': Utility_Payments_USD,
    'Loan_Repayment_History': Loan_Repayment_History,
    'Income_Source': Income_Source
})

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard", "🎯 Assessments", "🔍 Analysis", "📋 Monthly Reports"
])

# ================= TAB 1: DASHBOARD (unchanged) =================
with tab1:
    st.markdown('<h2 style="color: #2c3e50; font-weight: 800; margin-bottom: 0px;">🌍 Zimbabwe Smart Credit Overview</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #7f8c8d; font-size: 1.1rem; margin-bottom: 2rem;">A robust, world-class AI-powered credit scoring engine leveraging <b>alternative data</b> for maximum financial inclusion.</p>', unsafe_allow_html=True)
    
    st.markdown("### 📊 Platform Analytics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #3498db; background: linear-gradient(135deg, white, #f8fbff);">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">📊 Processed Records</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #9b59b6; background: linear-gradient(135deg, white, #fcf8ff);">
            <div class="metric-value">{df['Credit_Score'].nunique()}</div>
            <div class="metric-label">🎯 Predictive Tiers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #2ecc71; background: linear-gradient(135deg, white, #f4fff8);">
            <div class="metric-value">{len(st.session_state.assessments_history)}</div>
            <div class="metric-label">📈 Assessments (30d)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        approval_rate = (df['Credit_Score'] >= 3).mean() * 100
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #e67e22; background: linear-gradient(135deg, white, #fffcf8);">
            <div class="metric-value">{approval_rate:.0f}%</div>
            <div class="metric-label">✅ Base Approval Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Score Stratification")
        score_counts = df['Credit_Score'].value_counts().sort_index()
        colors = ['#e74c3c' if x <= 2 else '#f39c12' if x <= 3 else '#2ecc71' for x in score_counts.index]
        fig_score = go.Figure(data=[go.Bar(
            x=['Poor (1)', 'Fair (2)', 'Avg (3)', 'Good (4)', 'V.Good (5)', 'Exc (6)'], 
            y=score_counts.values, 
            marker_color=colors,
            text=score_counts.values, textposition='auto'
        )])
        fig_score.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig_score, use_container_width=True)
    
    with col2:
        st.markdown("#### 🌍 Geographic Footprint")
        location_counts = df['Location'].value_counts().head(6)
        fig_loc = go.Figure(data=[go.Bar(
            x=location_counts.values, 
            y=location_counts.index, 
            orientation='h', 
            marker_color='#3498db',
            text=location_counts.values, textposition='auto'
        )])
        fig_loc.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=0, l=0, r=0), yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_loc, use_container_width=True)

# ================= TAB 2: ASSESSMENTS (updated) =================
with tab2:
    st.markdown("### 🎯 Credit Assessment")
    
    # Input summary (gender removed)
    input_data = pd.DataFrame({
        "Feature": ["📍 Location", "🎂 Age", "📱 Mobile Transactions", "📞 Airtime Spend",
                    "💡 Utility Payments", "📊 Repayment History", "💰 Income Source"],
        "Value": [Location, f"{Age}", f"{Mobile_Money_Txns:.0f}",
                  f"${Airtime_Spend_USD:.2f}", f"${Utility_Payments_USD:.2f}",
                  Loan_Repayment_History, Income_Source]
    })
    st.dataframe(input_data, use_container_width=True, hide_index=True)
    
    # Score display – now out of 100
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("📈 Credit Score", f"{transparent_score}/100")
        st.progress(transparent_score / 100)
    with col2:
        if transparent_score >= 75:
            st.success("### ✅ EXCELLENT CREDITWORTHINESS")
            # REMOVED st.balloons()
        elif transparent_score >= 50:
            st.warning("### ⚠️ MODERATE RISK PROFILE")
        else:
            st.error("### ❌ HIGHER RISK PROFILE")
        st.write(f"**Risk Level:** {risk_level}")
    
    # ---- TRANSPARENT SCORE BREAKDOWN ----
    st.markdown("#### 🔍 How was this score calculated?")
    with st.expander("See detailed breakdown", expanded=True):
        col_detail1, col_detail2 = st.columns(2)
        with col_detail1:
            for feat in ['Loan Repayment History', 'Mobile Money Transactions', 'Airtime Spend']:
                pts = score_details[feat]
                max_pts = {'Loan Repayment History': 40, 'Mobile Money Transactions': 25, 'Airtime Spend': 10}[feat]
                st.markdown(f"**{feat}:** {pts}/{max_pts} points")
                st.progress(pts / max_pts)
        with col_detail2:
            for feat in ['Utility Payments', 'Age', 'Income Source']:
                pts = score_details[feat]
                max_pts = {'Utility Payments': 10, 'Age': 10, 'Income Source': 5}[feat]
                st.markdown(f"**{feat}:** {pts}/{max_pts} points")
                st.progress(pts / max_pts)
    
    # AI model supplement (optional)
    st.markdown("#### 🤖 AI Model Prediction (Random Forest)")
    st.write(f"**Predicted Class:** {predicted_class} | **Confidence:** {confidence:.1f}%")
    if predicted_class != "Unknown":
        st.caption("The AI model is trained on thousands of historical records and considers the same features (excluding gender). "
                   "It often agrees with the transparent score, providing a second opinion.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Assessment", type="primary", use_container_width=True):
            assessment_data = {
                'location': Location,
                'age': Age,
                'mobile_money_txns': Mobile_Money_Txns,
                'airtime_spend': Airtime_Spend_USD,
                'utility_payments': Utility_Payments_USD,
                'repayment_history': Loan_Repayment_History,
                'income_source': Income_Source,
                'score': transparent_score,
                'max_score': 100,
                'risk_level': risk_level,
                'predicted_class': predicted_class,
                'confidence': confidence
            }
            assessment_id = save_assessment(assessment_data)
            st.success(f"✅ Assessment saved! ID: {assessment_id}")
            st.rerun()
    
    st.markdown("### 📝 Actionable Recommendations")
    recs = []
    
    if Loan_Repayment_History == "Poor":
        recs.append("❌ **Critical:** Poor loan repayment history. Requires intense scrutiny or decline.")
    elif Loan_Repayment_History == "Excellent":
        recs.append("✅ **Strength:** Excellent repayment record – prime borrower candidate.")
        
    median_txns = df['Mobile_Money_Txns'].median()
    if Mobile_Money_Txns < (median_txns * 0.5):
        recs.append("⚠️ **Digital Footprint:** Very low mobile money activity. Request alternative income proof.")
    elif Mobile_Money_Txns > (median_txns * 1.5):
        recs.append("✅ **Digital Footprint:** High transaction volume indicates strong cash flow.")
    
    if Utility_Payments_USD == 0:
        recs.append("⚠️ **Verifications:** No utility payments on record – manual KYC recommended.")
    
    if transparent_score >= 75:
        recs.append("🎯 **Final:** Approve. Applicant qualifies for premium limits (ZWL 50,000+) at prime rates.")
        st.success("\n\n".join(recs))
    elif transparent_score >= 50:
        recs.append("🎯 **Final:** Conditional Approval. Start with ZWL 5,000 – ZWL 15,000 limit, review after 6 months.")
        st.warning("\n\n".join(recs))
    else:
        recs.append("🎯 **Final:** Decline. Risk profile below minimum threshold. Advise building credit history.")
        st.error("\n\n".join(recs))

# ================= TAB 3: ANALYSIS (unchanged but gender‑free) =================
with tab3:
    st.markdown("### 🔍 Data Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Age Distribution")
        fig = px.histogram(df, x='Age', nbins=20, title='Age Distribution', color_discrete_sequence=['#3498db'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        # Gender pie chart removed – replace with something else, e.g. Income Source distribution
        st.markdown("#### Income Source Distribution")
        inc_counts = df['Income_Source'].value_counts()
        fig = go.Figure(data=[go.Pie(labels=inc_counts.index, values=inc_counts.values, hole=0.3)])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### Credit Score by Location")
    location_scores = df.groupby('Location')['Credit_Score'].mean().sort_values(ascending=True)
    colors_loc = ['#e74c3c' if x < 3 else '#f39c12' if x < 4 else '#2ecc71' for x in location_scores.values]
    fig = go.Figure(data=[go.Bar(x=location_scores.values, y=location_scores.index, orientation='h', marker_color=colors_loc)])
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# ================= TAB 4: MONTHLY REPORTS (unchanged logic but score scale handled) =================
with tab4:
    st.markdown("### 📋 Monthly Assessment Reports")
    stats = get_monthly_stats()
    
    if stats:
        st.markdown('<div class="glass-panel" style="margin-bottom: 2rem;">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #3498db;">
                <div class="metric-value">{stats['total']}</div>
                <div class="metric-label">📋 Total Applications</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #9b59b6;">
                <div class="metric-value">{stats['avg_score']:.1f}/100</div>
                <div class="metric-label">📈 Average Score</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #2ecc71;">
                <div class="metric-value">{stats['approval_rate']:.1f}%</div>
                <div class="metric-label">✅ Approval Rate</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #e74c3c;">
                <div class="metric-value">{stats['high_risk']:.1f}%</div>
                <div class="metric-label">⚠️ High Risk Rate</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if len(st.session_state.assessments_history) > 0:
            df_history = pd.DataFrame(st.session_state.assessments_history)
            df_history['date'] = pd.to_datetime(df_history['timestamp']).dt.date
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### 📈 Assessment Score Trends")
                daily_scores = df_history.groupby('date')['score'].mean().reset_index()
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=daily_scores['date'], y=daily_scores['score'], 
                    mode='lines+markers', 
                    line=dict(color='#3498db', width=3),
                    marker=dict(size=8, color='#2980b9'),
                    fill='tozeroy', fillcolor='rgba(52, 152, 219, 0.1)'
                ))
                fig_trend.add_hline(y=50, line_dash="dash", line_color="#e74c3c", annotation_text="Approval Threshold (50)")
                fig_trend.update_layout(
                    height=350, margin=dict(l=0, r=0, t=30, b=0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(range=[0, 105])
                )
                st.plotly_chart(fig_trend, use_container_width=True)
                
            with col2:
                st.markdown("#### 🎯 Risk Distribution")
                risk_counts = df_history['risk_level'].value_counts()
                color_map = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
                colors = [color_map.get(x, '#95a5a6') for x in risk_counts.index]
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=risk_counts.index, values=risk_counts.values, 
                    hole=0.5, marker_colors=colors
                )])
                fig_pie.update_layout(
                    height=350, margin=dict(l=0, r=0, t=30, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### 🗄️ Assessment Register")
            display_cols = ['assessment_id', 'date', 'location', 'score', 'risk_level']
            st.dataframe(
                df_history[display_cols].sort_values('date', ascending=False), 
                use_container_width=True, hide_index=True
            )
            
            csv = df_history.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Monthly Report (CSV)",
                data=csv,
                file_name=f"monthly_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("📭 No assessments recorded in the last 30 days. Save an assessment on the Assessment tab to populate this dashboard!")

# Footer (unchanged)
st.markdown("---")
st.markdown("### 💡 About Zim Smart Credit")
st.markdown("Leveraging alternative data (mobile money, utility payments, airtime usage) to provide fair and inclusive credit scoring for Zimbabweans.")
