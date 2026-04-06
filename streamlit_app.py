import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import uuid
import requests

# Page configuration
st.set_page_config(
    page_title="Zim Smart Credit App",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= LIGHT UI CSS =================
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

# Initialize session state
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
        'score': 0, 'max_score': 6, 'predicted_class': None,
        'confidence': None, 'risk_level': 'Medium', 'assessment_id': None, 'timestamp': None
    }

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/Mthabisincube/Credit-Smart-project/refs/heads/master/smart_credit_scoring_zimbabwe.csv")
    np.random.seed(42)
    income_sources = ['Formal Employment', 'Informal Business', 'Farming', 'Remittances', 'Other']
    df['Income_Source'] = np.random.choice(income_sources, size=len(df), p=[0.4, 0.25, 0.15, 0.1, 0.1])
    df.rename(columns={'Airtime_Spend_ZWL': 'Airtime_Spend_USD', 'Utility_Payments_ZWL': 'Utility_Payments_USD'}, inplace=True)
    return df

df = load_data()

# Map locations to provinces
location_to_province = {
    'Harare': 'Harare', 'Bulawayo': 'Bulawayo', 'Mutare': 'Manicaland',
    'Marondera': 'Mashonaland East', 'Chinhoyi': 'Mashonaland West',
    'Bindura': 'Mashonaland Central', 'Masvingo': 'Masvingo',
    'Gweru': 'Midlands', 'Kwekwe': 'Midlands', 'Hwange': 'Matabeleland North',
    'Victoria Falls': 'Matabeleland North', 'Gwanda': 'Matabeleland South'
}
df['Province'] = df['Location'].map(location_to_province).fillna('Other')
df = df[df['Province'] != 'Other']

# Calculate province-level metrics
province_metrics = df.groupby('Province').agg({
    'Credit_Score': ['mean', 'count', lambda x: (x < 3).mean() * 100]
}).round(2)
province_metrics.columns = ['avg_score', 'count', 'high_risk_pct']
province_metrics = province_metrics.reset_index()

# Train model function
def train_model():
    try:
        feature_cols = ['Location', 'Gender', 'Age', 'Mobile_Money_Txns', 
                       'Airtime_Spend_USD', 'Utility_Payments_USD', 'Loan_Repayment_History']
        
        X = df[feature_cols].copy()
        
        # Convert Credit_Score to categorical classes for classification
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
        
        # Encode categorical features
        label_encoders = {}
        categorical_cols = ['Location', 'Gender', 'Loan_Repayment_History']
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Encode target
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y_categorical)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
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
        
        st.session_state.model_metrics = {
            'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1_score': f1, 'feature_importance': dict(zip(feature_cols, model.feature_importances_))
        }
        return True
    except Exception as e:
        st.error(f"Training error: {str(e)}")
        return False

# Train model if not already trained
if not st.session_state.model_trained:
    with st.spinner("🤖 Training AI Model..."):
        train_model()

# Helper functions
def get_risk_level(score):
    if score >= 5: return "Low"
    elif score >= 3: return "Medium"
    else: return "High"

def predict_credit(input_data):
    if not st.session_state.model_trained:
        return "Unknown", 0
    try:
        feature_cols = st.session_state.feature_columns
        X_input = pd.DataFrame([[
            input_data['Location'], input_data['Gender'], input_data['Age'],
            input_data['Mobile_Money_Txns'], input_data['Airtime_Spend_USD'],
            input_data['Utility_Payments_USD'], input_data['Loan_Repayment_History']
        ]], columns=feature_cols)
        
        for col in ['Location', 'Gender', 'Loan_Repayment_History']:
            if col in st.session_state.label_encoders:
                le = st.session_state.label_encoders[col]
                X_input[col] = le.transform(X_input[col].astype(str))
        
        prediction = st.session_state.model.predict(X_input)[0]
        proba = st.session_state.model.predict_proba(X_input)[0]
        confidence = max(proba) * 100
        predicted_class = st.session_state.target_encoder.inverse_transform([prediction])[0]
        return predicted_class, confidence
    except:
        return "Unknown", 0

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
        'total': len(monthly), 'avg_score': monthly['score'].mean(),
        'approval_rate': (monthly['score'] >= 3).mean() * 100,
        'high_risk': (monthly['score'] < 3).mean() * 100
    }

# Sidebar inputs
with st.sidebar:
    st.markdown("### 🎯 Applicant Information")
    st.markdown("---")
    
    Location = st.selectbox("📍 Location", sorted(df['Location'].unique()))
    gender = st.selectbox("👤 Gender", sorted(df['Gender'].unique()))
    Age = st.slider("🎂 Age", 18, 80, 35)
    
    st.markdown("### 💰 Financial Behavior")
    
    Mobile_Money_Txns = st.slider("📱 Mobile Money Transactions", 0.0, 300.0, 75.0)
    Airtime_Spend_USD = st.slider("📞 Airtime Spend (USD)", 0.0, 300.0, 50.0)
    Utility_Payments_USD = st.slider("💡 Utility Payments (USD)", 0.0, 300.0, 80.0)
    Loan_Repayment_History = st.selectbox("📊 Loan Repayment History", ['Poor', 'Fair', 'Good', 'Excellent'])
    Income_Source = st.selectbox("💰 Source of Income", ['Informal Business', 'Farming', 'Remittances', 'Other'])

# Calculate score
score = 0
max_score = 6
if 30 <= Age <= 50:
    score += 2
elif 25 <= Age < 30 or 50 < Age <= 60:
    score += 1

if Mobile_Money_Txns > df['Mobile_Money_Txns'].median():
    score += 1

repayment_scores = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
score += repayment_scores[Loan_Repayment_History]

risk_level = get_risk_level(score)

# Get AI prediction
predicted_class, confidence = predict_credit({
    'Location': Location, 'Gender': gender, 'Age': Age,
    'Mobile_Money_Txns': Mobile_Money_Txns, 'Airtime_Spend_USD': Airtime_Spend_USD,
    'Utility_Payments_USD': Utility_Payments_USD, 'Loan_Repayment_History': Loan_Repayment_History
})

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard", "🎯 Assessments", "🔍 Analysis", "📋 Monthly Reports"
])

# ================= TAB 1: DASHBOARD =================
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

# ================= TAB 2: ASSESSMENTS =================
with tab2:
    st.markdown("### 🎯 Credit Assessment")
    
    input_data = pd.DataFrame({
        "Feature": ["📍 Location", "👤 Gender", "🎂 Age", "📱 Mobile Transactions", "📞 Airtime Spend", "💡 Utility Payments", "📊 Repayment History", "💰 Income Source"],
        "Value": [Location, gender, f"{Age}", f"{Mobile_Money_Txns:.0f}", f"${Airtime_Spend_USD:.2f}", f"${Utility_Payments_USD:.2f}", Loan_Repayment_History, Income_Source]
    })
    st.dataframe(input_data, use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("📈 Credit Score", f"{score}/{max_score}")
        st.progress(score / max_score)
    with col2:
        if score >= 5:
            st.success(f"### ✅ EXCELLENT CREDITWORTHINESS")
            st.balloons()
        elif score >= 3:
            st.warning(f"### ⚠️ MODERATE RISK PROFILE")
        else:
            st.error(f"### ❌ HIGHER RISK PROFILE")
        st.write(f"**Risk Level:** {risk_level}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Assessment", type="primary", use_container_width=True):
            assessment_data = {
                'location': Location, 'gender': gender, 'age': Age,
                'mobile_money_txns': Mobile_Money_Txns, 'airtime_spend': Airtime_Spend_USD,
                'utility_payments': Utility_Payments_USD, 'repayment_history': Loan_Repayment_History,
                'income_source': Income_Source, 'score': score, 'max_score': max_score,
                'risk_level': risk_level, 'predicted_class': predicted_class, 'confidence': confidence
            }
            assessment_id = save_assessment(assessment_data)
            st.session_state.assessment_results.update({'score': score, 'risk_level': risk_level, 'assessment_id': assessment_id})
            st.success(f"✅ Assessment saved! ID: {assessment_id}")
            st.rerun()
    
    st.markdown("### 📝 Actionable Recommendations")
    
    recs = []
    
    if Loan_Repayment_History == "Poor":
        recs.append("❌ **Critical:** Applicant has a history of poor loan repayments. Flag for elevated scrutiny.")
    elif Loan_Repayment_History == "Excellent":
        recs.append("✅ **Strength:** Excellent historical repayment behavior indicates high reliability.")
        
    median_txns = df['Mobile_Money_Txns'].median()
    if Mobile_Money_Txns < (median_txns * 0.5):
        recs.append("⚠️ **Digital Footprint:** Insufficient mobile money transaction volume. Gather alternative income proofs.")
    elif Mobile_Money_Txns > (median_txns * 1.5):
        recs.append("✅ **Digital Footprint:** High transaction density suggests healthy, verifiable alternative income.")
        
    if Utility_Payments_USD == 0:
        recs.append("⚠️ **Verifications:** No utility payments detected on record. Require manual KYC or proof of residence.")
        
    if score >= 5:
        recs.append("🎯 **Final Outcome:** Approve. Applicant is a premium candidate eligible for high credit tiering ($5,000 - $10,000) at prime rates.")
        st.success("\n\n".join(recs))
    elif score >= 3:
        recs.append("🎯 **Final Outcome:** Conditional Approval. Applicant meets baseline criteria. Cap initial limits at $500 - $2,000.")
        st.warning("\n\n".join(recs))
    else:
        recs.append("🎯 **Final Outcome:** Decline. Applicant exhibits severe risk attributes below organizational thresholds.")
        st.error("\n\n".join(recs))

# ================= TAB 3: ANALYSIS =================
with tab3:
    st.markdown("### 🔍 Data Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Age Distribution")
        fig = px.histogram(df, x='Age', nbins=20, title='Age Distribution', color_discrete_sequence=['#3498db'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### Gender Distribution")
        gender_counts = df['Gender'].value_counts()
        fig = go.Figure(data=[go.Pie(labels=gender_counts.index, values=gender_counts.values, hole=0.3, marker_colors=['#3498db', '#e74c3c'])])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### Credit Score by Location")
    location_scores = df.groupby('Location')['Credit_Score'].mean().sort_values(ascending=True)
    colors_loc = ['#e74c3c' if x < 3 else '#f39c12' if x < 4 else '#2ecc71' for x in location_scores.values]
    fig = go.Figure(data=[go.Bar(x=location_scores.values, y=location_scores.index, orientation='h', marker_color=colors_loc)])
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# ================= TAB 4: MONTHLY REPORTS =================
with tab4:
    st.markdown("### 📋 Monthly Assessment Reports")
    st.markdown("Overview of all credit scoring assessments performed in the last 30 days.")
    
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
                <div class="metric-value">{stats['avg_score']:.1f}/6</div>
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
                fig_trend.add_hline(y=3, line_dash="dash", line_color="#e74c3c", annotation_text="Approval Threshold")
                fig_trend.update_layout(
                    height=350, margin=dict(l=0, r=0, t=30, b=0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(range=[0, 6.5])
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
            st.markdown("#### �️ Assessment Register")
            
            display_cols = ['assessment_id', 'date', 'location', 'gender', 'score', 'risk_level']
            st.dataframe(
                df_history[display_cols].sort_values('date', ascending=False), 
                use_container_width=True, hide_index=True
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
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

# Footer
st.markdown("---")
st.markdown("### 💡 About Zim Smart Credit")
st.markdown("Leveraging alternative data (mobile money, utility payments, airtime usage) to provide fair and inclusive credit scoring for Zimbabweans.")
