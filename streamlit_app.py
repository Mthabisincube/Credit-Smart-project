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
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    [data-testid="stSidebar"] label {
        color: #ecf0f1 !important;
        font-weight: 500;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {
        color: #bdc3c7 !important;
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
                       'Airtime_Spend_ZWL', 'Utility_Payments_ZWL', 'Loan_Repayment_History']
        
        X = df[feature_cols].copy()
        y = df['Credit_Score'].copy()
        
        label_encoders = {}
        categorical_cols = ['Location', 'Gender', 'Loan_Repayment_History']
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        
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
        st.success("✅ Model ready!")

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
            input_data['Mobile_Money_Txns'], input_data['Airtime_Spend_ZWL'],
            input_data['Utility_Payments_ZWL'], input_data['Loan_Repayment_History']
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
    Airtime_Spend_ZWL = st.slider("📞 Airtime Spend (ZWL)", 0.0, 300.0, 50.0)
    Utility_Payments_ZWL = st.slider("💡 Utility Payments (ZWL)", 0.0, 300.0, 80.0)
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
    'Mobile_Money_Txns': Mobile_Money_Txns, 'Airtime_Spend_ZWL': Airtime_Spend_ZWL,
    'Utility_Payments_ZWL': Utility_Payments_ZWL, 'Loan_Repayment_History': Loan_Repayment_History
})

# Main Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", "🎯 Assessments", "🔍 Analysis", "📋 Monthly Reports", "🗺️ Portfolio Risk Map"
])

# ================= TAB 1: DASHBOARD =================
with tab1:
    st.markdown('<div class="glass-panel" style="margin-bottom: 1.5rem;">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("### 👋 Welcome to Zim Smart Credit")
        st.markdown("Zimbabwe's first AI-powered credit scoring platform leveraging **alternative data** for financial inclusion.")
    with col2:
        current_time = datetime.now().strftime("%I:%M %p")
        st.metric("🕐 System Time", current_time, datetime.now().strftime("%B %d"))
    with col3:
        if st.session_state.model_trained:
            st.markdown('<span class="badge badge-success">✅ AI Model Active</span>', unsafe_allow_html=True)
            st.caption(f"🎯 Accuracy: {st.session_state.model_metrics['accuracy']:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 📊 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #3498db;">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">📊 Total Records</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #9b59b6;">
            <div class="metric-value">{df['Credit_Score'].nunique()}</div>
            <div class="metric-label">🎯 Credit Classes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #2ecc71;">
            <div class="metric-value">{len(st.session_state.assessments_history)}</div>
            <div class="metric-label">📈 Assessments (30d)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        approval_rate = (df['Credit_Score'] >= 3).mean() * 100
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #e67e22;">
            <div class="metric-value">{approval_rate:.0f}%</div>
            <div class="metric-label">✅ Approval Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.model_trained:
        st.markdown('<div class="section-header">🤖 AI Model Performance</div>', unsafe_allow_html=True)
        metrics = st.session_state.model_metrics
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="stat-card"><div style="font-size: 2rem; font-weight: 800; color: #3498db;">{metrics['accuracy']:.1f}%</div><div>Accuracy</div></div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card"><div style="font-size: 2rem; font-weight: 800; color: #9b59b6;">{metrics['precision']:.1f}%</div><div>Precision</div></div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="stat-card"><div style="font-size: 2rem; font-weight: 800; color: #2ecc71;">{metrics['recall']:.1f}%</div><div>Recall</div></div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="stat-card"><div style="font-size: 2rem; font-weight: 800; color: #e67e22;">{metrics['f1_score']:.1f}%</div><div>F1 Score</div></div>
            """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Credit Score Distribution")
        score_counts = df['Credit_Score'].value_counts().sort_index()
        colors = ['#e74c3c' if x <= 2 else '#f39c12' if x <= 3 else '#2ecc71' for x in score_counts.index]
        fig = go.Figure(data=[go.Bar(x=score_counts.index, y=score_counts.values, marker_color=colors)])
        fig.update_layout(height=350, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🌍 Top Locations")
        location_counts = df['Location'].value_counts().head(6)
        fig = go.Figure(data=[go.Bar(x=location_counts.values, y=location_counts.index, orientation='h', marker_color='#3498db')])
        fig.update_layout(height=350, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

# ================= TAB 2: ASSESSMENTS =================
with tab2:
    st.markdown("### 🎯 Credit Assessment")
    
    input_data = pd.DataFrame({
        "Feature": ["📍 Location", "👤 Gender", "🎂 Age", "📱 Mobile Transactions", "📞 Airtime Spend", "💡 Utility Payments", "📊 Repayment History", "💰 Income Source"],
        "Value": [Location, gender, f"{Age}", f"{Mobile_Money_Txns:.0f}", f"{Airtime_Spend_ZWL:.0f} ZWL", f"{Utility_Payments_ZWL:.0f} ZWL", Loan_Repayment_History, Income_Source]
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
    
    if st.session_state.model_trained and predicted_class != "Unknown":
        st.markdown("### 🤖 AI Credit Prediction")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("AI Predicted Class", predicted_class)
        with col2:
            st.metric("Confidence Level", f"{confidence:.1f}%")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Assessment", type="primary", use_container_width=True):
            assessment_data = {
                'location': Location, 'gender': gender, 'age': Age,
                'mobile_money_txns': Mobile_Money_Txns, 'airtime_spend': Airtime_Spend_ZWL,
                'utility_payments': Utility_Payments_ZWL, 'repayment_history': Loan_Repayment_History,
                'income_source': Income_Source, 'score': score, 'max_score': max_score,
                'risk_level': risk_level, 'predicted_class': predicted_class, 'confidence': confidence
            }
            assessment_id = save_assessment(assessment_data)
            st.session_state.assessment_results.update({'score': score, 'risk_level': risk_level, 'assessment_id': assessment_id})
            st.success(f"✅ Assessment saved! ID: {assessment_id}")
            st.rerun()
    
    st.markdown("### 📝 Recommendations")
    if score >= 5:
        st.success("✅ Strong candidate for credit approval\n✅ Eligible for higher credit limits (up to ZWL 50,000)\n✅ Favorable interest rates (12-15% p.a.)")
    elif score >= 3:
        st.warning("⚠️ Standard credit verification required\n⚠️ Moderate credit limits (ZWL 10,000-25,000)\n⚠️ Standard interest rates (18-22% p.a.)")
    else:
        st.error("❌ Enhanced verification required\n❌ Collateral might be necessary\n❌ Lower credit limits (up to ZWL 5,000)")

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
    st.markdown("### 📋 Monthly Reports")
    stats = get_monthly_stats()
    
    if stats:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Assessments", stats['total'])
        with col2:
            st.metric("📈 Average Score", f"{stats['avg_score']:.2f}/6")
        with col3:
            st.metric("✅ Approval Rate", f"{stats['approval_rate']:.1f}%")
        
        if len(st.session_state.assessments_history) > 0:
            df_history = pd.DataFrame(st.session_state.assessments_history[-10:])
            df_history['date'] = pd.to_datetime(df_history['timestamp']).dt.date
            daily_scores = df_history.groupby('date')['score'].mean().reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_scores['date'], y=daily_scores['score'], mode='lines+markers', line=dict(color='#2ecc71', width=3)))
            fig.add_hline(y=3, line_dash="dash", line_color="#e74c3c", annotation_text="Approval Threshold")
            fig.update_layout(title='Score Trend (Last 10 Assessments)', height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📭 No assessments recorded in the last 30 days.")

# ================= TAB 5: PORTFOLIO RISK MAP - GEOJSON CHOROPLETH =================
with tab5:
    st.markdown("### 🗺️ Zimbabwe Credit Risk Map")
    st.markdown("Geographic distribution of credit scores across Zimbabwean provinces")
    
    # Try to load GeoJSON for Zimbabwe
    try:
        # Fetch Zimbabwe GeoJSON from a reliable source
        geojson_url = "https://raw.githubusercontent.com/geoiq/zimbabwe-geojson/master/zimbabwe.geojson"
        response = requests.get(geojson_url)
        
        if response.status_code == 200:
            geojson_data = response.json()
            
            # Create choropleth map
            fig_choropleth = go.Figure(go.Choroplethmapbox(
                geojson=geojson_data,
                locations=province_metrics['Province'],
                z=province_metrics['avg_score'],
                featureidkey="properties.name",
                colorscale=[
                    [0, '#e74c3c'],
                    [0.33, '#f39c12'],
                    [0.66, '#3498db'],
                    [1, '#2ecc71']
                ],
                marker_opacity=0.8,
                marker_line_width=1,
                marker_line_color='white',
                colorbar=dict(
                    title="Avg Credit Score",
                    titleside="right",
                    titlefont=dict(size=12, color='#2c3e50'),
                    tickfont=dict(color='#2c3e50')
                ),
                text=province_metrics['count'],
                hovertemplate='<b>%{location}</b><br>' +
                              'Average Score: %{z:.2f}/6<br>' +
                              'Applicants: %{text}<br>' +
                              '<extra></extra>'
            ))
            
            fig_choropleth.update_layout(
                mapbox_style="carto-positron",
                mapbox_zoom=5.5,
                mapbox_center={"lat": -19.0154, "lon": 29.1549},
                height=600,
                margin={"r": 0, "t": 40, "l": 0, "b": 0},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_choropleth, use_container_width=True)
        else:
            st.warning("GeoJSON data unavailable. Using alternative visualization.")
            # Fallback to scatter map
            fig_map = go.Figure()
            
            province_coords = {
                'Harare': {'lat': -17.8252, 'lon': 31.0335},
                'Bulawayo': {'lat': -20.1325, 'lon': 28.6265},
                'Manicaland': {'lat': -18.9216, 'lon': 32.1746},
                'Mashonaland Central': {'lat': -16.7740, 'lon': 30.9882},
                'Mashonaland East': {'lat': -17.5900, 'lon': 31.3150},
                'Mashonaland West': {'lat': -16.6500, 'lon': 29.5000},
                'Masvingo': {'lat': -20.0625, 'lon': 30.8325},
                'Matabeleland North': {'lat': -18.6400, 'lon': 27.0000},
                'Matabeleland South': {'lat': -21.0000, 'lon': 29.0000},
                'Midlands': {'lat': -19.0000, 'lon': 29.5000}
            }
            
            province_metrics['lat'] = province_metrics['Province'].apply(lambda x: province_coords.get(x, {'lat': 0})['lat'])
            province_metrics['lon'] = province_metrics['Province'].apply(lambda x: province_coords.get(x, {'lon': 0})['lon'])
            
            fig_map.add_trace(go.Scattermapbox(
                lat=province_metrics['lat'],
                lon=province_metrics['lon'],
                mode='markers+text',
                marker=dict(
                    size=province_metrics['count'] / 30,
                    color=province_metrics['avg_score'],
                    colorscale='RdYlGn',
                    colorbar=dict(title="Avg Credit Score"),
                    showscale=True,
                    sizemin=10
                ),
                text=province_metrics['Province'],
                textposition="top center",
                textfont=dict(size=11, color='#2c3e50'),
                hovertemplate='<b>%{text}</b><br>Score: %{marker.color:.2f}<extra></extra>'
            ))
            
            fig_map.update_layout(
                mapbox_style="carto-positron",
                mapbox_center=dict(lat=-19.0154, lon=29.1549),
                mapbox_zoom=5.5,
                height=600,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig_map, use_container_width=True)
            
    except Exception as e:
        st.warning(f"Map visualization using alternative method")
    
    # Province-level detailed metrics
    st.markdown("### 📊 Province-Level Risk Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        province_sorted = province_metrics.sort_values('avg_score', ascending=True)
        colors_prov = ['#e74c3c' if x < 3 else '#f39c12' if x < 4 else '#2ecc71' for x in province_sorted['avg_score']]
        
        fig_bar = go.Figure(data=[go.Bar(
            x=province_sorted['avg_score'], y=province_sorted['Province'], orientation='h',
            marker_color=colors_prov, text=province_sorted['avg_score'].round(2), textposition='outside'
        )])
        fig_bar.update_layout(title='Average Credit Score by Province', height=400,
                             xaxis_title='Average Credit Score (0-6)', xaxis=dict(range=[0, 6]),
                             plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        high_risk_sorted = province_metrics.sort_values('high_risk_pct', ascending=False)
        fig_risk = go.Figure(data=[go.Bar(
            x=high_risk_sorted['high_risk_pct'], y=high_risk_sorted['Province'], orientation='h',
            marker_color='#e74c3c', text=high_risk_sorted['high_risk_pct'].round(1), textposition='outside'
        )])
        fig_risk.update_layout(title='Percentage of High-Risk Applicants by Province', height=400,
                              xaxis_title='High Risk Applicants (%)', xaxis=dict(range=[0, 100]),
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_risk, use_container_width=True)
    
    # Detailed data table
    st.markdown("#### 📋 Detailed Province Statistics")
    display_df = province_metrics.copy()
    display_df.columns = ['Province', 'Avg Score', 'Total Applicants', 'High Risk %']
    display_df = display_df.sort_values('Avg Score', ascending=False)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # High risk alerts
    st.markdown("#### ⚠️ High Risk Alerts")
    high_risk_locations = df[df['Credit_Score'] <= 2]['Location'].value_counts().head(5)
    if len(high_risk_locations) > 0:
        for loc, count in high_risk_locations.items():
            st.warning(f"📍 **{loc}**: {count} high-risk applicants identified")
    else:
        st.success("✅ No high-risk concentrations detected")
    
    # Risk summary
    st.markdown("#### 📈 Risk Summary Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        safest = province_metrics.loc[province_metrics['avg_score'].idxmax(), 'Province']
        st.metric("🏆 Safest Province", safest, f"Score: {province_metrics['avg_score'].max():.2f}")
    with col2:
        riskiest = province_metrics.loc[province_metrics['avg_score'].idxmin(), 'Province']
        st.metric("⚠️ Riskiest Province", riskiest, f"Score: {province_metrics['avg_score'].min():.2f}")
    with col3:
        st.metric("📊 National Average", f"{df['Credit_Score'].mean():.2f}/6")

# Footer
st.markdown("---")
st.markdown("### 💡 About Zim Smart Credit")
st.markdown("Leveraging alternative data (mobile money, utility payments, airtime usage) to provide fair and inclusive credit scoring for Zimbabweans.")
