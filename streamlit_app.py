import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests
import json

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Zim Smart Credit App",
    page_icon="💳",
    layout="wide"
)

# ================= LIGHT UI =================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #e4ecf7 50%, #d6e4f0 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.main-header {
    text-align: center;
    font-size: 2.5rem;
    font-weight: 800;
    color: #2c3e50;
    margin-bottom: 0.5rem;
    padding: 1rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sub-header {
    text-align: center;
    color: #5a6c7e;
    margin-bottom: 2rem;
}
.metric-card {
    background: #ffffff;
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid #e3e8ef;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: all 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.section-header {
    font-weight: 700;
    font-size: 1.2rem;
    margin-top: 1rem;
    margin-bottom: 1rem;
    color: #2c3e50;
    border-left: 4px solid #667eea;
    padding-left: 1rem;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
}
[data-testid="stSidebar"] label {
    color: #ecf0f1 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: #bdc3c7 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102,126,234,0.4);
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
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
}
.badge-success {
    background: #d4edda;
    color: #155724;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}
.badge-warning {
    background: #fff3cd;
    color: #856404;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}
.badge-danger {
    background: #f8d7da;
    color: #721c24;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏦 Zim Smart Credit App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">🚀 AI-Powered Credit Scoring | Alternative Data Intelligence | Financial Inclusion for Zimbabwe</div>', unsafe_allow_html=True)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/Mthabisincube/Credit-Smart-project/refs/heads/master/smart_credit_scoring_zimbabwe.csv")
    np.random.seed(42)
    income_sources = ['Formal Employment', 'Informal Business', 'Farming', 'Remittances', 'Other']
    df['Income_Source'] = np.random.choice(income_sources, size=len(df), p=[0.4, 0.25, 0.15, 0.1, 0.1])
    return df

df = load_data()

# ================= PROVINCE MAPPING =================
location_to_province = {
    'Harare': 'Harare',
    'Bulawayo': 'Bulawayo',
    'Mutare': 'Manicaland',
    'Marondera': 'Mashonaland East',
    'Chinhoyi': 'Mashonaland West',
    'Bindura': 'Mashonaland Central',
    'Masvingo': 'Masvingo',
    'Gweru': 'Midlands',
    'Kwekwe': 'Midlands',
    'Hwange': 'Matabeleland North',
    'Victoria Falls': 'Matabeleland North',
    'Gwanda': 'Matabeleland South'
}
df['Province'] = df['Location'].map(location_to_province)

# ================= SESSION STATE =================
if 'history' not in st.session_state:
    st.session_state.history = []
if 'assessment_results' not in st.session_state:
    st.session_state.assessment_results = None

# ================= TRAIN MODEL =================
@st.cache_resource
def train_model(df):
    X = df[['Location','Gender','Age','Mobile_Money_Txns',
            'Airtime_Spend_ZWL','Utility_Payments_ZWL','Loan_Repayment_History']].copy()
    y = df['Credit_Score']

    encoders = {}
    for col in ['Location','Gender','Loan_Repayment_History']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    model = RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42)
    model.fit(X, y)
    
    # Calculate accuracy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100

    return model, encoders, accuracy

model, encoders, model_accuracy = train_model(df)

# ================= SIDEBAR =================
st.sidebar.header("📝 Applicant Information")
st.sidebar.markdown("---")

Location = st.sidebar.selectbox("📍 Location", sorted(df['Location'].unique()))
Gender = st.sidebar.selectbox("👤 Gender", sorted(df['Gender'].unique()))
Age = st.sidebar.slider("🎂 Age", 18, 80, 35)

st.sidebar.markdown("### 💰 Financial Behavior")
st.sidebar.markdown("---")

Mobile = st.sidebar.slider("📱 Mobile Money Txns", 0.0, 300.0, 70.0)
Airtime = st.sidebar.slider("📞 Airtime Spend (ZWL)", 0.0, 300.0, 50.0)
Utility = st.sidebar.slider("💡 Utility Payments (ZWL)", 0.0, 300.0, 80.0)
Repayment = st.sidebar.selectbox("📊 Repayment History", ['Poor','Fair','Good','Excellent'])
Income_Source = st.sidebar.selectbox("💰 Income Source", ['Informal Business', 'Farming', 'Remittances', 'Other'])

# ================= SCORING =================
score = 0
max_score = 6
if 30 <= Age <= 50:
    score += 2
elif 25 <= Age < 30 or 50 < Age <= 60:
    score += 1

if Mobile > df['Mobile_Money_Txns'].median():
    score += 1

rep_map = {'Poor':0,'Fair':1,'Good':2,'Excellent':3}
score += rep_map[Repayment]

risk_level = "Low" if score >= 5 else ("Medium" if score >= 3 else "High")

# ================= PREDICTION =================
def predict():
    data = pd.DataFrame([[Location, Gender, Age, Mobile, Airtime, Utility, Repayment]],
        columns=['Location','Gender','Age','Mobile_Money_Txns',
                 'Airtime_Spend_ZWL','Utility_Payments_ZWL','Loan_Repayment_History'])

    for col in encoders:
        data[col] = encoders[col].transform(data[col])

    pred = model.predict(data)[0]
    prob = model.predict_proba(data).max() * 100
    return pred, prob

prediction, confidence = predict()

# ================= TABS =================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🎯 Assessment", "📈 Analysis", "🗺️ Risk Map"])

# ================= DASHBOARD =================
with tab1:
    st.markdown("### 📊 Overview Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem; font-weight: 800; color: #667eea;">{len(df):,}</div>
            <div style="color: #5a6c7e;">Total Records</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem; font-weight: 800; color: #2ecc71;">{df['Credit_Score'].mean():.2f}</div>
            <div style="color: #5a6c7e;">Average Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem; font-weight: 800; color: #9b59b6;">{model_accuracy:.1f}%</div>
            <div style="color: #5a6c7e;">Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem; font-weight: 800; color: #e67e22;">{len(st.session_state.history)}</div>
            <div style="color: #5a6c7e;">Assessments</div>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">📊 Credit Score Distribution</div>', unsafe_allow_html=True)
        score_counts = df['Credit_Score'].value_counts().sort_index()
        colors = ['#e74c3c' if x <= 2 else '#f39c12' if x <= 3 else '#2ecc71' for x in score_counts.index]
        fig = go.Figure(data=[go.Bar(x=score_counts.index, y=score_counts.values, marker_color=colors)])
        fig.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                         xaxis_title='Credit Score', yaxis_title='Number of Applicants')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">📍 Top Locations</div>', unsafe_allow_html=True)
        location_counts = df['Location'].value_counts().head(8)
        fig = go.Figure(data=[go.Bar(x=location_counts.values, y=location_counts.index, orientation='h', 
                                    marker_color='#3498db', text=location_counts.values, textposition='outside')])
        fig.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                         xaxis_title='Number of Applicants')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="section-header">🏆 Province Performance</div>', unsafe_allow_html=True)
    province_scores = df.groupby('Province')['Credit_Score'].mean().sort_values(ascending=True)
    colors_prov = ['#e74c3c' if x < 3 else '#f39c12' if x < 4 else '#2ecc71' for x in province_scores.values]
    fig = go.Figure(data=[go.Bar(x=province_scores.values, y=province_scores.index, orientation='h', 
                                marker_color=colors_prov, text=province_scores.values.round(2), textposition='outside')])
    fig.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                     xaxis_title='Average Credit Score', xaxis=dict(range=[0, 6]))
    st.plotly_chart(fig, use_container_width=True)

# ================= ASSESSMENT =================
with tab2:
    st.markdown("### 🎯 Credit Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 3rem; font-weight: 800; color: #667eea;">{score}/{max_score}</div>
            <div style="color: #5a6c7e;">Credit Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        risk_color = "#2ecc71" if risk_level == "Low" else ("#f39c12" if risk_level == "Medium" else "#e74c3c")
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 2rem; font-weight: 800; color: {risk_color};">{risk_level} Risk</div>
            <div style="color: #5a6c7e;">Risk Level</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.progress(score / max_score)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 1.5rem; font-weight: 700;">🤖 AI Prediction</div>
            <div style="font-size: 2rem; font-weight: 800; color: #9b59b6;">{prediction}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 1.5rem; font-weight: 700;">📊 Confidence</div>
            <div style="font-size: 2rem; font-weight: 800; color: #2ecc71;">{confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendations
    st.markdown('<div class="section-header">📝 Recommendations</div>', unsafe_allow_html=True)
    if score >= 5:
        st.success("✅ Strong candidate for credit approval\n✅ Eligible for higher credit limits (up to ZWL 50,000)\n✅ Favorable interest rates (12-15% p.a.)")
    elif score >= 3:
        st.warning("⚠️ Standard credit verification required\n⚠️ Moderate credit limits (ZWL 10,000-25,000)\n⚠️ Standard interest rates (18-22% p.a.)")
    else:
        st.error("❌ Enhanced verification required\n❌ Collateral might be necessary\n❌ Lower credit limits (up to ZWL 5,000)")
    
    # Save button
    if st.button("💾 Save Assessment", type="primary", use_container_width=True):
        st.session_state.history.append({
            "date": datetime.now(),
            "score": score,
            "risk": risk_level,
            "prediction": prediction,
            "location": Location,
            "gender": Gender,
            "age": Age
        })
        st.session_state.assessment_results = {
            'score': score, 'risk': risk_level, 'prediction': prediction, 'confidence': confidence
        }
        st.success(f"✅ Assessment saved! Total saved: {len(st.session_state.history)}")
        st.balloons()
    
    # Show recent assessments
    if st.session_state.history:
        st.markdown('<div class="section-header">📋 Recent Assessments</div>', unsafe_allow_html=True)
        recent = pd.DataFrame(st.session_state.history[-5:])
        if 'date' in recent.columns:
            recent['date'] = pd.to_datetime(recent['date']).dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(recent[['date', 'score', 'risk', 'prediction', 'location']], use_container_width=True)

# ================= ANALYSIS =================
with tab3:
    st.markdown("### 📈 Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">Gender Analysis</div>', unsafe_allow_html=True)
        fig = px.box(df, x='Gender', y='Credit_Score', color='Gender',
                    color_discrete_sequence=['#3498db', '#e74c3c'],
                    title='Credit Score Distribution by Gender')
        fig.update_layout(height=450, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">Age Analysis</div>', unsafe_allow_html=True)
        fig = px.scatter(df, x='Age', y='Credit_Score', color='Gender',
                        color_discrete_sequence=['#3498db', '#e74c3c'],
                        title='Age vs Credit Score', trendline='ols')
        fig.update_layout(height=450, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">Mobile Money Impact</div>', unsafe_allow_html=True)
        df['Mobile_Category'] = pd.cut(df['Mobile_Money_Txns'], bins=3, labels=['Low', 'Medium', 'High'])
        mobile_impact = df.groupby('Mobile_Category')['Credit_Score'].mean().reset_index()
        fig = go.Figure(data=[go.Bar(x=mobile_impact['Mobile_Category'], y=mobile_impact['Credit_Score'],
                                    marker_color=['#e74c3c', '#f39c12', '#2ecc71'])])
        fig.update_layout(title='Average Credit Score by Mobile Money Usage', height=400,
                         plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">Repayment History Impact</div>', unsafe_allow_html=True)
        repayment_impact = df.groupby('Loan_Repayment_History')['Credit_Score'].mean().reset_index()
        fig = go.Figure(data=[go.Bar(x=repayment_impact['Loan_Repayment_History'], y=repayment_impact['Credit_Score'],
                                    marker_color='#9b59b6')])
        fig.update_layout(title='Average Credit Score by Repayment History', height=400,
                         plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    numeric_cols = ['Age', 'Mobile_Money_Txns', 'Airtime_Spend_ZWL', 'Utility_Payments_ZWL', 'Credit_Score']
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect='auto', color_continuous_scale='RdBu_r')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# ================= CHOROPLETH MAP =================
with tab4:
    st.markdown("### 🗺️ Zimbabwe Credit Risk Map")
    st.markdown("Geographic distribution of credit scores across Zimbabwean provinces")
    
    # Calculate province-level metrics
    province_risk = df.groupby('Province').agg({
        'Credit_Score': ['mean', 'count', lambda x: (x < 3).mean() * 100]
    }).reset_index()
    province_risk.columns = ['Province', 'Avg_Score', 'Count', 'High_Risk_Pct']
    province_risk = province_risk.dropna()
    
    # Create choropleth map
    try:
        # Use a simpler approach with mapbox
        fig = go.Figure()
        
        # Add scatter mapbox for provinces
        province_coords = {
            'Harare': {'lat': -17.8252, 'lon': 31.0335},
            'Bulawayo': {'lat': -20.1325, 'lon': 28.6265},
            'Manicaland': {'lat': -18.9216, 'lon': 32.1746},
            'Mashonaland East': {'lat': -17.5900, 'lon': 31.3150},
            'Mashonaland West': {'lat': -16.6500, 'lon': 29.5000},
            'Mashonaland Central': {'lat': -16.7740, 'lon': 30.9882},
            'Masvingo': {'lat': -20.0625, 'lon': 30.8325},
            'Midlands': {'lat': -19.0000, 'lon': 29.5000},
            'Matabeleland North': {'lat': -18.6400, 'lon': 27.0000},
            'Matabeleland South': {'lat': -21.0000, 'lon': 29.0000}
        }
        
        province_risk['lat'] = province_risk['Province'].apply(lambda x: province_coords.get(x, {'lat': 0})['lat'])
        province_risk['lon'] = province_risk['Province'].apply(lambda x: province_coords.get(x, {'lon': 0})['lon'])
        
        # Color scale based on average score
        colors = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71']
        
        fig.add_trace(go.Scattermapbox(
            lat=province_risk['lat'],
            lon=province_risk['lon'],
            mode='markers+text',
            marker=dict(
                size=province_risk['Count'] / 50,
                color=province_risk['Avg_Score'],
                colorscale='RdYlGn',
                colorbar=dict(title="Avg Credit Score"),
                showscale=True,
                sizemin=10,
                sizemode='area'
            ),
            text=province_risk['Province'],
            textposition="top center",
            textfont=dict(size=12, color='#2c3e50'),
            hovertemplate='<b>%{text}</b><br>' +
                          'Average Score: %{marker.color:.2f}/6<br>' +
                          'Applicants: %{marker.size:.0f}<br>' +
                          'High Risk: ' + province_risk['High_Risk_Pct'].round(1).astype(str) + '%<extra></extra>'
        ))
        
        fig.update_layout(
            mapbox=dict(
                style='carto-positron',
                center=dict(lat=-19.0154, lon=29.1549),
                zoom=5.5
            ),
            height=600,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Map display using alternative method")
        # Fallback to bar chart
        fig = px.bar(province_risk, x='Province', y='Avg_Score', 
                    color='Avg_Score', color_continuous_scale='RdYlGn',
                    title='Average Credit Score by Province')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Province-level detailed metrics
    st.markdown("### 📊 Province Risk Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        province_sorted = province_risk.sort_values('Avg_Score', ascending=True)
        colors_prov = ['#e74c3c' if x < 3 else '#f39c12' if x < 4 else '#2ecc71' for x in province_sorted['Avg_Score']]
        
        fig_bar = go.Figure(data=[go.Bar(
            x=province_sorted['Avg_Score'],
            y=province_sorted['Province'],
            orientation='h',
            marker_color=colors_prov,
            text=province_sorted['Avg_Score'].round(2),
            textposition='outside'
        )])
        fig_bar.update_layout(title='Average Credit Score by Province', height=400,
                             xaxis_title='Average Credit Score (0-6)', xaxis=dict(range=[0, 6]))
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        high_risk_sorted = province_risk.sort_values('High_Risk_Pct', ascending=False)
        fig_risk = go.Figure(data=[go.Bar(
            x=high_risk_sorted['High_Risk_Pct'],
            y=high_risk_sorted['Province'],
            orientation='h',
            marker_color='#e74c3c',
            text=high_risk_sorted['High_Risk_Pct'].round(1),
            textposition='outside'
        )])
        fig_risk.update_layout(title='Percentage of High-Risk Applicants by Province', height=400,
                              xaxis_title='High Risk Applicants (%)', xaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_risk, use_container_width=True)
    
    # Detailed data table
    st.markdown("#### 📋 Detailed Province Statistics")
    display_df = province_risk.copy()
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
    st.markdown("#### 📈 Risk Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        safest = province_risk.loc[province_risk['Avg_Score'].idxmax(), 'Province']
        st.metric("🏆 Safest Province", safest, f"{province_risk['Avg_Score'].max():.2f}/6")
    with col2:
        riskiest = province_risk.loc[province_risk['Avg_Score'].idxmin(), 'Province']
        st.metric("⚠️ Riskiest Province", riskiest, f"{province_risk['Avg_Score'].min():.2f}/6")
    with col3:
        st.metric("📊 National Average", f"{df['Credit_Score'].mean():.2f}/6")

# Footer
st.markdown("---")
st.markdown("### 💡 About Zim Smart Credit")
st.markdown("Leveraging alternative data (mobile money, utility payments, airtime usage) to provide fair and inclusive credit scoring for Zimbabweans without traditional banking history.")
