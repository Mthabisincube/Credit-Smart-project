import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Page configuration with corporate theme
st.set_page_config(
    page_title="ZimSmart Credit - Corporate Banking",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for corporate banking theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0c2c4a 0%, #1a4b6d 50%, #0c2c4a 100%);
        color: #ffffff;
    }
    
    /* Corporate Header */
    .corporate-header {
        background: linear-gradient(90deg, #0c2c4a 0%, #1a5276 50%, #0c2c4a 100%);
        padding: 1rem 2rem;
        border-bottom: 3px solid #f39c12;
        margin-bottom: 2rem;
    }
    
    .main-header {
        font-size: 2.5rem;
        color: #ffffff;
        text-align: center;
        font-weight: 700;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .sub-header {
        font-size: 1rem;
        color: #f39c12;
        text-align: center;
        font-weight: 300;
        margin: 0;
        letter-spacing: 1px;
    }
    
    /* Navigation Bar */
    .nav-bar {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Main Content Container */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        color: #2c3e50;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Credit Assessment Widget */
    .assessment-widget {
        background: linear-gradient(135deg, #1a5276 0%, #2e86c1 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #f39c12;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #2e86c1 0%, #1a5276 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #3498db;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Form Styling */
    .form-section {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        backdrop-filter: blur(5px);
    }
    
    /* Result Cards */
    .result-excellent {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #f39c12;
    }
    
    .result-moderate {
        background: linear-gradient(135deg, #f39c12 0%, #f1c40f 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #e67e22;
    }
    
    .result-poor {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #f39c12;
    }
    
    /* Corporate Footer */
    .corporate-footer {
        background: rgba(0, 0, 0, 0.3);
        padding: 2rem;
        border-radius: 10px;
        margin-top: 3rem;
        text-align: center;
        border-top: 2px solid #f39c12;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(243, 156, 18, 0.4);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
    }
    
    /* Slider Styling */
    .stSlider {
        color: #2c3e50;
    }
    
    /* Selectbox Styling */
    .stSelectbox {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Load data with caching
@st.cache_data
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/Mthabisincube/Credit-Smart-project/refs/heads/master/smart_credit_scoring_zimbabwe.csv")

df = load_data()

# Corporate Header
st.markdown("""
<div class="corporate-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 class="main-header">🏦 ZimSmart Credit</h1>
            <p class="sub-header">Corporate Banking & Investment Solutions</p>
        </div>
        <div style="text-align: right;">
            <p style="margin: 0; color: #f39c12; font-weight: 600;">Zimbabwe 🇿🇼</p>
            <p style="margin: 0; font-size: 0.9rem;">Digital Banking Platform</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation Bar
st.markdown("""
<div class="nav-bar">
    <div style="display: flex; justify-content: space-around; align-items: center;">
        <a href="#assessment" style="color: #f39c12; text-decoration: none; font-weight: 600;">Credit Assessment</a>
        <a href="#analytics" style="color: #ffffff; text-decoration: none; font-weight: 600;">Portfolio Analytics</a>
        <a href="#reports" style="color: #ffffff; text-decoration: none; font-weight: 600;">Financial Reports</a>
        <a href="#about" style="color: #ffffff; text-decoration: none; font-weight: 600;">About Us</a>
        <a href="#contact" style="color: #ffffff; text-decoration: none; font-weight: 600;">Contact Us</a>
        <a href="#locate" style="color: #ffffff; text-decoration: none; font-weight: 600;">Locate Us</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Content Container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Credit Assessment as First Widget
st.markdown("""
<div class="assessment-widget">
    <h2 style="color: white; margin-bottom: 1rem;">🔍 Instant Credit Assessment</h2>
    <p style="color: rgba(255,255,255,0.9);">Complete the form below for an immediate creditworthiness evaluation using our advanced scoring algorithm.</p>
</div>
""", unsafe_allow_html=True)

# Two-column layout for the assessment form
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📋 Client Information")
    
    with st.container():
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        col1a, col1b = st.columns(2)
        with col1a:
            Location = st.selectbox(
                "📍 Business Location", 
                sorted(df['Location'].unique()),
                key="location"
            )
        with col1b:
            gender = st.selectbox(
                "👤 Client Gender", 
                sorted(df['Gender'].unique()),
                key="gender"
            )
        
        Age = st.slider(
            "🎂 Client Age", 
            int(df['Age'].min()), 
            int(df['Age'].max()), 
            int(df['Age'].mean()),
            key="age"
        )
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### 💰 Financial Profile")
    
    with st.container():
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        Mobile_Money_Txns = st.slider(
            "📱 Monthly Mobile Transactions", 
            float(df['Mobile_Money_Txns'].min()), 
            float(df['Mobile_Money_Txns'].max()), 
            float(df['Mobile_Money_Txns'].mean()),
            key="mobile"
        )
        
        Airtime_Spend_ZWL = st.slider(
            "📞 Monthly Airtime Spend (ZWL)", 
            float(df['Airtime_Spend_ZWL'].min()), 
            float(df['Airtime_Spend_ZWL'].max()), 
            float(df['Airtime_Spend_ZWL'].mean()),
            key="airtime"
        )
        
        Utility_Payments_ZWL = st.slider(
            "💡 Monthly Utility Payments (ZWL)", 
            float(df['Utility_Payments_ZWL'].min()), 
            float(df['Utility_Payments_ZWL'].max()), 
            float(df['Utility_Payments_ZWL'].mean()),
            key="utility"
        )
        
        Loan_Repayment_History = st.selectbox(
            "📊 Loan Repayment History", 
            sorted(df['Loan_Repayment_History'].unique()),
            key="repayment"
        )
        st.markdown('</div>', unsafe_allow_html=True)

# Assessment Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    assess_button = st.button("🚀 **RUN CREDIT ASSESSMENT**", use_container_width=True)

# Assessment Results
if assess_button:
    st.markdown("---")
    st.markdown("## 📊 Assessment Results")
    
    # Calculate score
    score = 0
    max_score = 6
    
    # Age scoring
    if 30 <= Age <= 50:
        score += 2
    elif 25 <= Age < 30 or 50 < Age <= 60:
        score += 1
    
    # Transaction activity scoring
    mobile_median = df['Mobile_Money_Txns'].median()
    if Mobile_Money_Txns > mobile_median:
        score += 1
    
    # Repayment history scoring
    repayment_scores = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
    score += repayment_scores[Loan_Repayment_History]
    
    percentage = (score / max_score) * 100
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Credit Score</h3>
            <h1>{score}/{max_score}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Success Rate</h3>
            <h1>{percentage:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        risk_level = "Low" if score >= 5 else "Medium" if score >= 3 else "High"
        st.markdown(f"""
        <div class="metric-card">
            <h3>Risk Level</h3>
            <h1>{risk_level}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        status = "Approved" if score >= 4 else "Review" if score >= 2 else "Declined"
        st.markdown(f"""
        <div class="metric-card">
            <h3>Status</h3>
            <h1>{status}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    # Progress bar
    st.markdown("### 📈 Credit Score Progress")
    st.progress(percentage / 100)
    
    # Final recommendation
    st.markdown("### 🎯 Credit Recommendation")
    
    if score >= 5:
        st.markdown("""
        <div class="result-excellent">
            <h3>✅ EXCELLENT CREDIT PROFILE</h3>
            <p><strong>Recommendation:</strong> Credit application APPROVED with preferential terms</p>
            <p><strong>Credit Limit:</strong> Up to ZWL 50,000</p>
            <p><strong>Interest Rate:</strong> Prime + 2%</p>
            <p><strong>Next Steps:</strong> Document verification for immediate disbursement</p>
        </div>
        """, unsafe_allow_html=True)
    elif score >= 3:
        st.markdown("""
        <div class="result-moderate">
            <h3>⚠️ STANDARD CREDIT PROFILE</h3>
            <p><strong>Recommendation:</strong> Credit application APPROVED with standard terms</p>
            <p><strong>Credit Limit:</strong> Up to ZWL 25,000</p>
            <p><strong>Interest Rate:</strong> Prime + 4%</p>
            <p><strong>Next Steps:</strong> Additional documentation and collateral assessment required</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-poor">
            <h3>❌ CREDIT APPLICATION UNDER REVIEW</h3>
            <p><strong>Recommendation:</strong> Application requires additional review</p>
            <p><strong>Credit Limit:</strong> Subject to further assessment</p>
            <p><strong>Interest Rate:</strong> To be determined after full review</p>
            <p><strong>Next Steps:</strong> Contact our credit department for personalized assessment</p>
        </div>
        """, unsafe_allow_html=True)

# Portfolio Analytics Section
st.markdown("---")
st.markdown("## 📈 Portfolio Analytics Dashboard")

# Metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Total Applications</h4>
        <h2>{len(df):,}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    good_ratio = (df['Credit_Score'] == 'Good').mean()
    st.markdown(f"""
    <div class="metric-card">
        <h4>Approval Rate</h4>
        <h2>{good_ratio:.1%}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Active Clients</h4>
        <h2>{(len(df) * 0.65):.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Portfolio Value</h4>
        <h2>ZWL {(len(df) * 15000):,}</h2>
    </div>
    """, unsafe_allow_html=True)

# Charts and Data
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Credit Score Distribution")
    score_counts = df['Credit_Score'].value_counts().sort_index()
    st.bar_chart(score_counts)

with col2:
    st.markdown("#### Location Performance")
    location_counts = df['Location'].value_counts()
    st.dataframe(location_counts, use_container_width=True)

# Close main container
st.markdown('</div>', unsafe_allow_html=True)

# Corporate Footer
st.markdown("""
<div class="corporate-footer">
    <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
        <div>
            <h4 style="color: #f39c12; margin-bottom: 0.5rem;">ZimSmart Credit</h4>
            <p style="margin: 0; font-size: 0.8rem;">Corporate Banking Division</p>
        </div>
        <div>
            <p style="margin: 0; font-size: 0.8rem;">📍 Harare, Zimbabwe</p>
            <p style="margin: 0; font-size: 0.8rem;">📞 +263 24 123 4567</p>
        </div>
        <div>
            <p style="margin: 0; font-size: 0.8rem;">✉️ corporate@zimscredit.co.zw</p>
            <p style="margin: 0; font-size: 0.8rem;">🕒 Mon-Fri: 8AM-5PM</p>
        </div>
    </div>
    <hr style="border-color: #f39c12; margin: 1rem 0;">
    <p style="margin: 0; font-size: 0.7rem; color: rgba(255,255,255,0.7);">
        © 2024 ZimSmart Credit. All rights reserved. | Member of Zimbabwe Banking Association
    </p>
</div>
""", unsafe_allow_html=True)
