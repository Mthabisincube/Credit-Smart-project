import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Page configuration with FNB theme
st.set_page_config(
    page_title="FNB Smart Credit - Corporate Banking",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for FNB banking theme
st.markdown("""
<style>
    .stApp {
        background: #ffffff;
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    
    /* FNB Header */
    .fnb-header {
        background: linear-gradient(90deg, #000000 0%, #8B0000 100%);
        padding: 1.5rem 2rem;
        border-bottom: 4px solid #FF0000;
        margin-bottom: 1rem;
    }
    
    .main-header {
        font-size: 2.2rem;
        color: #ffffff;
        text-align: center;
        font-weight: 700;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    .sub-header {
        font-size: 1rem;
        color: #FFD700;
        text-align: center;
        font-weight: 300;
        margin: 0;
        letter-spacing: 1px;
    }
    
    /* FNB Navigation Bar */
    .fnb-nav {
        background: #8B0000;
        padding: 1rem 2rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Main Content Container */
    .main-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        color: #333333;
        border: 1px solid #ddd;
    }
    
    /* FNB Assessment Widget */
    .fnb-assessment {
        background: linear-gradient(135deg, #8B0000 0%, #000000 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #FF0000;
        box-shadow: 0 4px 8px rgba(139, 0, 0, 0.3);
    }
    
    /* FNB Metric Cards */
    .fnb-metric {
        background: linear-gradient(135deg, #8B0000 0%, #A52A2A 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        border: 2px solid #FF0000;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
    }
    
    /* Form Styling */
    .fnb-form {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #8B0000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Result Cards */
    .fnb-excellent {
        background: linear-gradient(135deg, #006400 0%, #228B22 100%);
        color: white;
        padding: 2rem;
        border-radius: 8px;
        border-left: 5px solid #FFD700;
    }
    
    .fnb-moderate {
        background: linear-gradient(135deg, #B8860B 0%, #DAA520 100%);
        color: white;
        padding: 2rem;
        border-radius: 8px;
        border-left: 5px solid #FF8C00;
    }
    
    .fnb-poor {
        background: linear-gradient(135deg, #8B0000 0%, #B22222 100%);
        color: white;
        padding: 2rem;
        border-radius: 8px;
        border-left: 5px solid #FF0000;
    }
    
    /* FNB Footer */
    .fnb-footer {
        background: linear-gradient(90deg, #000000 0%, #8B0000 100%);
        padding: 2rem;
        border-radius: 8px;
        margin-top: 3rem;
        text-align: center;
        border-top: 3px solid #FF0000;
    }
    
    /* FNB Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #8B0000 0%, #FF0000 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(139, 0, 0, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(139, 0, 0, 0.4);
        background: linear-gradient(135deg, #FF0000 0%, #8B0000 100%);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #8B0000 0%, #FF0000 100%);
    }
    
    /* Slider Styling */
    .stSlider {
        color: #8B0000;
    }
    
    /* Selectbox Styling */
    .stSelectbox {
        color: #8B0000;
    }
    
    /* Navigation Links */
    .nav-link {
        color: #ffffff !important;
        text-decoration: none;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    
    .nav-link:hover {
        background: rgba(255, 255, 255, 0.1);
        color: #FFD700 !important;
    }
    
    /* FNB Gold Accent */
    .gold-text {
        color: #FFD700;
        font-weight: 600;
    }
    
    /* Corporate Metrics */
    .corporate-stats {
        background: #ffffff;
        border: 2px solid #8B0000;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data with caching
@st.cache_data
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/Mthabisincube/Credit-Smart-project/refs/heads/master/smart_credit_scoring_zimbabwe.csv")

df = load_data()

# FNB Corporate Header
st.markdown("""
<div class="fnb-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div style="flex: 1;">
            <h1 class="main-header">🏦 FNB Smart Credit</h1>
            <p class="sub-header">Corporate & Business Banking Solutions</p>
        </div>
        <div style="text-align: right;">
            <p style="margin: 0; color: #FFD700; font-weight: 600; font-size: 1.1rem;">Zimbabwe 🇿🇼</p>
            <p style="margin: 0; font-size: 0.9rem; color: #ffffff;">Digital Banking Excellence</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# FNB Navigation Bar
st.markdown("""
<div class="fnb-nav">
    <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
        <a href="#assessment" class="nav-link" style="background: rgba(255,255,255,0.1);">Credit Assessment</a>
        <a href="#analytics" class="nav-link">Business Analytics</a>
        <a href="#loans" class="nav-link">Loan Products</a>
        <a href="#reports" class="nav-link">Financial Reports</a>
        <a href="#support" class="nav-link">Client Support</a>
        <a href="#locations" class="nav-link">Branch Locator</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Content Container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Credit Assessment as First Widget - FNB Style
st.markdown("""
<div class="fnb-assessment">
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <div style="flex: 1;">
            <h2 style="color: white; margin: 0; font-size: 1.8rem;">🔍 FNB Instant Credit Assessment</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">Complete your business credit profile assessment in minutes</p>
        </div>
        <div style="background: #FF0000; padding: 0.5rem 1rem; border-radius: 20px;">
            <span style="color: white; font-weight: 600;">BUSINESS BANKING</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Two-column layout for the assessment form
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📋 Business Information")
    
    with st.container():
        st.markdown('<div class="fnb-form">', unsafe_allow_html=True)
        col1a, col1b = st.columns(2)
        with col1a:
            Location = st.selectbox(
                "📍 Business Location", 
                sorted(df['Location'].unique()),
                key="location"
            )
        with col1b:
            gender = st.selectbox(
                "👤 Primary Contact", 
                sorted(df['Gender'].unique()),
                key="gender"
            )
        
        Age = st.slider(
            "🎂 Business Owner Age", 
            int(df['Age'].min()), 
            int(df['Age'].max()), 
            int(df['Age'].mean()),
            key="age"
        )
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### 💰 Financial Profile")
    
    with st.container():
        st.markdown('<div class="fnb-form">', unsafe_allow_html=True)
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
            "📊 Previous Loan History", 
            sorted(df['Loan_Repayment_History'].unique()),
            key="repayment"
        )
        st.markdown('</div>', unsafe_allow_html=True)

# Assessment Button - FNB Style
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    assess_button = st.button("🚀 **PROCEED WITH CREDIT ASSESSMENT**", use_container_width=True)

# Assessment Results
if assess_button:
    st.markdown("---")
    st.markdown("## 📊 FNB Credit Assessment Results")
    
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
    
    # Display FNB metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="fnb-metric">
            <h4>FNB Credit Score</h4>
            <h2>{score}/{max_score}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="fnb-metric">
            <h4>Approval Probability</h4>
            <h2>{percentage:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        risk_level = "Low" if score >= 5 else "Medium" if score >= 3 else "High"
        risk_color = "#228B22" if score >= 5 else "#DAA520" if score >= 3 else "#B22222"
        st.markdown(f"""
        <div class="fnb-metric">
            <h4>Risk Category</h4>
            <h2 style="color: {risk_color};">{risk_level}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        status = "Approved" if score >= 4 else "Under Review" if score >= 2 else "Referred"
        status_color = "#228B22" if score >= 4 else "#DAA520" if score >= 2 else "#B22222"
        st.markdown(f"""
        <div class="fnb-metric">
            <h4>Application Status</h4>
            <h2 style="color: {status_color};">{status}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Progress bar
    st.markdown("### 📈 Credit Score Progress")
    st.progress(percentage / 100)
    
    # Final FNB recommendation
    st.markdown("### 🎯 FNB Credit Decision")
    
    if score >= 5:
        st.markdown("""
        <div class="fnb-excellent">
            <h3>✅ CREDIT APPLICATION APPROVED</h3>
            <p><strong>FNB Recommendation:</strong> Full credit approval with preferential terms</p>
            <p><strong>Credit Facility:</strong> Up to ZWL 75,000 revolving credit</p>
            <p><strong>Interest Rate:</strong> Prime + 1.5% (Preferential Rate)</p>
            <p><strong>Next Steps:</strong> Visit any FNB branch for immediate document processing</p>
            <p style="margin-top: 1rem; font-style: italic;">🎉 Welcome to FNB Business Banking Premium</p>
        </div>
        """, unsafe_allow_html=True)
    elif score >= 3:
        st.markdown("""
        <div class="fnb-moderate">
            <h3>⚠️ CREDIT APPLICATION APPROVED</h3>
            <p><strong>FNB Recommendation:</strong> Standard credit approval with monitoring</p>
            <p><strong>Credit Facility:</strong> Up to ZWL 35,000 term loan</p>
            <p><strong>Interest Rate:</strong> Prime + 3.5% (Standard Rate)</p>
            <p><strong>Next Steps:</strong> Additional business documentation required</p>
            <p style="margin-top: 1rem; font-style: italic;">📞 Your relationship manager will contact you</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="fnb-poor">
            <h3>📋 APPLICATION REQUIRES REVIEW</h3>
            <p><strong>FNB Recommendation:</strong> Application referred for manual assessment</p>
            <p><strong>Credit Facility:</strong> Subject to further evaluation</p>
            <p><strong>Next Steps:</strong> Contact FNB Business Banking for personalized assistance</p>
            <p style="margin-top: 1rem; font-style: italic;">💼 Consider our Business Starter packages</p>
        </div>
        """, unsafe_allow_html=True)

# FNB Business Analytics Section
st.markdown("---")
st.markdown("## 📈 FNB Business Analytics")

# Corporate metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="corporate-stats">
        <h4 style="color: #8B0000; margin: 0;">Total Applications</h4>
        <h2 style="color: #8B0000; margin: 0;">{len(df):,}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    good_ratio = (df['Credit_Score'] == 'Good').mean()
    st.markdown(f"""
    <div class="corporate-stats">
        <h4 style="color: #8B0000; margin: 0;">Approval Rate</h4>
        <h2 style="color: #8B0000; margin: 0;">{good_ratio:.1%}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="corporate-stats">
        <h4 style="color: #8B0000; margin: 0;">Active Business Clients</h4>
        <h2 style="color: #8B0000; margin: 0;">{(len(df) * 0.72):.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="corporate-stats">
        <h4 style="color: #8B0000; margin: 0;">Portfolio Value</h4>
        <h2 style="color: #8B0000; margin: 0;">ZWL {(len(df) * 25000):,}</h2>
    </div>
    """, unsafe_allow_html=True)

# Charts and Data
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Credit Distribution")
    score_counts = df['Credit_Score'].value_counts().sort_index()
    st.bar_chart(score_counts)

with col2:
    st.markdown("#### Regional Performance")
    location_counts = df['Location'].value_counts()
    st.dataframe(location_counts, use_container_width=True)

# Close main container
st.markdown('</div>', unsafe_allow_html=True)

# FNB Corporate Footer
st.markdown("""
<div class="fnb-footer">
    <div style="display: flex; justify-content: space-around; align-items: start; flex-wrap: wrap; text-align: left;">
        <div style="margin: 0 1rem 1rem 0;">
            <h4 style="color: #FFD700; margin-bottom: 0.5rem;">FNB Business Banking</h4>
            <p style="margin: 0; font-size: 0.8rem; color: #ffffff;">Corporate & Investment Division</p>
            <p style="margin: 0; font-size: 0.8rem; color: #ffffff;">First National Bank Zimbabwe</p>
        </div>
        <div style="margin: 0 1rem 1rem 0;">
            <p style="margin: 0; font-size: 0.8rem; color: #ffffff;">📍 FNB House, Harare</p>
            <p style="margin: 0; font-size: 0.8rem; color: #ffffff;">📞 +263 24 275 0000</p>
            <p style="margin: 0; font-size: 0.8rem; color: #ffffff;">📱 FNB App Available</p>
        </div>
        <div style="margin: 0 1rem 1rem 0;">
            <p style="margin: 0; font-size: 0.8rem; color: #ffffff;">✉️ businessbanking@fnb.co.zw</p>
            <p style="margin: 0; font-size: 0.8rem; color: #ffffff;">🕒 Mon-Fri: 8AM-4:30PM</p>
            <p style="margin: 0; font-size: 0.8rem; color: #ffffff;">Sat: 8AM-12PM</p>
        </div>
    </div>
    <hr style="border-color: #FF0000; margin: 1rem 0;">
    <p style="margin: 0; font-size: 0.7rem; color: rgba(255,255,255,0.7);">
        © 2024 First National Bank Zimbabwe. A subsidiary of FirstRand Limited. | Registered Commercial Bank
    </p>
    <p style="margin: 0; font-size: 0.7rem; color: #FFD700;">
        How can we help you? | Call 0800 1100 | Use FNB App | Visit fnb.co.zw
    </p>
</div>
""", unsafe_allow_html=True)
