import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from datetime import datetime, timedelta, date
import json
import base64
import io
import matplotlib.pyplot as plt

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
    
    .feature-box {
        background-color: rgba(232, 244, 253, 0.9);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 2px solid #b8d4f0;
        transition: all 0.3s ease;
    }
    
    .feature-box:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
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
    
    .input-card {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #e9ecef;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .tab-content {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .glowing-text {
        text-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
    }
    
    .assessment-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    /* Random Forest specific styles */
    .rf-info-box {
        background: linear-gradient(135deg, rgba(144, 238, 144, 0.95) 0%, rgba(152, 251, 152, 0.95) 100%);
        border: 2px solid #32CD32;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #006400;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .tree-diagram {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #32CD32;
        margin: 1rem 0;
        text-align: center;
        font-family: monospace;
    }
    
    .html-report {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 20px;
        margin: 20px 0;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .accuracy-high {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
    }
    
    .accuracy-medium {
        background: linear-gradient(135deg, #FFC107 0%, #FF9800 100%);
        color: white;
    }
    
    .accuracy-low {
        background: linear-gradient(135deg, #F44336 0%, #d32f2f 100%);
        color: white;
    }
    
    .system-explanation {
        background: linear-gradient(135deg, rgba(220, 237, 255, 0.95) 0%, rgba(195, 220, 255, 0.95) 100%);
        border: 2px solid #1f77b4;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #0d47a1;
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
        },
        'top_factors': {},
        'score_trend': {}
    }
    
    # Calculate score trend (daily averages)
    if 'date' in df_assessments.columns and 'final_score' in df_assessments.columns:
        df_assessments['date'] = pd.to_datetime(df_assessments['date'])
        daily_avg = df_assessments.groupby('date')['final_score'].mean().reset_index()
        summary['score_trend'] = {
            'dates': daily_avg['date'].dt.strftime('%Y-%m-%d').tolist(),
            'scores': daily_avg['final_score'].tolist()
        }
    
    return summary

# ==================== REPORT GENERATION FUNCTIONS ====================

def generate_html_report(summary_data, user_name="User"):
    """Generate an HTML report for 30-day summary"""
    today = date.today().strftime("%B %d, %Y")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Zim Smart Credit - 30-Day Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
            .header {{ text-align: center; border-bottom: 3px solid #1f77b4; padding-bottom: 20px; margin-bottom: 30px; }}
            .title {{ color: #1f77b4; font-size: 28px; font-weight: bold; }}
            .subtitle {{ color: #666; font-size: 16px; }}
            .section {{ margin-bottom: 30px; }}
            .section-title {{ color: #2e86ab; font-size: 20px; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            .metrics {{ display: flex; justify-content: space-between; flex-wrap: wrap; }}
            .metric-card {{ background: #f8f9fa; border-left: 4px solid #1f77b4; padding: 15px; margin: 10px; flex: 1; min-width: 200px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #1f77b4; }}
            .metric-label {{ color: #666; font-size: 14px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th {{ background: #1f77b4; color: white; padding: 12px; text-align: left; }}
            td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
            tr:nth-child(even) {{ background: #f9f9f9; }}
            .insight {{ background: #e8f4fd; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            .footer {{ text-align: center; margin-top: 50px; color: #999; font-size: 12px; border-top: 1px solid #eee; padding-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="title">ZIM SMART CREDIT APP</div>
            <div class="subtitle">30-Day Assessment Summary Report</div>
            <div>Generated for: {user_name} | Date: {today}</div>
            <div>Report Period: {summary_data['date_range']['start']} to {summary_data['date_range']['end']}</div>
        </div>
        
        <div class="section">
            <div class="section-title">📊 EXECUTIVE SUMMARY</div>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value">{summary_data['total_assessments']}</div>
                    <div class="metric-label">Total Assessments</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary_data['average_score']:.1f}/100</div>
                    <div class="metric-label">Average Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{max(summary_data['risk_distribution'].items(), key=lambda x: x[1])[0] if summary_data['risk_distribution'] else 'N/A'}</div>
                    <div class="metric-label">Most Common Risk Level</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">🎯 RISK DISTRIBUTION</div>
    """
    
    # Risk distribution table
    if summary_data['risk_distribution']:
        html_content += """
            <table>
                <tr>
                    <th>Risk Level</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
        """
        
        for risk, count in summary_data['risk_distribution'].items():
            percentage = (count / summary_data['total_assessments']) * 100
            color = "#28a745" if risk == "Low" else "#ffc107" if risk == "Medium" else "#dc3545"
            html_content += f"""
                <tr>
                    <td><span style="color: {color}; font-weight: bold;">{risk}</span></td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
            """
        
        html_content += "</table>"
    
    # Key Insights
    html_content += """
        </div>
        
        <div class="section">
            <div class="section-title">💡 KEY INSIGHTS</div>
    """
    
    insights = [
        f"The system processed <strong>{summary_data['total_assessments']}</strong> assessments in the last 30 days",
        f"Average credit score of <strong>{summary_data['average_score']:.1f}/100</strong> indicates overall moderate creditworthiness",
        "Mobile money transactions remain the strongest predictor of creditworthiness",
        "Consistent utility payments correlate with score stability"
    ]
    
    for insight in insights:
        html_content += f'<div class="insight">✓ {insight}</div>'
    
    # Recommendations
    html_content += """
        </div>
        
        <div class="section">
            <div class="section-title">📝 RECOMMENDATIONS</div>
    """
    
    recommendations = [
        "Continue focusing on mobile money transaction frequency for credit assessment",
        "Encourage regular utility payments to improve credit scores",
        "Consider implementing credit education programs for high-risk individuals",
        "Review and update the machine learning model quarterly for optimal performance",
        "Expand data collection to include savings behavior for more comprehensive assessment"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        html_content += f'<p>{i}. {rec}</p>'
    
    # Footer
    html_content += f"""
        </div>
        
        <div class="footer">
            <p>Zim Smart Credit App | Alternative Credit Scoring for Zimbabwe</p>
            <p>Report generated on {today} | All data is anonymized and aggregated</p>
            <p><em>Confidential - For internal use only</em></p>
        </div>
    </body>
    </html>
    """
    
    return html_content

def create_download_link(content, filename, file_type="html"):
    """Create a download link for HTML or text content"""
    b64 = base64.b64encode(content.encode()).decode()
    
    if file_type == "html":
        mime_type = "text/html"
        button_text = "📄 Download HTML Report"
    elif file_type == "txt":
        mime_type = "text/plain"
        button_text = "📝 Download Text Report"
    else:
        mime_type = "text/plain"
        button_text = "📥 Download Report"
    
    href = f'''
    <a href="data:{mime_type};base64,{b64}" download="{filename}" 
       style="text-decoration: none; padding: 10px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
              color: white; border-radius: 5px; font-weight: bold; display: inline-block; margin: 5px;">
       {button_text}
    </a>
    '''
    return href

def generate_text_report(summary_data, user_name="User"):
    """Generate a simple text report"""
    today = date.today().strftime("%B %d, %Y")
    
    report = f"""
    {'='*60}
    ZIM SMART CREDIT APP - 30-DAY SUMMARY REPORT
    {'='*60}
    
    Generated for: {user_name}
    Report Date: {today}
    Period: {summary_data['date_range']['start']} to {summary_data['date_range']['end']}
    
    {'-'*60}
    EXECUTIVE SUMMARY
    {'-'*60}
    • Total Assessments: {summary_data['total_assessments']}
    • Average Credit Score: {summary_data['average_score']:.1f}/100
    • Assessment Period: Last 30 days
    
    {'-'*60}
    RISK DISTRIBUTION
    {'-'*60}
    """
    
    if summary_data['risk_distribution']:
        for risk, count in summary_data['risk_distribution'].items():
            percentage = (count / summary_data['total_assessments']) * 100
            report += f"• {risk}: {count} assessments ({percentage:.1f}%)\n"
    
    report += f"""
    {'-'*60}
    KEY INSIGHTS
    {'-'*60}
    1. Processed {summary_data['total_assessments']} credit assessments
    2. Average score of {summary_data['average_score']:.1f}/100 indicates moderate creditworthiness
    3. Mobile transactions remain the strongest credit predictor
    4. Utility payment consistency correlates with score stability
    
    {'-'*60}
    RECOMMENDATIONS
    {'-'*60}
    1. Focus on mobile money transaction frequency
    2. Encourage regular utility payments
    3. Implement credit education programs
    4. Quarterly model review and updates
    5. Consider additional data sources for assessment
    
    {'='*60}
    Report generated: {today}
    Zim Smart Credit App | Confidential
    {'='*60}
    """
    
    return report

# ==================== DASHBOARD VISUALIZATION FUNCTIONS ====================

def create_score_trend_chart(summary_data):
    """Create a line chart showing score trend over time"""
    if not summary_data.get('score_trend'):
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=summary_data['score_trend']['dates'],
        y=summary_data['score_trend']['scores'],
        mode='lines+markers',
        name='Average Daily Score',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    # Add trend line if we have enough data points
    if len(summary_data['score_trend']['scores']) > 2:
        x_numeric = list(range(len(summary_data['score_trend']['scores'])))
        z = np.polyfit(x_numeric, summary_data['score_trend']['scores'], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=summary_data['score_trend']['dates'],
            y=p(x_numeric),
            mode='lines',
            name='Trend Line',
            line=dict(color='red', dash='dash', width=2)
        ))
    
    fig.update_layout(
        title='📈 Credit Score Trend (Last 30 Days)',
        xaxis_title='Date',
        yaxis_title='Average Score',
        height=400,
        hovermode='x unified',
        plot_bgcolor='rgba(240, 240, 240, 0.8)',
        paper_bgcolor='rgba(255, 255, 255, 0.9)'
    )
    
    return fig

def create_risk_distribution_chart(summary_data):
    """Create a pie chart showing risk distribution"""
    if not summary_data.get('risk_distribution'):
        return None
    
    labels = list(summary_data['risk_distribution'].keys())
    values = list(summary_data['risk_distribution'].values())
    
    colors_risk = {
        'Low': '#28a745',
        'Medium': '#ffc107',
        'High': '#dc3545'
    }
    
    pie_colors = [colors_risk.get(label, '#6c757d') for label in labels]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        marker_colors=pie_colors
    )])
    
    fig.update_layout(
        title='🎯 Risk Level Distribution',
        height=400,
        plot_bgcolor='rgba(240, 240, 240, 0.8)',
        paper_bgcolor='rgba(255, 255, 255, 0.9)'
    )
    
    return fig

# ==================== MAIN APP CODE ====================

# Initialize assessment history
initialize_assessment_history()

# Header Section
st.markdown('<h1 class="main-header glowing-text">🏦 Zim Smart Credit App</h1>', unsafe_allow_html=True)
st.markdown("### 💳 Revolutionizing Credit Scoring with Alternative Data in Zimbabwe")
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
    
    # User name for reporting
    st.markdown("---")
    user_name = st.text_input("👤 Your Name (for reports)", "Valued Customer")

# Main content with tabs - ADDED NEW REPORTS TAB
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🔍 Analysis", "🎯 Quick Assessment", "🤖 Advanced ML Model", "📈 Reports"])

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
    st.markdown("### 🔍 Data Analysis & Insights")
    
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["📊 Distributions", "📈 Statistics", "🌍 Geographic"])
    
    with analysis_tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Credit Score Distribution")
            score_counts = df['Credit_Score'].value_counts().sort_index()
            st.bar_chart(score_counts)
            
            # Show as table
            st.markdown("**Count by Credit Score:**")
            dist_df = score_counts.reset_index()
            dist_df.columns = ['Credit Score', 'Count']
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### Location Distribution")
            location_counts = df['Location'].value_counts()
            st.bar_chart(location_counts)
            
            st.markdown("**Count by Location:**")
            loc_df = location_counts.reset_index()
            loc_df.columns = ['Location', 'Count']
            st.dataframe(loc_df, use_container_width=True, hide_index=True)
    
    with analysis_tab2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        selected_feature = st.selectbox("Select feature for detailed analysis:", numeric_cols)
        
        if selected_feature:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### {selected_feature} Distribution")
                hist_values = np.histogram(df[selected_feature], bins=20)[0]
                st.bar_chart(hist_values)
            
            with col2:
                st.markdown(f"#### 📊 Statistics for {selected_feature}")
                stats_data = {
                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th %ile', '75th %ile'],
                    'Value': [
                        f"{df[selected_feature].mean():.2f}",
                        f"{df[selected_feature].median():.2f}",
                        f"{df[selected_feature].std():.2f}",
                        f"{df[selected_feature].min():.2f}",
                        f"{df[selected_feature].max():.2f}",
                        f"{df[selected_feature].quantile(0.25):.2f}",
                        f"{df[selected_feature].quantile(0.75):.2f}"
                    ]
                }
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with analysis_tab3:
        st.markdown("#### Credit Scores by Location")
        location_summary = df.groupby('Location')['Credit_Score'].value_counts().unstack().fillna(0)
        st.dataframe(location_summary, use_container_width=True)
        
        st.markdown("#### Location Performance Summary")
        location_stats = df.groupby('Location').agg({
            'Credit_Score': lambda x: (x == 'Good').mean()  # Example metric
        }).round(3)
        st.dataframe(location_stats, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== UPDATED QUICK ASSESSMENT TAB (tab3) ====================

with tab3:  # RENAMED to "🎯 Quick Assessment"
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("### 🎯 AI-Powered Credit Assessment")
    
    # Input summary in beautiful cards
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
        
        with st.spinner("🌳 Random Forest analyzing your financial profile..."):
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
                
                # Predict with Random Forest
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
                score_percentage = confidence  # Use confidence as score percentage
                
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
                        st.markdown("""
                        <div class="success-box">
                            <h3>✅ EXCELLENT CREDITWORTHINESS</h3>
                            <p><strong>Recommendation:</strong> Strong candidate for credit approval with favorable terms and higher limits</p>
                            <p><strong>Risk Level:</strong> Low</p>
                            <p><strong>AI Confidence:</strong> {:.1f}%</p>
                        </div>
                        """.format(confidence), unsafe_allow_html=True)
                    elif predicted_class == 'Fair':
                        st.markdown("""
                        <div class="warning-box">
                            <h3>⚠️ MODERATE RISK PROFILE</h3>
                            <p><strong>Recommendation:</strong> Standard verification process with moderate credit limits</p>
                            <p><strong>Risk Level:</strong> Medium</p>
                            <p><strong>AI Confidence:</strong> {:.1f}%</p>
                        </div>
                        """.format(confidence), unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="danger-box">
                            <h3>❌ HIGHER RISK PROFILE</h3>
                            <p><strong>Recommendation:</strong> Enhanced verification and possible collateral required</p>
                            <p><strong>Risk Level:</strong> High</p>
                            <p><strong>AI Confidence:</strong> {:.1f}%</p>
                        </div>
                        """.format(confidence), unsafe_allow_html=True)
                
                # Probability distribution
                st.markdown("#### 📊 AI Probability Distribution")
                prob_df = pd.DataFrame({
                    'Credit Score': st.session_state.target_encoder.classes_,
                    'Probability (%)': (prediction_proba[0] * 100).round(2)
                }).sort_values('Probability (%)', ascending=False)
                
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
                
                # Save assessment button
                st.markdown("---")
                if st.button("💾 Save This Assessment", type="primary", use_container_width=True):
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
                        'assessment_type': 'AI-Powered'
                    }
                    
                    save_assessment(assessment_data)
                    st.success(f"✅ AI assessment saved successfully! Total assessments: {len(st.session_state.assessment_history)}")
                
            except Exception as e:
                st.error(f"❌ Error in AI assessment: {str(e)}")
                st.info("Please ensure the AI model has been trained in the Advanced ML Model tab first.")
    else:
        st.warning("""
        ⚠️ **AI Model Not Ready**
        
        To use AI-powered credit assessment:
        
        1. Go to the **🤖 Advanced ML Model** tab
        2. Click **"Train Random Forest Model"**
        3. Once trained, return here for AI-powered predictions
        """)
        
        # Fallback to simple assessment if AI not ready
        st.markdown("---")
        st.markdown("#### 📊 Basic Assessment (Fallback)")
        
        # Simple assessment calculation (as backup)
        score = 0
        max_score = 6
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 🎂 Age Factor")
            if 30 <= Age <= 50:
                score += 2
                st.success("✅ Optimal (30-50 years)")
                st.progress(1.0)
            elif 25 <= Age < 30 or 50 < Age <= 60:
                score += 1
                st.warning("⚠️ Moderate")
                st.progress(0.5)
            else:
                st.error("❌ Higher Risk")
                st.progress(0.2)
        
        with col2:
            st.markdown("##### 💰 Transaction Activity")
            mobile_median = df['Mobile_Money_Txns'].median()
            if Mobile_Money_Txns > mobile_median:
                score += 1
                st.success(f"✅ Above Average")
                st.progress(1.0)
            else:
                st.warning("⚠️ Below Average")
                st.progress(0.3)
        
        with col3:
            st.markdown("##### 📈 Repayment History")
            repayment_scores = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
            rep_score = repayment_scores[Loan_Repayment_History]
            score += rep_score
            progress_map = {'Poor': 0.2, 'Fair': 0.4, 'Good': 0.7, 'Excellent': 1.0}
            st.info(f"📊 {Loan_Repayment_History}")
            st.progress(progress_map[Loan_Repayment_History])
        
        # Final assessment
        st.markdown("---")
        percentage = (score / max_score) * 100
        
        # Determine risk level
        if percentage >= 80:
            risk_level = "Low"
        elif percentage >= 50:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### 📈 Basic Score")
            st.markdown(f"# {score}/{max_score}")
            st.markdown(f"### {percentage:.1f}%")
            st.progress(percentage / 100)
            
            # Score interpretation
            if score >= 5:
                st.success("🎉 Excellent Score!")
            elif score >= 3:
                st.info("📊 Good Score")
            else:
                st.warning("📝 Needs Improvement")
        
        with col2:
            st.markdown("#### 🎯 Basic Assessment")
            if score >= 5:
                st.markdown("""
                <div class="success-box">
                    <h3>✅ GOOD CREDITWORTHINESS</h3>
                    <p><strong>Recommendation:</strong> Consider applying for credit with standard terms</p>
                    <p><strong>Risk Level:</strong> Low</p>
                    <p><em>Note: For more accurate assessment, train the AI model</em></p>
                </div>
                """, unsafe_allow_html=True)
            elif score >= 3:
                st.markdown("""
                <div class="warning-box">
                    <h3>⚠️ MODERATE RISK PROFILE</h3>
                    <p><strong>Recommendation:</strong> Consider credit building options first</p>
                    <p><strong>Risk Level:</strong> Medium</p>
                    <p><em>Note: For more accurate assessment, train the AI model</em></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="danger-box">
                    <h3>❌ HIGHER RISK PROFILE</h3>
                    <p><strong>Recommendation:</strong> Focus on improving financial habits</p>
                    <p><strong>Risk Level:</strong> High</p>
                    <p><em>Note: For more accurate assessment, train the AI model</em></p>
                </div>
                """, unsafe_allow_html=True)
        
        # Save basic assessment button
        st.markdown("---")
        if st.button("💾 Save Basic Assessment", type="primary", use_container_width=True):
            assessment_data = {
                'user_name': user_name,
                'location': Location,
                'gender': gender,
                'age': Age,
                'mobile_money_txns': Mobile_Money_Txns,
                'airtime_spend': Airtime_Spend_ZWL,
                'utility_payments': Utility_Payments_ZWL,
                'repayment_history': Loan_Repayment_History,
                'final_score': percentage,
                'risk_level': risk_level,
                'score_breakdown': score,
                'max_score': max_score,
                'assessment_type': 'Basic'
            }
            
            save_assessment(assessment_data)
            st.success(f"✅ Basic assessment saved successfully! Total assessments: {len(st.session_state.assessment_history)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== UPDATED AI MODEL TAB (tab4) ====================

with tab4:  # RENAMED to "🤖 Advanced ML Model"
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("### 🤖 Advanced Machine Learning Model")
    
    # System explanation
    st.markdown("""
    <div class="system-explanation">
        <h4>🔬 About This Model</h4>
        <p>This Random Forest model powers the <strong>AI-Powered Credit Assessment</strong> in the previous tab.</p>
        <p><strong>Key Features:</strong></p>
        <ul>
            <li>🌳 <strong>Random Forest Algorithm</strong>: Ensemble of decision trees for robust predictions</li>
            <li>📊 <strong>Trained on Historical Data</strong>: Learns patterns from Zimbabwe credit data</li>
            <li>🎯 <strong>Powers Real Assessments</strong>: Once trained, used for live credit scoring</li>
            <li>🔍 <strong>Explainable AI</strong>: Shows feature importance and confidence scores</li>
        </ul>
        <p><em>Train this model once, then use it for AI-powered assessments!</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Random Forest parameters configuration
    st.markdown("#### ⚙️ Model Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_estimators = st.slider("Number of Trees", 10, 500, 100, 10,
                                help="Number of decision trees in the forest")
    
    with col2:
        max_depth = st.slider("Max Tree Depth", 2, 20, 10, 1,
                             help="Maximum depth of each decision tree")
    
    with col3:
        min_samples_split = st.slider("Min Samples Split", 2, 20, 5, 1,
                                     help="Minimum number of samples required to split an internal node")
    
    # Additional model parameters
    st.markdown("#### 🔧 Advanced Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider("Test Size %", 10, 40, 20, 5,
                             help="Percentage of data to use for testing")
    
    with col2:
        random_state = st.slider("Random State", 0, 100, 42, 1,
                                help="Random seed for reproducibility")
    
    with col3:
        use_oob = st.checkbox("Use OOB Score", value=True,
                             help="Use Out-of-Bag samples for validation")
    
    if st.button("🌳 Train AI Model", type="primary", use_container_width=True):
        with st.spinner("🌳 Training AI model... This may take a few moments."):
            try:
                # Prepare data
                X = df.drop("Credit_Score", axis=1)
                y = df["Credit_Score"]
                
                # Check class distribution
                st.info(f"📊 Training on {len(df)} historical credit records")
                class_distribution = y.value_counts()
                
                # Encode categorical variables
                label_encoders = {}
                for column in X.select_dtypes(include=['object']).columns:
                    le = LabelEncoder()
                    X[column] = le.fit_transform(X[column])
                    label_encoders[column] = le
                
                # Encode target
                target_encoder = LabelEncoder()
                y_encoded = target_encoder.fit_transform(y)
                
                # Check if any class has only 1 sample
                unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
                
                if any(class_counts < 2):
                    st.warning(f"⚠️ Some classes have very few samples. Using simple train-test split.")
                    # Split data without stratification
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_encoded, 
                        test_size=test_size/100, 
                        random_state=random_state
                    )
                else:
                    # Split data with stratification
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_encoded, 
                        test_size=test_size/100, 
                        random_state=random_state, 
                        stratify=y_encoded
                    )
                
                # Train Random Forest model with user parameters
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=random_state,
                    n_jobs=-1,  # Use all available processors
                    oob_score=use_oob,
                    bootstrap=True
                )
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
                
                # ============= CALCULATE ACCURACY METRICS =============
                accuracy = accuracy_score(y_test, y_pred)
                train_accuracy = model.score(X_train, y_train)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # For multiclass ROC-AUC
                try:
                    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                    roc_auc_formatted = f"{roc_auc:.3f}"
                except:
                    roc_auc = None
                    roc_auc_formatted = "N/A"
                
                # Save model and encoders to session state
                st.session_state.trained_model = model
                st.session_state.label_encoders = label_encoders
                st.session_state.target_encoder = target_encoder
                st.session_state.model_accuracy = accuracy
                
                st.success("""
                ✅ AI Model Trained Successfully!
                
                **Now go to the 🎯 Quick Assessment tab to use this model for live credit scoring!**
                """)
                
                # ============= CLEAR ACCURACY PERCENTAGE DISPLAY =============
                st.markdown("---")
                st.markdown("### 🎯 MODEL ACCURACY RESULTS")
                
                # Simple, clear accuracy display
                accuracy_percentage = accuracy * 100
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 25px; border-radius: 15px; color: white; margin: 10px;">
                        <h3 style="margin: 0; font-size: 16px;">OVERALL ACCURACY</h3>
                        <h1 style="margin: 10px 0; font-size: 42px; font-weight: bold;">{accuracy:.1%}</h1>
                        <p style="margin: 0; font-size: 14px;">Test Set Performance</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="text-align: center; background: #28a745; padding: 25px; border-radius: 15px; 
                                color: white; margin: 10px;">
                        <h3 style="margin: 0; font-size: 16px;">PRECISION</h3>
                        <h1 style="margin: 10px 0; font-size: 42px; font-weight: bold;">{precision:.1%}</h1>
                        <p style="margin: 0; font-size: 14px;">Minimizes false approvals</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div style="text-align: center; background: #17a2b8; padding: 25px; border-radius: 15px; 
                                color: white; margin: 10px;">
                        <h3 style="margin: 0; font-size: 16px;">RECALL</h3>
                        <h1 style="margin: 10px 0; font-size: 42px; font-weight: bold;">{recall:.1%}</h1>
                        <p style="margin: 0; font-size: 14px;">Minimizes false rejections</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div style="text-align: center; background: #ffc107; padding: 25px; border-radius: 15px; 
                                color: white; margin: 10px;">
                        <h3 style="margin: 0; font-size: 16px;">F1-SCORE</h1>
                        <h1 style="margin: 10px 0; font-size: 42px; font-weight: bold;">{f1:.1%}</h1>
                        <p style="margin: 0; font-size: 14px;">Balanced performance</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Simple interpretation
                st.markdown(f"""
                <div style="background: #e8f4fd; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #1f77b4;">
                    <h4 style="margin: 0 0 15px 0; color: #1f77b4;">📊 AI MODEL PERFORMANCE SUMMARY</h4>
                    <p style="margin: 0; font-size: 16px; line-height: 1.6;">
                        The Random Forest AI model achieves <strong>{accuracy:.1%} overall accuracy</strong> 
                        ({accuracy_percentage:.0f} out of 100 correct predictions).
                        This model is now <strong>ready to power real-time credit assessments</strong> in the Quick Assessment tab.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # ============= DETAILED METRICS SECTION =============
                st.markdown("---")
                st.markdown("#### 📋 Detailed Performance Metrics")
                
                # Per-class metrics with error handling
                try:
                    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
                    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
                    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
                except:
                    # If per-class metrics fail, use weighted averages
                    precision_per_class = [precision] * len(target_encoder.classes_)
                    recall_per_class = [recall] * len(target_encoder.classes_)
                    f1_per_class = [f1] * len(target_encoder.classes_)
                
                # Cross-validation score with error handling
                try:
                    cv_scores = cross_val_score(model, X, y_encoded, cv=min(5, len(X)), scoring='accuracy')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                except:
                    cv_mean = accuracy
                    cv_std = 0
                    st.warning("⚠️ Cross-validation could not be performed with current data distribution")
                
                # ============= FEATURE IMPORTANCE SECTION =============
                st.markdown("---")
                st.markdown("#### 🔍 Random Forest Feature Importance")
                
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Create two columns for visualization and table
                col1, col2 = st.columns(2)
                
                with col1:
                    # Interactive feature importance plot
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
                
                with col2:
                    # Display as styled dataframe
                    def color_feature_importance(val):
                        if val >= 0.2:
                            return 'background-color: rgba(0, 255, 0, 0.3)'
                        elif val >= 0.1:
                            return 'background-color: rgba(255, 255, 0, 0.3)'
                        else:
                            return 'background-color: rgba(255, 0, 0, 0.3)'
                    
                    styled_features = feature_importance.style.applymap(
                        color_feature_importance, subset=['Importance']
                    )
                    
                    st.dataframe(styled_features, 
                               use_container_width=True, 
                               hide_index=True,
                               height=400)
                
                # ============= TEST THE TRAINED MODEL WITH USER INPUT =============
                st.markdown("---")
                st.markdown("#### 🎯 Test the Trained Model")
                
                if st.button("🔮 Test with My Data", type="secondary", use_container_width=True):
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
                    
                    # Predict with Random Forest
                    prediction_encoded = model.predict(user_data)
                    prediction_proba = model.predict_proba(user_data)
                    
                    predicted_class = target_encoder.inverse_transform(prediction_encoded)[0]
                    confidence = np.max(prediction_proba) * 100
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="success-box">
                            <h3>Model Prediction</h3>
                            <h1>{predicted_class}</h1>
                            <p><strong>Confidence:</strong> {confidence:.1f}%</p>
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
                st.error(f"❌ Error training AI model: {str(e)}")
                st.exception(e)
    
    # If model is already trained, show status
    elif 'trained_model' in st.session_state:
        st.markdown("---")
        st.markdown("""
        <div class="success-box">
            <h3>✅ AI Model Already Trained</h3>
            <p>The Random Forest model is ready for use in the <strong>🎯 Quick Assessment</strong> tab.</p>
            <p><strong>Model Accuracy:</strong> {:.1%}</p>
        </div>
        """.format(st.session_state.model_accuracy), unsafe_allow_html=True)
        
        # Option to retrain
        if st.button("🔄 Retrain Model with New Parameters", type="secondary"):
            st.session_state.pop('trained_model', None)
            st.session_state.pop('label_encoders', None)
            st.session_state.pop('target_encoder', None)
            st.rerun()
    
    # Model information section
    st.markdown("---")
    st.markdown("""
    <div class="card">
        <h4>🌳 How Random Forest Works for Credit Scoring</h4>
        <ol>
            <li><strong>Multiple Decision Trees:</strong> Creates an ensemble of decision trees from random data subsets</li>
            <li><strong>Voting System:</strong> Each tree makes a prediction, majority vote determines final result</li>
            <li><strong>Feature Importance:</strong> Calculates which features most influence credit decisions</li>
            <li><strong>Robustness:</strong> Reduces overfitting compared to single decision trees</li>
            <li><strong>Confidence Scores:</strong> Provides probability estimates for each prediction</li>
        </ol>
        <p><em>This approach makes credit assessments more accurate and reliable.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== REPORTS TAB (tab5) ====================

with tab5:  # REPORTS TAB
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("### 📈 30-Day Assessment Reports")
    
    # Check if we have assessment history
    recent_assessments = get_last_30_days_assessments()
    
    if not recent_assessments:
        st.warning("📭 No assessment history found for the last 30 days.")
        st.info("Complete an assessment in the '🎯 Assessment' tab and click 'Save This Assessment' to generate reports.")
    else:
        # Calculate summary
        summary = calculate_30_day_summary()
        
        # Display summary statistics
        st.markdown("#### 📊 Executive Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Assessments", summary['total_assessments'])
        with col2:
            st.metric("Average Score", f"{summary['average_score']:.1f}/100")
        with col3:
            most_common_risk = max(summary['risk_distribution'].items(), key=lambda x: x[1])[0] if summary['risk_distribution'] else "N/A"
            st.metric("Most Common Risk", most_common_risk)
        with col4:
            date_range = f"{summary['date_range']['start']} to {summary['date_range']['end']}"
            st.metric("Period", date_range)
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Score trend chart
            trend_fig = create_score_trend_chart(summary)
            if trend_fig:
                st.plotly_chart(trend_fig, use_container_width=True)
            else:
                st.info("Not enough data for trend analysis")
        
        with col2:
            # Risk distribution chart
            risk_fig = create_risk_distribution_chart(summary)
            if risk_fig:
                st.plotly_chart(risk_fig, use_container_width=True)
            else:
                st.info("Risk distribution data not available")
        
        st.markdown("---")
        
        # Report Generation Section
        st.markdown("#### 📄 Generate Comprehensive Report")
        
        st.markdown("""
        <div class="report-box">
            <h4>📊 Report Includes:</h4>
            <ul>
                <li>Executive Summary with key metrics</li>
                <li>Risk level distribution analysis</li>
                <li>Score trend visualization</li>
                <li>Detailed assessment history</li>
                <li>Key insights and recommendations</li>
                <li>Professional formatting</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Additional report options
            include_trends = st.checkbox("Include trend analysis", value=True)
            include_details = st.checkbox("Include detailed history", value=True)
            include_recommendations = st.checkbox("Include recommendations", value=True)
        
        with col2:
            # Generate HTML report
            if st.button("📥 Generate HTML Report", type="primary", use_container_width=True):
                with st.spinner("Generating HTML report..."):
                    try:
                        html_report = generate_html_report(summary, user_name)
                        
                        # Create download link
                        st.markdown(
                            create_download_link(html_report, f"Zim_Credit_Report_{date.today()}.html", "html"),
                            unsafe_allow_html=True
                        )
                        
                        st.success("✅ HTML report generated successfully!")
                        
                        # Show preview
                        st.markdown("#### 👁️ Report Preview")
                        st.components.v1.html(html_report, height=400, scrolling=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error generating report: {str(e)}")
        
        with col3:
            # Export to CSV
            if st.button("📊 Export to CSV", type="secondary", use_container_width=True):
                history_df = pd.DataFrame(recent_assessments)
                csv = history_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'''
                <a href="data:file/csv;base64,{b64}" download="credit_assessments_{date.today()}.csv" 
                   style="text-decoration: none; padding: 10px 20px; background: #6c757d; color: white; 
                          border-radius: 5px; font-weight: bold; display: inline-block; margin: 5px;">
                   📊 Download CSV
                </a>
                '''
                st.markdown(href, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed Assessment History
        st.markdown("#### 📋 Detailed Assessment History")
        
        # Create a DataFrame for display
        history_df = pd.DataFrame(recent_assessments)
        
        # Select columns to display
        display_columns = ['date', 'user_name', 'final_score', 'risk_level', 'age', 
                          'mobile_money_txns', 'repayment_history']
        
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
                'age': 'Age',
                'mobile_money_txns': 'Mobile Txns',
                'repayment_history': 'Repayment History'
            }
            
            display_df.rename(columns=column_names, inplace=True)
            
            # Display with pagination
            st.dataframe(
                display_df,
                use_container_width=True,
                height=300
            )
        else:
            st.info("No detailed assessment data available")
        
        st.markdown("---")
        
        # Advanced Analytics Section
        st.markdown("#### 🔍 Advanced Analytics")
        
        if 'final_score' in history_df.columns and 'risk_level' in history_df.columns:
            analytics_tab1, analytics_tab2 = st.tabs(["Score Analysis", "Risk Correlation"])
            
            with analytics_tab1:
                # Score distribution histogram
                fig_hist = px.histogram(
                    history_df, 
                    x='final_score',
                    nbins=20,
                    title='Score Distribution',
                    labels={'final_score': 'Credit Score'},
                    color_discrete_sequence=['#1f77b4']
                )
                fig_hist.update_layout(bargap=0.1)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with analytics_tab2:
                # Correlation heatmap for numeric columns
                numeric_cols = history_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_matrix = history_df[numeric_cols].corr()
                    
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto='.2f',
                        aspect="auto",
                        title='Feature Correlation Matrix',
                        color_continuous_scale='RdBu'
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Not enough numeric data for correlation analysis")
        else:
            st.info("Advanced analytics require more assessment data")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 20px;">
    <p>Zim Smart Credit App | Alternative Credit Scoring for Zimbabwe | © 2024</p>
    <p><small>All assessments are stored locally in your browser session</small></p>
</div>
""", unsafe_allow_html=True)
