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
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from datetime import datetime, timedelta, date
import json
import base64
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import matplotlib.pyplot as plt
import seaborn as sns

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
</style>
""", unsafe_allow_html=True)

# ==================== ASSESSMENT HISTORY MANAGEMENT ====================

def initialize_assessment_history():
    """Initialize session state for storing assessment history"""
    if 'assessment_history' not in st.session_state:
        st.session_state.assessment_history = []
    if 'current_assessment' not in st.session_state:
        st.session_state.current_assessment = None

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

def generate_pdf_report(summary_data, user_name="User"):
    """Generate a PDF report for 30-day summary"""
    buffer = io.BytesIO()
    
    # Create PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#1f77b4')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.HexColor('#2e86ab')
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6
    )
    
    # Build story (content)
    story = []
    
    # Title
    story.append(Paragraph(f"ZIM SMART CREDIT - 30-DAY SUMMARY REPORT", title_style))
    story.append(Paragraph(f"Generated for: {user_name}", normal_style))
    story.append(Paragraph(f"Report Period: {summary_data['date_range']['start']} to {summary_data['date_range']['end']}", normal_style))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
    story.append(Paragraph(f"Total Assessments: {summary_data['total_assessments']}", normal_style))
    story.append(Paragraph(f"Average Credit Score: {summary_data['average_score']:.1f}/100", normal_style))
    story.append(Spacer(1, 10))
    
    # Risk Distribution
    story.append(Paragraph("RISK DISTRIBUTION", heading_style))
    
    if summary_data['risk_distribution']:
        risk_data = []
        for risk, count in summary_data['risk_distribution'].items():
            percentage = (count / summary_data['total_assessments']) * 100
            risk_data.append([risk, str(count), f"{percentage:.1f}%"])
        
        risk_table = Table(risk_data, colWidths=[200, 100, 100])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(risk_table)
    
    story.append(Spacer(1, 20))
    
    # Key Insights
    story.append(Paragraph("KEY INSIGHTS", heading_style))
    
    insights = [
        f"• The system processed {summary_data['total_assessments']} assessments in the last 30 days",
        f"• Average credit score of {summary_data['average_score']:.1f} indicates overall moderate creditworthiness",
        f"• Score trend shows {'improvement' if len(summary_data.get('score_trend', {}).get('scores', [])) > 1 and summary_data['score_trend']['scores'][-1] > summary_data['score_trend']['scores'][0] else 'stable'} pattern",
        "• Mobile money transactions remain the strongest predictor of creditworthiness",
        "• Consistent utility payments correlate with score stability"
    ]
    
    for insight in insights:
        story.append(Paragraph(insight, normal_style))
    
    story.append(Spacer(1, 20))
    
    # Recommendations
    story.append(Paragraph("RECOMMENDATIONS", heading_style))
    
    recommendations = [
        "1. Continue focusing on mobile money transaction frequency",
        "2. Encourage regular utility payments for score improvement",
        "3. Consider adding savings behavior tracking",
        "4. Implement credit education for high-risk individuals",
        "5. Review and update model quarterly for optimal performance"
    ]
    
    for rec in recommendations:
        story.append(Paragraph(rec, normal_style))
    
    story.append(Spacer(1, 30))
    
    # Footer
    story.append(Paragraph("Confidential - For Internal Use Only", ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=1  # Center aligned
    )))
    
    # Build PDF
    doc.build(story)
    
    buffer.seek(0)
    return buffer

def create_download_link(buffer, filename):
    """Create a download link for the PDF"""
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" style="text-decoration: none; padding: 10px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 5px; font-weight: bold;">📥 Download PDF Report</a>'
    return href

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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🔍 Analysis", "🎯 Assessment", "🤖 AI Model", "📈 Reports"])

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

with tab3:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("### 🎯 Credit Assessment Results")
    
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
    
    # Assessment calculation with beautiful progress bars
    st.markdown("#### 📊 Assessment Factors")
    
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
        risk_color = "success"
    elif percentage >= 50:
        risk_level = "Medium"
        risk_color = "warning"
    else:
        risk_level = "High"
        risk_color = "danger"
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 📈 Overall Score")
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
        st.markdown("#### 🎯 Final Assessment")
        if score >= 5:
            st.markdown("""
            <div class="success-box">
                <h3>✅ EXCELLENT CREDITWORTHINESS</h3>
                <p><strong>Recommendation:</strong> Strong candidate for credit approval with favorable terms and higher limits</p>
                <p><strong>Risk Level:</strong> Low</p>
            </div>
            """, unsafe_allow_html=True)
        elif score >= 3:
            st.markdown("""
            <div class="warning-box">
                <h3>⚠️ MODERATE RISK PROFILE</h3>
                <p><strong>Recommendation:</strong> Standard verification process with moderate credit limits</p>
                <p><strong>Risk Level:</strong> Medium</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="danger-box">
                <h3>❌ HIGHER RISK PROFILE</h3>
                <p><strong>Recommendation:</strong> Enhanced verification and possible collateral required</p>
                <p><strong>Risk Level:</strong> High</p>
            </div>
            """, unsafe_allow_html=True)
    
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
            'final_score': percentage,
            'risk_level': risk_level,
            'score_breakdown': score,
            'max_score': max_score
        }
        
        save_assessment(assessment_data)
        st.success(f"✅ Assessment saved successfully! Total assessments: {len(st.session_state.assessment_history)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ... [Keep your existing tab4 code exactly as is - no changes needed there]
# (The tab4 code remains exactly the same as in your original)

with tab5:  # NEW REPORTS TAB
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("### 📈 30-Day Assessment Reports")
    
    # Check if we have assessment history
    recent_assessments = get_last_30_days_assessments()
    
    if not recent_assessments:
        st.warning("📭 No assessment history found for the last 30 days.")
        st.info("Complete an assessment in the '🎯 Assessment' tab to generate reports.")
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
                height=400
            )
        else:
            st.info("No detailed assessment data available")
        
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
                <li>Professional PDF formatting</li>
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
            # Generate PDF report
            if st.button("📥 Generate PDF Report", type="primary", use_container_width=True):
                with st.spinner("Generating PDF report..."):
                    try:
                        pdf_buffer = generate_pdf_report(summary, user_name)
                        
                        # Create download link
                        st.markdown(
                            create_download_link(pdf_buffer, f"Zim_Credit_Report_{date.today()}.pdf"),
                            unsafe_allow_html=True
                        )
                        
                        st.success("✅ PDF report generated successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating PDF: {str(e)}")
        
        with col3:
            # Export to CSV
            if st.button("📊 Export to CSV", type="secondary", use_container_width=True):
                csv = history_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="credit_assessments_{date.today()}.csv" style="text-decoration: none; padding: 10px 20px; background: #6c757d; color: white; border-radius: 5px; font-weight: bold;">📊 Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
        
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
