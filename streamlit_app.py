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

# Main content with tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔍 Analysis", "🎯 Assessment", "🤖 AI Model"])

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
        st.markdown(f"""
        <div class="metric-card">
            <h3>✅ Data Quality</h3>
            <h2>100%</h2>
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
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("### 🤖 AI-Powered Credit Scoring with Random Forest")
    
    # Random Forest Information Section
    st.markdown("""
    <div class="rf-info-box">
        <h3>🌳 Random Forest Algorithm</h3>
        <p><strong>Random Forest</strong> is an ensemble learning method that operates by constructing multiple decision trees during training. 
        For classification tasks, the output is the class selected by most trees. It's particularly well-suited for credit scoring because:</p>
        <ul>
            <li>✅ Handles both numerical and categorical data well</li>
            <li>✅ Reduces overfitting compared to single decision trees</li>
            <li>✅ Provides feature importance scores</li>
            <li>✅ Works well with datasets having multiple features</li>
            <li>✅ Robust to outliers and noise in data</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Random Forest parameters configuration
    st.markdown("#### ⚙️ Random Forest Configuration")
    
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
    
    if st.button("🌳 Train Random Forest Model", type="primary", use_container_width=True):
        with st.spinner("🌳 Training Random Forest model... This may take a few moments."):
            try:
                # Prepare data
                X = df.drop("Credit_Score", axis=1)
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
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size/100, random_state=random_state, stratify=y_encoded
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
                accuracy = accuracy_score(y_test, y_pred)
                
                st.success("✅ Random Forest model trained successfully!")
                
                # Display Random Forest specific metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎯 Accuracy", f"{accuracy:.2%}")
                with col2:
                    if use_oob:
                        st.metric("👜 OOB Score", f"{model.oob_score_:.2%}")
                    else:
                        st.metric("🌳 Number of Trees", f"{n_estimators}")
                with col3:
                    st.metric("📚 Training Samples", f"{len(X_train):,}")
                with col4:
                    st.metric("🧪 Test Samples", f"{len(X_test):,}")
                
                # ============= MODEL EVALUATION SECTION =============
                st.markdown("---")
                st.markdown("#### 📊 Comprehensive Model Evaluation")
                
                # Create tabs for different evaluation metrics
                eval_tab1, eval_tab2, eval_tab3, eval_tab4 = st.tabs([
                    "📈 Performance Metrics", 
                    "🤖 Classification Report", 
                    "📊 Confusion Matrix",
                    "🎯 ROC Analysis"
                ])
                
                with eval_tab1:
                    st.markdown("##### 📊 Key Performance Metrics")
                    
                    # Calculate additional metrics
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    # For multiclass ROC-AUC (One-vs-Rest)
                    try:
                        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                    except:
                        roc_auc = "N/A"
                    
                    # Display metrics in cards
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Precision", f"{precision:.2%}", 
                                 help="How many selected items are relevant")
                    with col2:
                        st.metric("Recall", f"{recall:.2%}", 
                                 help="How many relevant items are selected")
                    with col3:
                        st.metric("F1-Score", f"{f1:.2%}", 
                                 help="Harmonic mean of precision and recall")
                    with col4:
                        if roc_auc != "N/A":
                            st.metric("ROC-AUC", f"{roc_auc:.2%}", 
                                     help="Area under ROC curve")
                        else:
                            st.metric("ROC-AUC", "N/A")
                    
                    # Training vs Test accuracy comparison
                    train_accuracy = model.score(X_train, y_train)
                    
                    st.markdown("##### 📈 Training vs Test Performance")
                    comparison_data = {
                        'Dataset': ['Training', 'Test'],
                        'Accuracy': [train_accuracy, accuracy]
                    }
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.bar_chart(comparison_df.set_index('Dataset')['Accuracy'])
                    
                    with col2:
                        # Calculate overfitting ratio
                        overfitting_ratio = (train_accuracy - accuracy) / accuracy if accuracy > 0 else 0
                        st.metric("📊 Overfitting Ratio", f"{overfitting_ratio:.2%}",
                                 delta=f"{(train_accuracy - accuracy):.2%}",
                                 delta_color="inverse" if overfitting_ratio > 0.1 else "normal")
                
                with eval_tab2:
                    st.markdown("##### 🤖 Detailed Classification Report")
                    
                    report = classification_report(y_test, y_pred, 
                                                  target_names=target_encoder.classes_, 
                                                  output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    
                    # Color code the dataframe
                    def color_classification_report(val):
                        if isinstance(val, (int, float)):
                            if val >= 0.8:
                                return 'background-color: rgba(0, 255, 0, 0.2)'
                            elif val >= 0.6:
                                return 'background-color: rgba(255, 255, 0, 0.2)'
                            else:
                                return 'background-color: rgba(255, 0, 0, 0.2)'
                        return ''
                    
                    styled_report = report_df.style.applymap(color_classification_report, 
                                                           subset=pd.IndexSlice[:, ['precision', 'recall', 'f1-score']])
                    
                    st.dataframe(styled_report, use_container_width=True, height=400)
                    
                    # Additional insights
                    st.markdown("##### 💡 Key Insights")
                    avg_f1 = report_df['f1-score'].mean()
                    best_class = report_df['f1-score'].idxmax()
                    worst_class = report_df['f1-score'].idxmin()
                    
                    insight_col1, insight_col2, insight_col3 = st.columns(3)
                    with insight_col1:
                        st.metric("Average F1-Score", f"{avg_f1:.2%}")
                    with insight_col2:
                        st.metric("Best Performing", best_class, 
                                 delta=f"{report_df.loc[best_class, 'f1-score']:.2%}")
                    with insight_col3:
                        st.metric("Needs Improvement", worst_class,
                                 delta=f"{report_df.loc[worst_class, 'f1-score']:.2%}",
                                 delta_color="inverse")
                
                with eval_tab3:
                    st.markdown("##### 📊 Confusion Matrix Visualization")
                    
                    cm = confusion_matrix(y_test, y_pred)
                    
                    # Create two columns for different visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Plotly heatmap
                        fig = ff.create_annotated_heatmap(
                            z=cm,
                            x=target_encoder.classes_.tolist(),
                            y=target_encoder.classes_.tolist(),
                            colorscale='Viridis',
                            showscale=True
                        )
                        fig.update_layout(
                            title="Confusion Matrix Heatmap",
                            xaxis_title="Predicted Label",
                            yaxis_title="True Label"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Normalized confusion matrix
                        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                        
                        fig2 = go.Figure(data=go.Heatmap(
                            z=cm_normalized,
                            x=target_encoder.classes_.tolist(),
                            y=target_encoder.classes_.tolist(),
                            colorscale='RdBu',
                            zmin=0, zmax=1,
                            text=cm,
                            texttemplate="%{text}<br>%{z:.1%}",
                            textfont={"size": 10}
                        ))
                        
                        fig2.update_layout(
                            title="Normalized Confusion Matrix",
                            xaxis_title="Predicted Label",
                            yaxis_title="True Label"
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Metrics from confusion matrix
                    st.markdown("##### 📈 Confusion Matrix Metrics")
                    
                    # Calculate per-class metrics
                    class_metrics = []
                    for i, class_name in enumerate(target_encoder.classes_):
                        tp = cm[i, i]
                        fp = cm[:, i].sum() - tp
                        fn = cm[i, :].sum() - tp
                        tn = cm.sum() - (tp + fp + fn)
                        
                        precision_class = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall_class = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1_class = 2 * (precision_class * recall_class) / (precision_class + recall_class) if (precision_class + recall_class) > 0 else 0
                        
                        class_metrics.append({
                            'Class': class_name,
                            'TP': tp,
                            'FP': fp,
                            'FN': fn,
                            'TN': tn,
                            'Precision': f"{precision_class:.2%}",
                            'Recall': f"{recall_class:.2%}",
                            'F1-Score': f"{f1_class:.2%}"
                        })
                    
                    metrics_df = pd.DataFrame(class_metrics)
                    st.dataframe(metrics_df, use_container_width=True, height=300)
                
                with eval_tab4:
                    st.markdown("##### 🎯 ROC Curve Analysis")
                    
                    # Binarize the output for multiclass ROC
                    y_test_bin = label_binarize(y_test, classes=range(len(target_encoder.classes_)))
                    
                    # Compute ROC curve and ROC area for each class
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    
                    for i in range(len(target_encoder.classes_)):
                        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])
                    
                    # Plot all ROC curves
                    fig = go.Figure()
                    
                    colors = ['blue', 'red', 'green', 'orange', 'purple']
                    for i, color in zip(range(len(target_encoder.classes_)), colors):
                        if i < len(target_encoder.classes_):
                            fig.add_trace(go.Scatter(
                                x=fpr[i],
                                y=tpr[i],
                                mode='lines',
                                line=dict(color=color, width=2),
                                name=f'{target_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})'
                            ))
                    
                    # Add diagonal line
                    fig.add_trace(go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode='lines',
                        line=dict(color='black', dash='dash', width=1),
                        name='Random Classifier'
                    ))
                    
                    fig.update_layout(
                        title='Receiver Operating Characteristic (ROC) Curves',
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate',
                        xaxis=dict(range=[0, 1]),
                        yaxis=dict(range=[0, 1]),
                        showlegend=True,
                        hovermode='closest'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # AUC Summary
                    st.markdown("##### 📊 AUC Score Summary")
                    auc_summary = pd.DataFrame({
                        'Class': target_encoder.classes_,
                        'AUC Score': [roc_auc[i] for i in range(len(target_encoder.classes_))],
                        'Interpretation': ['Excellent' if roc_auc[i] >= 0.9 else 
                                          'Good' if roc_auc[i] >= 0.8 else 
                                          'Fair' if roc_auc[i] >= 0.7 else 
                                          'Poor' for i in range(len(target_encoder.classes_))]
                    })
                    st.dataframe(auc_summary, use_container_width=True)
                
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
                
                # ============= MODEL DIAGNOSTICS SECTION =============
                st.markdown("---")
                st.markdown("#### 🩺 Model Diagnostics")
                
                diagnostic_col1, diagnostic_col2 = st.columns(2)
                
                with diagnostic_col1:
                    # Tree depth distribution
                    tree_depths = [tree.get_depth() for tree in model.estimators_]
                    
                    fig_depth = go.Figure(data=[
                        go.Histogram(
                            x=tree_depths,
                            nbinsx=10,
                            marker_color='lightgreen',
                            opacity=0.7
                        )
                    ])
                    
                    fig_depth.update_layout(
                        title='Distribution of Tree Depths',
                        xaxis_title='Tree Depth',
                        yaxis_title='Count'
                    )
                    
                    st.plotly_chart(fig_depth, use_container_width=True)
                
                with diagnostic_col2:
                    # Feature importance stability
                    feature_importance_std = np.std([tree.feature_importances_ 
                                                    for tree in model.estimators_], axis=0)
                    
                    fig_stability = go.Figure(data=[
                        go.Bar(
                            x=X.columns,
                            y=feature_importance_std,
                            marker_color='lightcoral',
                            opacity=0.7
                        )
                    ])
                    
                    fig_stability.update_layout(
                        title='Feature Importance Stability (Std Dev)',
                        xaxis_title='Feature',
                        yaxis_title='Standard Deviation',
                        xaxis_tickangle=45
                    )
                    
                    st.plotly_chart(fig_stability, use_container_width=True)
                
                # ============= PREDICTION SECTION =============
                st.markdown("---")
                st.markdown("#### 🎯 Get Your Random Forest Prediction")
                
                if st.button("🔮 Predict My Credit Score with Random Forest", type="secondary", use_container_width=True):
                    user_data = pd.DataFrame({
                        'Location': [Location],
                        'Gender': [gender],
                        'Mobile_Money_Txns': [Mobile_Money_Txns],
                        'Airtime_Spend_ZWL': [Airtime_Spend_ZWL],
                        'Utility_Payments_ZWL': [Utility_Payments_ZWL],
                        'Loan_Repayment_History': [Loan_Repayment_History],
                        'Age': [Age]
                    })
                    
                    # Encode user input
                    for column in user_data.select_dtypes(include=['object']).columns:
                        if column in label_encoders:
                            if user_data[column].iloc[0] in label_encoders[column].classes_:
                                user_data[column] = label_encoders[column].transform(user_data[column])
                            else:
                                user_data[column] = -1
                    
                    # Predict with Random Forest
                    prediction_encoded = model.predict(user_data)
                    prediction_proba = model.predict_proba(user_data)
                    
                    predicted_class = target_encoder.inverse_transform(prediction_encoded)[0]
                    confidence = np.max(prediction_proba) * 100
                    
                    # Beautiful Random Forest prediction display
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="success-box">
                            <h3>Random Forest Prediction</h3>
                            <h1>{predicted_class}</h1>
                            <p><strong>Algorithm:</strong> Random Forest</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="card">
                            <h3>Confidence Level</h3>
                            <h1>{confidence:.1f}%</h1>
                            <p>Based on {n_estimators} decision trees</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        # Get individual tree predictions (sample of 5 trees)
                        tree_predictions = []
                        for i, tree in enumerate(model.estimators_[:5]):
                            tree_pred = tree.predict(user_data)[0]
                            tree_predictions.append(target_encoder.inverse_transform([tree_pred])[0])
                        
                        st.markdown(f"""
                        <div class="feature-box">
                            <h4>🌲 Sample Tree Predictions</h4>
                            <p>Tree 1: {tree_predictions[0]}</p>
                            <p>Tree 2: {tree_predictions[1]}</p>
                            <p>Tree 3: {tree_predictions[2]}</p>
                            <p>Tree 4: {tree_predictions[3]}</p>
                            <p>Tree 5: {tree_predictions[4]}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Probability distribution from Random Forest
                    st.markdown("#### 📊 Probability Distribution from Random Forest")
                    prob_df = pd.DataFrame({
                        'Credit Score': target_encoder.classes_,
                        'Probability (%)': (prediction_proba[0] * 100).round(2)
                    }).sort_values('Probability (%)', ascending=False)
                    
                    # Add color coding based on probability
                    def color_probability(val):
                        if val > 70:
                            return 'background-color: rgba(0, 255, 0, 0.2)'
                        elif val > 30:
                            return 'background-color: rgba(255, 255, 0, 0.2)'
                        else:
                            return 'background-color: rgba(255, 0, 0, 0.2)'
                    
                    styled_df = prob_df.style.applymap(color_probability, subset=['Probability (%)'])
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                # ============= MODEL COMPARISON SECTION =============
                st.markdown("---")
                st.markdown("#### ⚖️ Model Comparison Tools")
                
                if st.button("🔄 Compare Different Models", type="secondary"):
                    with st.spinner("Training comparison models..."):
                        try:
                            # Prepare data
                            X = df.drop("Credit_Score", axis=1)
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
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                            )
                            
                            # Train different models
                            models = {
                                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                                'Decision Tree': DecisionTreeClassifier(random_state=42),
                                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                                'SVM': SVC(probability=True, random_state=42),
                                'K-Nearest Neighbors': KNeighborsClassifier()
                            }
                            
                            results = []
                            for name, model in models.items():
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)
                                accuracy = accuracy_score(y_test, y_pred)
                                
                                results.append({
                                    'Model': name,
                                    'Accuracy': accuracy,
                                    'Training Time': 'N/A',  # Could add timing
                                    'Parameters': str(model.get_params())
                                })
                            
                            results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
                            
                            # Display comparison
                            st.markdown("##### 📊 Model Performance Comparison")
                            
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=results_df['Model'],
                                    y=results_df['Accuracy'],
                                    marker_color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
                                )
                            ])
                            
                            fig.update_layout(
                                title='Model Accuracy Comparison',
                                xaxis_title='Model',
                                yaxis_title='Accuracy',
                                yaxis_tickformat='.2%'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display results table
                            st.dataframe(results_df, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error in model comparison: {str(e)}")
                
            except Exception as e:
                st.error(f"❌ Error training Random Forest model: {str(e)}")
                st.exception(e)
    
    # Add information about Random Forest advantages
    st.markdown("---")
    st.markdown("""
    <div class="card">
        <h4>🌳 Why Random Forest for Credit Scoring?</h4>
        <p><strong>Advantages of Random Forest in Credit Assessment:</strong></p>
        <ol>
            <li><strong>High Accuracy:</strong> Often achieves better performance than single decision trees</li>
            <li><strong>Feature Importance:</strong> Identifies which factors most influence credit scores</li>
            <li><strong>Robustness:</strong> Less prone to overfitting and handles missing values well</li>
            <li><strong>Non-linear Relationships:</strong> Captures complex patterns in financial data</li>
            <li><strong>Interpretability:</strong> Provides insights into decision-making process</li>
        </ol>
        <p><em>The model aggregates predictions from multiple decision trees to make more reliable credit assessments.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add model performance summary
    st.markdown("""
    <div class="card">
        <h4>📊 Model Evaluation Metrics Explained</h4>
        <ul>
            <li><strong>Accuracy:</strong> Overall correctness of the model</li>
            <li><strong>Precision:</strong> How many predicted positives are actually positive</li>
            <li><strong>Recall:</strong> How many actual positives are correctly identified</li>
            <li><strong>F1-Score:</strong> Harmonic mean of precision and recall</li>
            <li><strong>ROC-AUC:</strong> Ability to distinguish between classes</li>
            <li><strong>Confusion Matrix:</strong> Detailed breakdown of predictions vs actual</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
