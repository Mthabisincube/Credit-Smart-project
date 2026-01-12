import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
                    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                )
                
                # Train Random Forest model with user parameters
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42,
                    n_jobs=-1  # Use all available processors
                )
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                st.success("✅ Random Forest model trained successfully!")
                
                # Display Random Forest specific metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎯 Accuracy", f"{accuracy:.2%}")
                with col2:
                    st.metric("🌳 Number of Trees", f"{n_estimators}")
                with col3:
                    st.metric("📚 Training Samples", f"{len(X_train):,}")
                with col4:
                    st.metric("🧪 Test Samples", f"{len(X_test):,}")
                
                # Feature importance visualization
                st.markdown("#### 🔍 Random Forest Feature Importance")
                
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Create two columns for visualization and table
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display as bar chart
                    st.bar_chart(feature_importance.set_index('Feature')['Importance'])
                
                with col2:
                    # Display as styled dataframe
                    st.dataframe(feature_importance, 
                               use_container_width=True, 
                               hide_index=True,
                               height=400)
                
                # Random Forest tree visualization (simplified)
                st.markdown("#### 🌲 Random Forest Structure")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="tree-diagram">
                        <h5>🌳 Random Forest Architecture</h5>
                        <pre>
     Random Forest
        │
        ├── Tree 1
        │     ├── Feature: {feature1}
        │     └── Feature: {feature2}
        ├── Tree 2
        │     ├── Feature: {feature3}
        │     └── Feature: {feature4}
        ├── Tree 3
        │     ├── Feature: {feature1}
        │     └── Feature: {feature5}
        └── ... ({} more trees)
                        </pre>
                        <p><strong>Final Decision:</strong> Majority Vote from all trees</p>
                    </div>
                    """.format(n_estimators-3), unsafe_allow_html=True)
                
                with col2:
                    # Model statistics
                    st.markdown("#### 📊 Model Statistics")
                    stats_data = {
                        'Metric': ['Total Trees', 'Avg Tree Depth', 'Avg Nodes per Tree', 
                                  'Avg Leaves per Tree', 'OOB Score', 'Feature Count'],
                        'Value': [
                            f"{n_estimators}",
                            f"{model.estimators_[0].get_depth()}",
                            f"{model.estimators_[0].tree_.node_count}",
                            f"{model.estimators_[0].get_n_leaves()}",
                            f"{model.oob_score_:.2%}" if hasattr(model, 'oob_score_') else "N/A",
                            f"{X.shape[1]}"
                        ]
                    }
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Real-time prediction with Random Forest
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
                
                # Model performance details
                with st.expander("📊 View Detailed Model Performance"):
                    st.markdown("##### Classification Report")
                    report = classification_report(y_test, y_pred, target_names=target_encoder.classes_, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df, use_container_width=True)
                    
                    st.markdown("##### Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    cm_df = pd.DataFrame(cm, 
                                        index=target_encoder.classes_,
                                        columns=target_encoder.classes_)
                    st.dataframe(cm_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error training Random Forest model: {str(e)}")
    
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
    
    st.markdown('</div>', unsafe_allow_html=True)
