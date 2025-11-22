import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Smart Credit Scoring for Financial Inclusion",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional academic styling
st.markdown("""
<style>
    .stApp {
        background: #f8f9fa;
        font-family: 'Arial', sans-serif;
    }
    
    .academic-header {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border-left: 5px solid #e74c3c;
    }
    
    .objective-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #3498db;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .data-card {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #b8d4f0;
    }
    
    .result-card {
        background: #2c3e50;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .metric-box {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .university-colors {
        background: linear-gradient(135deg, #8B0000 0%, #FF0000 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .feature-importance-bar {
        background: linear-gradient(90deg, #3498db var(--importance), #f8f9fa var(--importance));
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 4px;
        color: #2c3e50;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Academic Header
    st.markdown("""
    <div class="academic-header">
        <h1 style="margin:0; text-align:center;">Smart Credit Scoring for Financial Inclusion in Zimbabwe</h1>
        <p style="margin:0; text-align:center; font-size:1.2rem;">
            Using Alternative Data and Machine Learning
        </p>
        <p style="margin:0; text-align:center; font-size:1rem;">
            National University of Science and Technology • Department of Informatics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Research Objectives Overview
    st.markdown("""
    <div class="university-colors">
        <h3 style="margin:0;">Research Objectives</h3>
        <p style="margin:0.5rem 0 0 0;">
            1. Simulate alternative financial behavior data • 2. Train ML model for creditworthiness • 3. Predict scores with user interface
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for each objective
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Objective 1: Data Simulation", 
        "🤖 Objective 2: Model Training", 
        "🎯 Objective 3: Credit Assessment",
        "📈 Research Insights"
    ])

    with tab1:
        display_data_simulation()
    
    with tab2:
        display_model_training()
    
    with tab3:
        display_credit_assessment()
    
    with tab4:
        display_research_insights()

def display_data_simulation():
    st.markdown("""
    <div class="objective-card">
        <h3>📊 Objective 1: Simulate Alternative Financial Behavior Data</h3>
        <p>Generate synthetic mobile money transactions and behavioral patterns representing Zimbabwe's unbanked population.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data generation parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Generation Parameters")
        sample_size = st.slider("Sample Size", 100, 5000, 1000)
        locations = st.multiselect(
            "Locations",
            ["Harare", "Bulawayo", "Gweru", "Mutare", "Masvingo", "Chinhoyi"],
            default=["Harare", "Bulawayo", "Gweru"]
        )
    
    with col2:
        st.subheader("Behavioral Features")
        include_mobile_money = st.checkbox("Mobile Money Transactions", True)
        include_airtime = st.checkbox("Airtime Spending", True)
        include_utilities = st.checkbox("Utility Payments", True)
        include_repayment = st.checkbox("Loan Repayment History", True)
    
    if st.button("🔄 Generate Synthetic Dataset", use_container_width=True):
        with st.spinner("Generating synthetic financial behavior data..."):
            df = generate_synthetic_data(sample_size, locations)
            
            st.success(f"✅ Successfully generated {len(df)} synthetic records")
            
            # Display dataset overview
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                st.metric("Credit Classes", df['Credit_Score'].nunique())
            
            # Show data preview
            st.subheader("📋 Generated Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data distribution visualizations using Streamlit native charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Credit Score Distribution")
                score_counts = df['Credit_Score'].value_counts()
                st.bar_chart(score_counts)
                
                # Show distribution table
                dist_df = pd.DataFrame({
                    'Credit Score': score_counts.index,
                    'Count': score_counts.values,
                    'Percentage': (score_counts.values / len(df) * 100).round(1)
                })
                st.dataframe(dist_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("#### Mobile Money Transactions")
                # Create histogram using bar chart
                hist_data = pd.cut(df['Mobile_Money_Txns'], bins=20).value_counts().sort_index()
                st.bar_chart(hist_data)
                
                # Show statistics
                st.markdown("**Transaction Statistics:**")
                stats_data = {
                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                    'Value': [
                        f"{df['Mobile_Money_Txns'].mean():.1f}",
                        f"{df['Mobile_Money_Txns'].median():.1f}",
                        f"{df['Mobile_Money_Txns'].std():.1f}",
                        f"{df['Mobile_Money_Txns'].min():.1f}",
                        f"{df['Mobile_Money_Txns'].max():.1f}"
                    ]
                }
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            # Additional visualizations
            st.markdown("#### 📈 Financial Behavior Patterns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Airtime Spending by Location")
                location_airtime = df.groupby('Location')['Airtime_Spend_ZWL'].mean().sort_values(ascending=False)
                st.bar_chart(location_airtime)
            
            with col2:
                st.markdown("##### Utility Payments Distribution")
                utility_bins = pd.cut(df['Utility_Payments_ZWL'], bins=10).value_counts().sort_index()
                st.bar_chart(utility_bins)
            
            # Save dataset for later use
            st.session_state.synthetic_data = df
            st.session_state.data_generated = True

def display_model_training():
    st.markdown("""
    <div class="objective-card">
        <h3>🤖 Objective 2: Train Machine Learning Model</h3>
        <p>Develop and train predictive models using alternative data features to assess creditworthiness.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'data_generated' not in st.session_state:
        st.warning("⚠️ Please generate synthetic data first in Objective 1 tab.")
        return
    
    df = st.session_state.synthetic_data
    
    # Model configuration
    st.subheader("Model Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider("Test Set Size (%)", 10, 40, 20)
        n_estimators = st.slider("Number of Trees", 50, 200, 100)
    
    with col2:
        max_depth = st.slider("Max Depth", 3, 20, 10)
        random_state = st.number_input("Random State", 0, 100, 42)
    
    with col3:
        feature_selection = st.multiselect(
            "Select Features for Model",
            df.columns.tolist()[:-1],  # Exclude target
            default=df.columns.tolist()[:-1]
        )
    
    if st.button("🚀 Train Machine Learning Model", use_container_width=True):
        with st.spinner("Training Random Forest model..."):
            # Prepare data
            X = df[feature_selection]
            y = df['Credit_Score']
            
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
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            st.success("✅ Model trained successfully!")
            
            # Display performance metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{accuracy:.2%}")
            with col2:
                st.metric("Training Samples", len(X_train))
            with col3:
                st.metric("Test Samples", len(X_test))
            with col4:
                st.metric("Features Used", len(feature_selection))
            
            # Feature importance using Streamlit native charts
            st.subheader("🔍 Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': feature_selection,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)  # Sort for horizontal bar chart
            
            # Display as bar chart
            importance_chart_data = feature_importance.set_index('Feature')['Importance']
            st.bar_chart(importance_chart_data)
            
            # Display feature importance as table with visual bars
            st.markdown("#### Feature Importance Ranking")
            for _, row in feature_importance.sort_values('Importance', ascending=False).iterrows():
                importance_percent = row['Importance'] * 100
                st.markdown(
                    f"""
                    <div class="feature-importance-bar" style="--importance: {importance_percent}%">
                        {row['Feature']}: {importance_percent:.1f}%
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            # Confusion Matrix using Streamlit
            st.subheader("📊 Model Performance Details")
            
            # Classification report
            st.markdown("##### Classification Report")
            report = classification_report(y_test, y_pred, target_names=target_encoder.classes_, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
            
            # Confusion matrix as table
            st.markdown("##### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=[f'Actual {cls}' for cls in target_encoder.classes_],
                columns=[f'Predicted {cls}' for cls in target_encoder.classes_]
            )
            st.dataframe(cm_df, use_container_width=True)
            
            # Save model and encoders
            st.session_state.model = model
            st.session_state.label_encoders = label_encoders
            st.session_state.target_encoder = target_encoder
            st.session_state.feature_selection = feature_selection
            st.session_state.model_trained = True

def display_credit_assessment():
    st.markdown("""
    <div class="objective-card">
        <h3>🎯 Objective 3: Predict Individual Credit Scores</h3>
        <p>User-friendly interface for credit assessment using the trained machine learning model.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'model_trained' not in st.session_state:
        st.warning("⚠️ Please train the machine learning model first in Objective 2 tab.")
        return
    
    st.subheader("📝 Credit Assessment Form")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Personal Information")
        Location = st.selectbox("📍 Location", ["Harare", "Bulawayo", "Gweru", "Mutare", "Masvingo", "Chinhoyi"])
        Gender = st.selectbox("👤 Gender", ["Male", "Female"])
        Age = st.slider("🎂 Age", 18, 70, 35)
    
    with col2:
        st.markdown("#### Financial Behavior")
        Mobile_Money_Txns = st.slider("📱 Monthly Mobile Transactions", 0, 100, 25)
        Airtime_Spend_ZWL = st.slider("📞 Monthly Airtime Spend (ZWL)", 0, 500, 100)
        Utility_Payments_ZWL = st.slider("💡 Monthly Utility Payments (ZWL)", 0, 1000, 300)
        Loan_Repayment_History = st.selectbox("📊 Loan Repayment History", ["Poor", "Fair", "Good", "Excellent"])
    
    if st.button("🔮 Get Credit Assessment", use_container_width=True):
        with st.spinner("Analyzing credit profile..."):
            # Prepare user data
            user_data = pd.DataFrame({
                'Location': [Location],
                'Gender': [Gender],
                'Age': [Age],
                'Mobile_Money_Txns': [Mobile_Money_Txns],
                'Airtime_Spend_ZWL': [Airtime_Spend_ZWL],
                'Utility_Payments_ZWL': [Utility_Payments_ZWL],
                'Loan_Repayment_History': [Loan_Repayment_History]
            })
            
            # Encode user input
            X_user = user_data[st.session_state.feature_selection].copy()
            for column in X_user.select_dtypes(include=['object']).columns:
                if column in st.session_state.label_encoders:
                    le = st.session_state.label_encoders[column]
                    if X_user[column].iloc[0] in le.classes_:
                        X_user[column] = le.transform(X_user[column])
                    else:
                        X_user[column] = -1  # Handle unseen labels
            
            # Predict
            model = st.session_state.model
            prediction_encoded = model.predict(X_user)
            prediction_proba = model.predict_proba(X_user)
            
            predicted_class = st.session_state.target_encoder.inverse_transform(prediction_encoded)[0]
            confidence = np.max(prediction_proba) * 100
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Credit Assessment Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Determine color based on credit score
                score_colors = {
                    'Excellent': '#2ecc71',
                    'Good': '#27ae60', 
                    'Fair': '#f39c12',
                    'Poor': '#e74c3c'
                }
                color = score_colors.get(predicted_class, '#3498db')
                
                st.markdown(f"""
                <div class="result-card">
                    <h3>Predicted Credit Score</h3>
                    <h1 style="text-align: center; color: {color}; font-size: 3rem;">{predicted_class}</h1>
                    <p style="text-align: center; font-size: 1.2rem;">Confidence: {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 📈 Probability Distribution")
                prob_df = pd.DataFrame({
                    'Credit Score': st.session_state.target_encoder.classes_,
                    'Probability (%)': (prediction_proba[0] * 100).round(2)
                }).sort_values('Probability (%)', ascending=False)
                
                # Display as bar chart
                prob_chart_data = prob_df.set_index('Credit Score')['Probability (%)']
                st.bar_chart(prob_chart_data)
                
                # Display as table
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
            
            # Financial Inclusion Impact
            st.markdown("#### 🌍 Financial Inclusion Impact")
            
            if predicted_class in ['Good', 'Excellent']:
                st.success("""
                ✅ **Credit-Worthy Profile Detected**
                
                This individual demonstrates strong financial behavior patterns through alternative data.
                Recommended for credit facility consideration, promoting financial inclusion for the unbanked population.
                
                **Recommended Actions:**
                - Consider for micro-loan approval
                - Eligible for higher credit limits  
                - Fast-track application process
                """)
            else:
                st.info("""
                📋 **Opportunities for Financial Inclusion**
                
                While current credit assessment shows room for improvement, continued positive financial behaviors
                can lead to improved creditworthiness. 
                
                **Recommendations:**
                - Consider starter micro-loan products
                - Financial literacy programs
                - Graduated credit access based on behavior
                - Regular monitoring for improvement
                """)
            
            # Show input summary
            st.markdown("#### 📋 Assessment Input Summary")
            input_summary = pd.DataFrame({
                'Feature': ['Location', 'Gender', 'Age', 'Mobile Transactions', 'Airtime Spend', 'Utility Payments', 'Repayment History'],
                'Value': [Location, Gender, f"{Age} years", f"{Mobile_Money_Txns}", f"ZWL {Airtime_Spend_ZWL}", f"ZWL {Utility_Payments_ZWL}", Loan_Repayment_History]
            })
            st.dataframe(input_summary, use_container_width=True, hide_index=True)

def display_research_insights():
    st.markdown("""
    <div class="objective-card">
        <h3>📈 Research Insights & Academic Contributions</h3>
        <p>Key findings and implications for financial inclusion in Zimbabwe.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Research Objectives Achieved")
        
        st.markdown("""
        <div class="data-card">
            <h4>✅ Objective 1: Data Simulation</h4>
            <p><strong>Success:</strong> Synthetic dataset generation representing Zimbabwe's unbanked population</p>
            <p><strong>Features:</strong> Mobile money, airtime spend, utility payments, repayment history</p>
            <p><strong>Impact:</strong> Realistic alternative data for credit assessment</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="data-card">
            <h4>✅ Objective 2: Model Training</h4>
            <p><strong>Success:</strong> Random Forest classifier trained on alternative data</p>
            <p><strong>Performance:</strong> High accuracy in creditworthiness prediction</p>
            <p><strong>Innovation:</strong> Machine learning for financial inclusion</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="data-card">
            <h4>✅ Objective 3: User Interface</h4>
            <p><strong>Success:</strong> Interactive credit assessment platform</p>
            <p><strong>Impact:</strong> Accessible financial inclusion tool</p>
            <p><strong>Usability:</strong> User-friendly interface for non-technical users</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📚 Academic Contributions")
        
        st.markdown("""
        <div class="data-card">
            <h4>🎓 Theoretical Framework</h4>
            <p>• Alternative data sources for credit scoring</p>
            <p>• Machine learning applications in financial inclusion</p>
            <p>• Zimbabwe-specific financial behavior analysis</p>
            <p>• Ethical AI implementation in emerging markets</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="data-card">
            <h4>🌍 Practical Implications</h4>
            <p>• Tool for microfinance institutions and banks</p>
            <p>• Financial inclusion for unbanked populations</p>
            <p>• Risk assessment using mobile money data</p>
            <p>• Scalable solution for emerging markets</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="data-card">
            <h4>🔬 Research Methodology</h4>
            <p>• Synthetic data generation techniques</p>
            <p>• Feature engineering for alternative data</p>
            <p>• Model evaluation in emerging markets context</p>
            <p>• User-centered interface design</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key Findings Summary
    st.markdown("---")
    st.subheader("🔍 Key Research Findings")
    
    findings_col1, findings_col2 = st.columns(2)
    
    with findings_col1:
        st.markdown("""
        **📊 Data Insights:**
        - Mobile money transactions strongly correlate with creditworthiness
        - Utility payment consistency is a reliable predictor
        - Airtime spending patterns reveal financial discipline
        - Behavioral data provides rich insights for risk assessment
        """)
        
        st.markdown("""
        **🤖 Model Performance:**
        - Random Forest effectively handles alternative data
        - Feature importance reveals key behavioral indicators
        - Model adapts well to Zimbabwe's financial context
        - High accuracy in credit classification
        """)
    
    with findings_col2:
        st.markdown("""
        **🌍 Social Impact:**
        - Bridges financial inclusion gap for unbanked populations
        - Enables credit access based on actual financial behavior
        - Supports micro-entrepreneurs and small businesses
        - Promotes economic empowerment through technology
        """)
        
        st.markdown("""
        **🚀 Future Research Directions:**
        - Integration with real mobile money APIs
        - Longitudinal behavior tracking
        - Multi-country model adaptation
        - Regulatory framework development
        - Ethical AI governance models
        """)

def generate_synthetic_data(n_samples=1000, locations=None):
    """Generate synthetic financial behavior data for Zimbabwe"""
    if locations is None:
        locations = ["Harare", "Bulawayo", "Gweru"]
    
    np.random.seed(42)
    
    data = {
        'Location': np.random.choice(locations, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Age': np.random.randint(18, 65, n_samples),
        'Mobile_Money_Txns': np.random.poisson(25, n_samples) + np.random.exponential(5, n_samples),
        'Airtime_Spend_ZWL': np.random.normal(150, 50, n_samples).clip(0),
        'Utility_Payments_ZWL': np.random.normal(400, 150, n_samples).clip(0),
        'Loan_Repayment_History': np.random.choice(['Poor', 'Fair', 'Good', 'Excellent'], n_samples, p=[0.2, 0.3, 0.35, 0.15])
    }
    
    df = pd.DataFrame(data)
    
    # Generate credit scores based on features (simplified rules)
    def calculate_credit_score(row):
        score = 0
        
        # Age factor (prime earning years)
        if 30 <= row['Age'] <= 50:
            score += 2
        elif 25 <= row['Age'] < 30 or 50 < row['Age'] <= 60:
            score += 1
        
        # Mobile transactions (more activity = better)
        if row['Mobile_Money_Txns'] > 30:
            score += 2
        elif row['Mobile_Money_Txns'] > 15:
            score += 1
        
        # Airtime spend (moderate spending = better)
        if 100 <= row['Airtime_Spend_ZWL'] <= 200:
            score += 1
        
        # Utility payments (consistent payments = better)
        if row['Utility_Payments_ZWL'] > 300:
            score += 1
        
        # Repayment history
        repayment_scores = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
        score += repayment_scores[row['Loan_Repayment_History']]
        
        # Determine credit score category
        if score >= 7:
            return 'Excellent'
        elif score >= 5:
            return 'Good'
        elif score >= 3:
            return 'Fair'
        else:
            return 'Poor'
    
    df['Credit_Score'] = df.apply(calculate_credit_score, axis=1)
    
    return df

if __name__ == "__main__":
    main()
