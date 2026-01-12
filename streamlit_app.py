import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

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
    
    .algorithm-card {
        background-color: rgba(248, 249, 250, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        backdrop-filter: blur(5px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .algorithm-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }
    
    .algorithm-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .best-algorithm {
        border: 3px solid #28a745;
        background: linear-gradient(135deg, rgba(212, 237, 218, 0.95) 0%, rgba(195, 230, 203, 0.95) 100%);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(40, 167, 69, 0); }
        100% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0); }
    }
    
    .metric-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .comparison-chart {
        background-color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<h1 class="main-header glowing-text">🏦 Zim Smart Credit App</h1>', unsafe_allow_html=True)
st.markdown("### 💳 Advanced ML Algorithms for Credit Scoring in Zimbabwe")
st.markdown("---")

# Load data with caching
@st.cache_data
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/Mthabisincube/Credit-Smart-project/refs/heads/master/smart_credit_scoring_zimbabwe.csv")

df = load_data()

# Sidebar for user input
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
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔍 Analysis", "🎯 Assessment", "🤖 ML Algorithms"])

# ... (Previous tabs 1, 2, and 3 remain the same as in your original code) ...

with tab4:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("### 🤖 Advanced Machine Learning Algorithms")
    
    st.markdown("""
    <div class="card">
        <h3>🚀 Multiple Algorithm Comparison</h3>
        <p>We train and compare multiple machine learning algorithms to find the best model for credit scoring. 
        Each algorithm has its strengths and is evaluated on accuracy, precision, recall, and F1-score.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Algorithm selection
    st.markdown("#### 🎯 Select Algorithms to Compare")
    
    algorithms_to_train = st.multiselect(
        "Choose algorithms:",
        [
            "Random Forest",
            "Gradient Boosting", 
            "Decision Tree",
            "Logistic Regression",
            "Support Vector Machine",
            "K-Nearest Neighbors",
            "AdaBoost",
            "Naive Bayes"
        ],
        default=["Random Forest", "Gradient Boosting", "Logistic Regression", "Decision Tree"]
    )
    
    if st.button("🎯 Train All Selected Algorithms", type="primary", use_container_width=True):
        with st.spinner("🤖 Training multiple algorithms... This may take a few moments."):
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
                
                # Standardize features for some algorithms
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Initialize algorithms
                algorithms = {}
                
                if "Random Forest" in algorithms_to_train:
                    algorithms["Random Forest"] = RandomForestClassifier(
                        n_estimators=100, 
                        random_state=42,
                        max_depth=10,
                        min_samples_split=5
                    )
                
                if "Gradient Boosting" in algorithms_to_train:
                    algorithms["Gradient Boosting"] = GradientBoostingClassifier(
                        n_estimators=100, 
                        random_state=42,
                        learning_rate=0.1
                    )
                
                if "Decision Tree" in algorithms_to_train:
                    algorithms["Decision Tree"] = DecisionTreeClassifier(
                        random_state=42,
                        max_depth=5
                    )
                
                if "Logistic Regression" in algorithms_to_train:
                    algorithms["Logistic Regression"] = LogisticRegression(
                        random_state=42,
                        max_iter=1000
                    )
                
                if "Support Vector Machine" in algorithms_to_train:
                    algorithms["Support Vector Machine"] = SVC(
                        random_state=42,
                        probability=True
                    )
                
                if "K-Nearest Neighbors" in algorithms_to_train:
                    algorithms["K-Nearest Neighbors"] = KNeighborsClassifier(
                        n_neighbors=5
                    )
                
                if "AdaBoost" in algorithms_to_train:
                    algorithms["AdaBoost"] = AdaBoostClassifier(
                        random_state=42,
                        n_estimators=50
                    )
                
                if "Naive Bayes" in algorithms_to_train:
                    algorithms["Naive Bayes"] = GaussianNB()
                
                # Train and evaluate algorithms
                results = []
                feature_importances = {}
                
                progress_bar = st.progress(0)
                for idx, (name, model) in enumerate(algorithms.items()):
                    # Update progress
                    progress_bar.progress((idx + 1) / len(algorithms))
                    
                    with st.spinner(f"Training {name}..."):
                        # Use scaled data for certain algorithms
                        if name in ["Logistic Regression", "Support Vector Machine", "K-Nearest Neighbors"]:
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)
                            y_pred_proba = model.predict_proba(X_test_scaled)
                        else:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            y_pred_proba = model.predict_proba(X_test)
                        
                        # Calculate metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        report = classification_report(y_test, y_pred, output_dict=True)
                        
                        # Store feature importance if available
                        if hasattr(model, 'feature_importances_'):
                            feature_importances[name] = model.feature_importances_
                        
                        # Cross-validation score
                        cv_scores = cross_val_score(model, X, y_encoded, cv=5, scoring='accuracy')
                        
                        results.append({
                            'Algorithm': name,
                            'Accuracy': accuracy,
                            'Precision (Weighted)': report['weighted avg']['precision'],
                            'Recall (Weighted)': report['weighted avg']['recall'],
                            'F1-Score (Weighted)': report['weighted avg']['f1-score'],
                            'CV Mean Accuracy': cv_scores.mean(),
                            'CV Std': cv_scores.std(),
                            'Training Time': datetime.now()  # Placeholder for timing
                        })
                
                progress_bar.empty()
                
                # Display results
                st.success(f"✅ {len(results)} algorithms trained successfully!")
                
                # Convert results to DataFrame
                results_df = pd.DataFrame(results)
                
                # Find best algorithm
                best_algorithm = results_df.loc[results_df['Accuracy'].idxmax()]
                
                st.markdown(f"""
                <div class="algorithm-header">
                    <h3>🏆 Best Performing Algorithm</h3>
                    <h2>{best_algorithm['Algorithm']}</h2>
                    <p>Accuracy: {best_algorithm['Accuracy']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display comparison metrics
                st.markdown("#### 📊 Algorithm Performance Comparison")
                
                # Interactive metrics chart
                metrics_to_show = st.multiselect(
                    "Select metrics to display:",
                    ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)', 'CV Mean Accuracy'],
                    default=['Accuracy', 'F1-Score (Weighted)']
                )
                
                if metrics_to_show:
                    # Create comparison chart
                    fig = go.Figure()
                    
                    colors = px.colors.qualitative.Set3
                    for i, metric in enumerate(metrics_to_show):
                        fig.add_trace(go.Bar(
                            name=metric,
                            x=results_df['Algorithm'],
                            y=results_df[metric],
                            marker_color=colors[i % len(colors)],
                            text=results_df[metric].apply(lambda x: f'{x:.2%}' if 'Accuracy' in metric else f'{x:.3f}'),
                            textposition='auto',
                        ))
                    
                    fig.update_layout(
                        title='Algorithm Performance Comparison',
                        xaxis_title='Algorithm',
                        yaxis_title='Score',
                        barmode='group',
                        template='plotly_white',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display detailed results table
                st.markdown("#### 📋 Detailed Performance Metrics")
                
                # Format the results for display
                display_df = results_df.copy()
                for col in ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 
                           'F1-Score (Weighted)', 'CV Mean Accuracy']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f'{x:.2%}')
                display_df['CV Std'] = display_df['CV Std'].apply(lambda x: f'{x:.4f}')
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Feature Importance Analysis
                if feature_importances:
                    st.markdown("#### 🔍 Feature Importance Analysis")
                    
                    # Show feature importance for Random Forest if available
                    if "Random Forest" in feature_importances:
                        rf_importance = feature_importances["Random Forest"]
                        feature_importance_df = pd.DataFrame({
                            'Feature': X.columns,
                            'Importance': rf_importance
                        }).sort_values('Importance', ascending=False)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Feature importance chart
                            fig_imp = go.Figure(go.Bar(
                                x=feature_importance_df['Importance'],
                                y=feature_importance_df['Feature'],
                                orientation='h',
                                marker_color='#667eea'
                            ))
                            
                            fig_imp.update_layout(
                                title='Random Forest Feature Importance',
                                xaxis_title='Importance',
                                yaxis_title='Feature',
                                height=400,
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig_imp, use_container_width=True)
                        
                        with col2:
                            st.dataframe(feature_importance_df, use_container_width=True, hide_index=True)
                
                # Individual Algorithm Details
                st.markdown("#### 📚 Algorithm Details & Predictions")
                
                for result in results:
                    with st.expander(f"🔍 {result['Algorithm']} Details (Accuracy: {result['Accuracy']:.2%})", expanded=result['Algorithm'] == best_algorithm['Algorithm']):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Accuracy", f"{result['Accuracy']:.2%}")
                        with col2:
                            st.metric("Precision", f"{result['Precision (Weighted)']:.2%}")
                        with col3:
                            st.metric("Recall", f"{result['Recall (Weighted)']:.2%}")
                        
                        # Make prediction with this algorithm
                        if st.button(f"🎯 Predict with {result['Algorithm']}", key=f"predict_{result['Algorithm']}"):
                            # Get the trained model
                            model = algorithms[result['Algorithm']]
                            
                            # Prepare user data
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
                            
                            # Scale data if needed
                            if result['Algorithm'] in ["Logistic Regression", "Support Vector Machine", "K-Nearest Neighbors"]:
                                user_data_scaled = scaler.transform(user_data)
                                prediction_encoded = model.predict(user_data_scaled)
                                prediction_proba = model.predict_proba(user_data_scaled)
                            else:
                                prediction_encoded = model.predict(user_data)
                                prediction_proba = model.predict_proba(user_data)
                            
                            predicted_class = target_encoder.inverse_transform(prediction_encoded)[0]
                            confidence = np.max(prediction_proba) * 100
                            
                            # Display prediction
                            st.markdown(f"""
                            <div class="{'best-algorithm' if result['Algorithm'] == best_algorithm['Algorithm'] else 'algorithm-card'}">
                                <h4>🎯 {result['Algorithm']} Prediction</h4>
                                <h2 style="color: #1f77b4;">{predicted_class}</h2>
                                <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                                <p><strong>Algorithm:</strong> {result['Algorithm']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show probability distribution
                            prob_df = pd.DataFrame({
                                'Credit Score': target_encoder.classes_,
                                'Probability (%)': (prediction_proba[0] * 100).round(2)
                            }).sort_values('Probability (%)', ascending=False)
                            
                            st.dataframe(prob_df, use_container_width=True, hide_index=True)
                
                # Ensemble Prediction
                st.markdown("#### 🏗️ Ensemble Prediction (Voting)")
                
                if st.button("🤝 Get Ensemble Prediction", type="secondary", use_container_width=True):
                    predictions = []
                    confidences = []
                    
                    for name, model in algorithms.items():
                        # Prepare user data
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
                        
                        # Make prediction
                        if name in ["Logistic Regression", "Support Vector Machine", "K-Nearest Neighbors"]:
                            user_data_scaled = scaler.transform(user_data)
                            pred = model.predict(user_data_scaled)[0]
                        else:
                            pred = model.predict(user_data)[0]
                        
                        predictions.append(pred)
                        confidences.append(1.0)  # Equal weight for now
                    
                    # Majority voting
                    ensemble_pred = max(set(predictions), key=predictions.count)
                    ensemble_class = target_encoder.inverse_transform([ensemble_pred])[0]
                    
                    st.markdown(f"""
                    <div class="best-algorithm">
                        <h4>🏆 Ensemble Prediction (Majority Voting)</h4>
                        <h1 style="color: #28a745;">{ensemble_class}</h1>
                        <p><strong>Based on:</strong> {len(algorithms)} algorithms</p>
                        <p><strong>Consensus:</strong> {predictions.count(ensemble_pred)} out of {len(predictions)} algorithms agree</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show individual algorithm votes
                    vote_df = pd.DataFrame({
                        'Algorithm': list(algorithms.keys()),
                        'Prediction': [target_encoder.inverse_transform([p])[0] for p in predictions],
                        'Vote': ['✅' if p == ensemble_pred else '❌' for p in predictions]
                    })
                    
                    st.dataframe(vote_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"❌ Error training algorithms: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
