import streamlit as st
import pandas as pd

st.title('Smart Credit App')
st.write('Taking Credit Scoring to the next level')

# Load data
df = pd.read_csv("https://raw.githubusercontent.com/Mthabisincube/Credit-Smart-project/refs/heads/master/smart_credit_scoring_zimbabwe.csv")

# Display raw data and features
with st.expander('Data'):
    st.write('**Raw Data**')
    st.dataframe(df)

    st.write('**X (Features)**')
    X = df.drop("Credit_Score", axis=1)
    st.dataframe(X)

    st.write('**Y (Target)**')
    Y = df["Credit_Score"]
    st.dataframe(Y)

# Sidebar inputs
with st.sidebar:
    st.header('**Alternative Data**')
    Location = st.selectbox("Location", ('Bulawayo', 'Chiredzi', 'Masvingo', 'Chinhoyi', 'Mutare', 'Harare', 'Gweru', 'Gokwe'))
    gender = st.selectbox("Gender", ("Male", "Female"))
    Mobile_Money_Txns = st.slider("Mobile Money Transactions", 20.0, 299.0, 159.97)
    Airtime_Spend_ZWL = st.slider("Airtime Spend (ZWL)", 100.0, 999.0, 552.6)
    Utility_Payments_ZWL = st.slider("Utility Payments (ZWL)", 202.0, 1499.0, 870.058)
    Loan_Repayment_History = st.selectbox("Loan Repayment History", ('Poor', 'Fair', 'Good', 'Excellent'))
    Age = st.slider("Age", 18, 64, 50)

# Create summary string
# Create a DataFrame for input summary
input_summary_df = pd.DataFrame({
    "Feature": ["Location", "Gender", "Mobile Money Transactions", "Airtime Spend (ZWL)",
                "Utility Payments (ZWL)", "Loan Repayment History", "Age"],
    "Value": [Location, gender, Mobile_Money_Txns, Airtime_Spend_ZWL,
              Utility_Payments_ZWL, Loan_Repayment_History, Age]
})

# Display in dropdown-style expander with columns
with st.expander("***Summary of Your Inputs***"):
    st.write("** Summary**")
    st.dataframe(input_summary_df, use_container_width=True)

# Display summary



# Create input
