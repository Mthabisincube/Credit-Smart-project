import streamlit as st
import pandas as pd 

st.title('Smart Credit App')
st.write('Taking Credit Scoring to the next level')

df = pd.read_csv("https://raw.githubusercontent.com/Mthabisincube/Credit-Smart-project/refs/heads/master/smart_credit_scoring_zimbabwe.csv")

with st.expander('Data'):
    st.write('**Raw Data**')
    st.write(df)

    st.write('**X**')
    X = df.drop("Credit_Score", axis=1)
    st.write(X)

    st.write('**Y**')
    Y = df.Credit_Score
    st.write(Y)

# Data preparation
with st.sidebar:
    st.header ('**Alternative Data**') 
    Location = st.selectbox(
    "Location",
    ('Bulawayo', 'Chiredzi', 'Masvingo', 'Chinhoyi', 'Mutare', 'Harare', 'Gweru', 'Gokwe'))
    gender = st.selectbox("Gender", ("Male","Female"))
    Mobile_Money_Txns = st.slider("Mobile Money Transactions", 20.0, 299.0, 159.97)
    Airtime_Spend_ZWL = st.slider("Airtime_Spend_ZWL", 100.0,999.0, 552.6)
    Utility_Payments_ZWL = st.slider ("Utility_Payments_ZWL",202.0,1499.0, 870.058)
    Loan_Repayment_History = st.selectbox("Loan_Repayment_History", ('Poor', 'Fair','Good','Excellent'))
    Age = st.slider ("Age",18,64, 50)
    
                                          












    
         





