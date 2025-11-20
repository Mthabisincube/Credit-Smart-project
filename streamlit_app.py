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

with st.expander('Data Visualisations'):
    st.write("Scatter plot of Location vs Credit Score")
    st.scatter_chart(df[['Location', 'Credit_Score']])
         





