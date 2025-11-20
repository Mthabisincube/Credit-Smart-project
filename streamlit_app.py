import streamlit as st
import pandas as pd 

st.title('Smart Credit App')
st.write('Taking Credit Scoring to the next level')

with st.expander('Data'):
  st.write ('Raw Data')

df = pd.read_csv("https://raw.githubusercontent.com/Mthabisincube/Credit-Smart-project/refs/heads/master/smart_credit_scoring_zimbabwe.csv")
df
