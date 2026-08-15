import streamlit as st
import pandas as pd
import Treatment as Treatment
import numpy as np
import datetime 
st.set_page_config(page_title="3IDSD MLOps project")

st.title("3IDSD MLOps project")

amt = st.number_input('Amount:')
zip = st.number_input('zip:')
lat = st.number_input('lat:')
long = st.number_input('long:')
merch_lat = st.number_input('merch_lat:')
merch_long=st.number_input('merch_long:')
age=st.number_input('age')
gender=st.selectbox(
    'Gender:',('Men','Women')
)
if gender=='Men':
    gender=0
else:
    gender=1

date= st.date_input(

    "Transaction date",datetime.datetime(2019, 1, 1, 9, 30))

time = st.time_input('Hours,minutes,secondes', datetime.time(8, 45,0))

datetimeTransaction = datetime.datetime.combine(date,time)
category = st.selectbox(
    'Which Merchant category?',
    (
'category_entertainment',
'category_food_dining',
'category_gas_transport',
'category_grocery_net',
'category_grocery_pos',
'category_health_fitness',
'category_home',
'category_kids_pets',
'category_misc_net',
'category_misc_pos',
'category_personal_care',
'category_shopping_net',
'category_shopping_pos',
'category_travel',
    ))
#st.write(datetimeTransaction)
#button prediction
if st.button('Predict'):
     with st.spinner("Please wait to predict"):
        res=Treatment.getPredict(amt,zip,lat,long,merch_lat,merch_long,age,gender,category,datetimeTransaction)
        result=res.json()
        if(result['predictions'][0]==0):
            st.write("Non Fraud")
        else:
            st.write("Fraud")
