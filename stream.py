import streamlit as st
import joblib
import pandas as pd 


model=joblib.load('models/ml_model.pkl')
preprocessor=joblib.load('preprocesser/preprocessor.pkl')

st.title('Car Price Prediction')

df=pd.read_csv('cars.csv')


brand= st.selectbox('brand',df['brand'].unique())
km_driven= st.number_input('km_driven')
fuel= st.selectbox('fuel',df['fuel'].unique())
owner= st.selectbox('owner',df['owner'].unique())

df=pd.DataFrame({
    'brand':[brand],
    'km_driven':[km_driven],
    'fuel':[fuel],
    'owner':[owner]
},index=[0])

if  st.button('Predict'):

    df_preprocessed=preprocessor.transform(df) 
    prediction=model.predict(df_preprocessed)
    st.write(f'Predicted Price: {prediction[0]}')