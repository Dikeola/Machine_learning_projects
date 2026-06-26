import streamlit as st
import pickle
import numpy as np
import pandas as pd

loaded_model = pickle.load(open('California_house_price_model.pkl','rb'))

def predictor(input_data):
    input_data_as_array = np.asarray(input_data)
    reshaped_array = input_data_as_array.reshape(1,-1)
    
    prediction = loaded_model.predict(reshaped_array)
    return f"House Price is {prediction}"

def main():
    st.title('California House Price Predictor')
    
    medinc = st.text_input('MedInc')
    
    prediction = ''
    if st.button('Result'):
        prediction = predictor([float(medinc)])
    st.success(prediction)
    
if __name__=="__main__":
    main()