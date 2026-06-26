import streamlit as st
import pickle
import numpy as np
import pandas as pd

loaded_model = pickle.load(open('wine_class_predictor.pkl', 'rb'))

def predictor(input_data):
	input_data_as_numpy_array = np.asarray(input_data)
	reshaped_numpy_array = input_data_as_numpy_array.reshape(1,-1)
	
	prediction = loaded_model.predict(reshaped_numpy_array)
	return f'Wine Class {prediction[0]}'

def main():
	st.title('Wine Class Prediction')
	
	proline = st.text_input('Proline')
	flavanoids = st.text_input('Flavanoids')
	
	prediction = ''
	if st.button('Wine Class Result'):
		prediction = predictor([float(proline), float(flavanoids)])
		st.success(prediction)

if __name__=="__main__":
	main()