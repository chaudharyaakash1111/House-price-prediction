import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved machine learning model
model = pickle.load(open('model.pkl', 'rb'))

# Define the function to predict the price
def predict_price(location, sqft, bhk, bath):
    """Predicts the price of the house using the input features"""
    location_index = np.where(X.columns == location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if location_index >= 0:
        x[location_index] = 1

    return model.predict([x])[0]

# Load the dataframe used for training the model
X = pd.read_csv('X.csv')

# Create the Streamlit app
def main():
    st.title('House Price Prediction')

    # Create the input features
    location = st.selectbox('Location', X.columns[3:])
    sqft = st.number_input('Square feet', min_value=100, max_value=10000, value=1000)
    bhk = st.number_input('BHK', min_value=1, max_value=10, value=2)
    bath = st.number_input('Bathrooms', min_value=1, max_value=10, value=2)

    # Predict the price using the input features
    price = predict_price(location, sqft, bhk, bath)

    # Display the predicted price
    st.subheader('Predicted Price')
    st.write(f'${price:,.2f}')

if __name__ == '__main__':
    main()
