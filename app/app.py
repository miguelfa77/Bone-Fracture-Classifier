import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

from model.utils.parse_func import parse_streamlit

st.title('Fracture Image Classifier')

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('/Users/miguelfa/Desktop/Bone-Fracture-Classifier/app/model/CNN-FractureImageClassifier.keras')
    return model

# Classify based on image and model
@st.cache(allow_output_mutation=True)
def model_predict(model, image):       
    try:
        # predict 0 or 1 ('not fracture' or 'fracture')
        prediction = model.predict(image)
        prediction_class = np.where(prediction > 0.5, 'fractured', 'not fractured')
        return prediction, prediction_class
    except Exception as e:
        print(f'Failed to make predictions: {e}')

model = load_model()

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption=f'Uploaded Image: {uploaded_file}', use_column_width=True)

    # Parse the image
    image = parse_streamlit(uploaded_file)

    # Make prediction
    prediction, prediction_class = model_predict(model,image)

    # Display prediction
    st.write(f'Prediction: {prediction}')
    st.write(f'Prediction Class: {prediction_class}')


