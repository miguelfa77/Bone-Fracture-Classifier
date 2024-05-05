import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from model.utils.parse_func import parse_streamlit

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('/Users/miguelfa/Desktop/Bone-Fracture-Classifier/app/model/CNN-FractureImageClassifier.keras')
    model_history = pd.read_csv('training.log',sep=',',engine='python')
    return model, model_history

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

# Plot training and validation losses  
def plot(model_history):   
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    # Plot training and validation losses
    model_history[['loss', 'val_loss']].plot(ax=axes[0])
    axes[0].set_title('Training and Validation Losses')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(['Training Loss', 'Validation Loss'])

    # Plot training and validation accuracies
    model_history[['binary_accuracy', 'val_binary_accuracy']].plot(ax=axes[1])
    axes[1].set_title('Training and Validation Accuracies')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend(['Training Accuracy', 'Validation Accuracy'])

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    return plt
    
# Load the model before anything
model, model_history = load_model()
# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Upload Image", "Plots"])


if page == "Upload Image":
    st.title('Fracture Image Classifier')
    # Upload image and perform prediction
    st.write("Upload image and perform prediction here... Allowed Types: [JPG, JPEG, PNG]")
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


elif page == "Plots":
    st.title('Fracture Image Classifier')
    # Show plots
    st.write("Displaying plots...")
    plt = plot(model_history=model_history)
    st.pyplot(plt)


