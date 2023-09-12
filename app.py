import streamlit as st
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image  # Import the Image module from PIL

# Define a custom object scope to register the KerasLayer
custom_object_scope = tf.keras.utils.custom_object_scope({'KerasLayer': hub.KerasLayer})

# Load your trained model within the custom object scope
with custom_object_scope:
    model = tf.keras.models.load_model('model.h5')

# Streamlit app code
st.title('Mask Detection App')

# File uploader to upload an image for prediction
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image for prediction
    img = image.resize((64, 64))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Make predictions
    prediction = model.predict(img)
    max_index = np.argmax(prediction)

    # Define class labels
    class_labels = ["With mask", "Without mask"]

    # Display the prediction result
    st.write(f"Prediction: {class_labels[max_index]} (Probability: {prediction[0][max_index]:.2f})")
