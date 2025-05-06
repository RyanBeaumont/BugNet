import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
model = tf.keras.models.load_model('BugNet.keras')
with open('insect_descriptions.json', 'r') as f:
    insect_descriptions = json.load(f)

st.title("BugNet - Insect Family Classifier")
st.markdown(
    """
    <style>
    body {
        background-color: #bae4d0;
    }
    </style>
    """,
    unsafe_allow_html=True
)
uploaded_file = st.file_uploader("Choose an image...", type="jpg")


if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)

    # Convert the image to grayscale
    image = image.convert('L')  # 'L' mode stands for grayscale

    # Resize the image
    image = image.resize((64, 64))  # Preprocess as needed

    # Convert the image to a numpy array and normalize
    image_array = np.array(image) / 255.0  # Normalize if required

    # Ensure the image has the correct shape for the model (add batch dimension)
    image_array = image_array.reshape((1, 64, 64, 1))  # Assuming model expects (64, 64, 1) shape

    # Make a prediction (logits)
    logits = model.predict(image_array)

    # Get the top 3 predictions and their probabilities
    top_3_indices = np.argsort(logits[0])[::-1][:3]  # Sort and get the indices of the top 3
    top_3_probabilities = logits[0][top_3_indices]

    # Assuming you have a list of class labels, for example:
    class_labels = class_labels = sorted(insect_descriptions.keys())

    # Display the top 3 predicted class names and their probabilities
    st.write("Top 3 Predictions:")
    for i in range(3):
        predicted_class_name = class_labels[top_3_indices[i]]
        probability = top_3_probabilities[i]
        common_name = insect_descriptions[predicted_class_name]["common_name"]
        description = insect_descriptions[predicted_class_name]["description"]
        
        st.markdown(f"<h3 style='color:green;'>{common_name} - Family: {predicted_class_name} ({round(probability*100)}%)</h3>", unsafe_allow_html=True)
        st.markdown(f"<i>{description}</i>", unsafe_allow_html=True)


