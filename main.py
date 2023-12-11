import tensorflow as tf
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten, BatchNormalization
from keras.preprocessing.image import array_to_img, load_img, img_to_array
import streamlit as st
from keras.saving import load_model

loaded_model = load_model("my_model.keras")

img_classes = [
    "aeroplane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


def preprocess_img(pic):
    img = load_img(pic, target_size=(32, 32))
    array = img_to_array(img)
    expanded = np.expand_dims(array, axis=0)
    return expanded


def load_app():
    st.header("Omari's Image Classification App")
    st.markdown("###### Please upload an image of a aeroplane, car, bird, cat, deer, dog, frog, horse, ship or a truck")
    uploaded_img = st.file_uploader("Upload image here (Max size 200MB)")
    if st.button("Classify Image"):
        processed_image = preprocess_img(uploaded_img)
        prediction = loaded_model.predict(processed_image)
        arg = np.argmax(prediction)
        st.write(f"Image is of a {img_classes[arg]}")
        st.write(f"Prediction probability: {prediction[0][arg]:.2%}")


load_app()
