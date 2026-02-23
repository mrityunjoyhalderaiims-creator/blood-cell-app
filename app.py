import streamlit as st
import numpy as np
from PIL import Image
import keras
from keras.layers import DepthwiseConv2D as _DepthwiseConv2D

class FixedDepthwiseConv2D(_DepthwiseConv2D):
    @classmethod
    def from_config(cls, config):
        config.pop('groups', None)
        return super().from_config(config)

@st.cache_resource
def load_model():
    model = keras.models.load_model(
        'keras_model.h5',
        custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D},
        compile=False
    )
    return model

@st.cache_resource
def load_labels():
    with open('labels.txt', 'r') as f:
        return [line.strip().split(' ', 1)[1] for line in f.readlines()]

model       = load_model()
class_names = load_labels()
IMG_SIZE    = (model.input_shape[1], model.input_shape[2])

st.title("Blood Cell Classifier")
st.write("Upload a blood cell image to identify its type.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', width=250)

    img_resized = image.resize(IMG_SIZE)
    img_array   = np.array(img_resized) / 255.0
    img_array   = np.expand_dims(img_array, axis=0).astype(np.float32)

    with st.spinner('Classifying...'):
        predictions   = model.predict(img_array)
        predicted_idx = np.argmax(predictions[0])
        confidence    = predictions[0][predicted_idx] * 100

    st.success(f"**Prediction:** {class_names[predicted_idx]}")
    st.info(f"**Confidence:** {confidence:.2f}%")