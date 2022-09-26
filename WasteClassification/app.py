import streamlit as st
import tensorflow as tf
import streamlit as st
from tensorflow.keras import models
from keras.models import load_model

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('/content/drive/MyDrive/combined_data/MobileNetV2_trashbox_combined.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Waste Classification
         """
         )

file = st.file_uploader("Please upload file", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)

def import_and_predict(image_data, model):
    
        size = (224,224)    
        
        class_names = ['carton', 'glassbottle', 'milkbox', 'napkin', 'paper', 'plasticbottle']

        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        predicted_value = class_names[np.argmax(prediction)]
        predicted_accuracy = round(np.max(prediction) * 100, 2)

        
        return predicted_value, predicted_accuracy
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    
    predictions = import_and_predict(image, model)
    
    st.write(predictions[0])
    st.write(predictions[1])
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(predictions[0], predictions[1])
)
