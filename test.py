import os
import string
import random
import json
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model(r"C:\Users\USER\Desktop\Project Laser\test app\pets")

CLASSES = ['Cat', 'Dog']
image_path = r"C:\Users\USER\Pictures\cat.jpg"
SIZE = 128
def get_prediction(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(SIZE, SIZE))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0]
    class_name = CLASSES[int(prediction > 0.5)]
    return class_name

print(get_prediction(image_path))