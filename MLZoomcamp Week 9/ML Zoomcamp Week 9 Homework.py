#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.lite as tflite
from io import BytesIO
from urllib import request
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


print(f'Tensorflow version: {tf.__version__}')

model = keras.models.load_model('dogs_cats_10_0.687.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('dogs_cats-model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)

interpreter = tflite.Interpreter(model_path='dogs_cats-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']
print(f'Input Index: {input_index}')
print(f'Output Index: {output_index}')

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

img = download_image('https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg')
img = prepare_image(img, (150,150))

train_datagen = ImageDataGenerator(rescale=1./255)

x = np.array(img)
X = np.array([x])
X = train_datagen.flow(X)

print(f'First pixel: {X[0][0][0][0]}') 

print(f'Prediction: {model.predict(X)[0][0]}')

