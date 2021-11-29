#!/usr/bin/env python
# coding: utf-8

import tflite_runtime.interpreter as tflite
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from io import BytesIO
from urllib import request
import numpy as np

from PIL import Image


interpreter = tflite.Interpreter(model_path='cats-dogs-v2.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# train_datagen = ImageDataGenerator(rescale=1./255)

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

def predict(url):
    image = download_image(url)
    img = prepare_image(image, (150,150))
    x = np.array(img)
    x = x/255.0
    X = np.array([x], dtype=np.float32)
    # X = train_datagen.flow(X)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)
    predictions = pred[0][0].tolist()
    return predictions

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result

