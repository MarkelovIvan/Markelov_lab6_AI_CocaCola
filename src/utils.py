# utils.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kp_image

def save_model_checkpoint(model, path):
    model.save(path)

def load_img(path, target_size):
    img = kp_image.load_img(path, target_size=target_size)
    arr = kp_image.img_to_array(img)
    arr = np.expand_dims(arr, 0)
    return arr

def preprocess_for_model(x, input_shape):
    x = tf.image.resize(x, input_shape[:2])
    x = tf.cast(x, tf.float32) / 255.0
    return x
