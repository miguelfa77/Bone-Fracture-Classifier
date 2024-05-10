import os
import tensorflow as tf
import numpy as np
from PIL import Image
# Decodes PNG, resizes Image and normalizes pixels from 0-1.
def parse_known(filename, shape=[128, 128]):
    parts = tf.strings.split(filename, os.sep)
    label = tf.cond(tf.strings.regex_full_match(parts[-2], 'fractured'),
                    lambda: 1,
                    lambda: 0)
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, shape)
    image = image / 255.0
    return image, tf.cast(label, tf.int32)

def parse_unknown(filename, shape=[128, 128]):
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=3)
    image = np.array(image)
    image = tf.image.resize(image, shape)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def parse_streamlit(filename, shape=[128, 128]):
    image = Image.open(filename)
    image = image.convert('RGB')
    image = np.array(image)
    image = tf.image.resize(image, shape)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image