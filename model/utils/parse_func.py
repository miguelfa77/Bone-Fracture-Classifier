import os
import tensorflow as tf
# Decodes PNG, resizes Image and normalizes pixels from 0-1.
def parse_known(filename):
    parts = tf.strings.split(filename, os.sep)
    label = tf.cond(tf.strings.regex_full_match(parts[-2], 'fractured'),
                    lambda: 1,
                    lambda: 0)
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [128, 128])
    image = image / 255.0
    return image, tf.cast(label, tf.int32)

def parse_unknown(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [128, 128])
    image = image / 255.0
    return image