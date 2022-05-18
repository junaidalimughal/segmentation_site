import tensorflow as tf
import numpy as np
import os
import time

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

BATCH_SIZE = 2
BUFFER_SIZE = 1000


def parse_function_test_images(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    image = tf.image.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    #image = tf.expand_dims(image, axis=0)

    return image

def prepare_dataset(paths):
    test_ds = tf.data.Dataset.from_tensor_slices(paths)
    test_dataset = test_ds.map(parse_function_test_images)

    return test_dataset

def test_normalize(image):
    image = tf.cast(image, tf.float32) / 255.0
    return image
    
def prepare_batches(paths):
    test_dataset = prepare_dataset(paths)
    test_images = test_dataset.map(test_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    test_batches = (
        test_images
            .cache()
            #.shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return test_batches
    
def denormalize(output_mask, threshold):
    output_mask = output_mask.reshape((512, 512))
    output_mask *= 255.0
    output_mask = np.where(output_mask > threshold, 255, 0)
    
    return output_mask



