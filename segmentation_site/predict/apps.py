import imp
from django.apps import AppConfig
import tensorflow as tf

def load_model():

    # define the input layers, with dimensions 512 x 512 x 3
    inputs = tf.keras.layers.Input((512, 512, 3))

    # Extraction Process layer_1
    # 1. first convolution operation to identify 16 fetures, with 3 x 3 filter size. 'relu' activation function
    # will be used, because it works better than the others
    # 2. Dropout layer will skip the neurons from the layers. 0.1 means 10% of neurons will be skipped
    # from training process at each epoch.
    # 3. Another convolution operation will identify another 16 features
    # 4. MaxPooling2D will reduce the images resolution to half by applying 2 x 2 filter.
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer="he_normal", padding="same")(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer="he_normal", padding="same")(c1)

    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    # Extraction Provess layer_2
    # the same process will be followed as layer 1.
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)

    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    # Extraction Provess layer_3
    # the same process will be followed as layer 1.
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)

    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    # Extraction Provess layer_3
    # the same process will be followed as layer 1.
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)

    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

    # Extraction Provess layer_4
    # the same process will be followed as layer 1 except the MaxPooling Layer.
    # Because by this point image is already in 32 x 32 x 3 dimension
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)

    # Expansion Process Layer 1
    # Conv2DTranspose is opposite of Conv2D layers
    # Conv2D tells what features are in the image. Conv2DTranspose tells the location of features 
    # identified by Conv2D
    # So, to combine the feature information with location information, we concatenate the u6 and c4 layers.
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])

    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c6)

    # Expansion Process Layer 2
    # Follow the same for expansaion layer 1
    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])

    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c7)

    # Expansion Process Layer 3
    # Follow the same for expansaion layer 1
    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])

    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c8)

    # Expansion Process Layer 2
    # Follow the same for expansaion layer 1
    # except extracting any featuers, use current features to get probabilities for each pixel between 0 and 1.
    # This value between 0 and 1 will represent the paths.
    # And these probabilities are actually learned by some loss functions.
    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(u9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.load_weights("model_for_road_segmentation.h5")

    return model

print("tensorflow imported.")

model = load_model()

print("model loaded.")

class PredictConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predict'

