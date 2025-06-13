import tensorflow as tf
from tensorflow.keras import layers, models
from keras import backend as K

def create_siamese_model(input_shape):
    base = tf.keras.applications.ResNet50(weights='imagenet', input_shape=input_shape, include_top=False)
    base.trainable = False

    def tower():
        return models.Sequential([
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='sigmoid')
        ])

    inp_A = layers.Input(shape=input_shape)
    inp_B = layers.Input(shape=input_shape)

    tA = tower()(inp_A)
    tB = tower()(inp_B)

    dist = layers.Lambda(lambda x: K.sqrt(K.sum(K.square(x[0] - x[1]), axis=-1, keepdims=True)))([tA, tB])
    output = layers.Dense(1, activation="sigmoid")(dist)

    return models.Model(inputs=[inp_A, inp_B], outputs=output)