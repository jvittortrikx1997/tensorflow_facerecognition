from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(112, 112), color_mode='grayscale')
    img = img_to_array(img)
    img = np.repeat(img, 3, axis=-1)
    img = img / 255.0
    return np.expand_dims(img, axis=0)