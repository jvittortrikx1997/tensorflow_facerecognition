from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def prepare_generators(images_A, images_B, labels):
    A_train, A_val, B_train, B_val, y_train, y_val = train_test_split(images_A, images_B, labels, test_size=0.2)

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_gen = datagen.flow([A_train, B_train], y_train, batch_size=16)
    val_gen = ImageDataGenerator(rescale=1./255).flow([A_val, B_val], y_val, batch_size=16)

    return train_gen, val_gen