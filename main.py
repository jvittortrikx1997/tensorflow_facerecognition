import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import matplotlib.pyplot as plt
import mysql.connector
from datetime import datetime
from sklearn.model_selection import train_test_split

dir_A = r"C:\Users\vitor\PycharmProjects\tensorflow_facerecognition\Solicitantes"
dir_B = r"C:\Users\vitor\PycharmProjects\tensorflow_facerecognition\blacklist"

db_config = {
    'user': 'root',
    'password': '040498',
    'host': '127.0.0.1',
    'database': 'face_recognition',
    'port': 3306
}

def connect_db():
    return mysql.connector.connect(**db_config)

def get_pesid(image_name):
    connection = connect_db()
    cursor = connection.cursor()
    query = f"SELECT pesid FROM imagem WHERE caminho_imagem LIKE '%{image_name}%'"
    cursor.execute(query)
    result = cursor.fetchone()
    cursor.close()
    connection.close()
    return result[0] if result else None

def insert_suspect(pesid, image_path):
    connection = connect_db()
    cursor = connection.cursor()
    insert_query = "INSERT INTO solicitacao_suspeita (pesid, data, imagem) VALUES (%s, %s, %s)"
    cursor.execute(insert_query, (pesid, datetime.now(), image_path))
    connection.commit()
    cursor.close()
    connection.close()

def preprocess_image(image_path):
    # Carregar a imagem
    img = cv2.imread(image_path)
    detector = MTCNN()

    # Detectar rostos
    result = detector.detect_faces(img)
    if result:
        # Extrair o rosto
        x, y, w, h = result[0]['box']
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))  # Tamanho padrão para o FaceNet
        face = face / 255.0  # Normalizar a imagem
        return np.expand_dims(face, axis=0)  # Adicionar dimensão para batch
    return None

import tensorflow as tf

def create_siamese_model(input_shape):
    base_model = tf.keras.models.load_model(r'C:\Users\vitor\PycharmProjects\tensorflow_facerecognition\keras-facenet\facenet_keras.h5')

    def siamese_tower():
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='sigmoid')  # Embeddings faciais compactos
        ])
        return model

    input_A = layers.Input(shape=input_shape)
    input_B = layers.Input(shape=input_shape)

    tower_A = siamese_tower()(input_A)
    tower_B = siamese_tower()(input_B)

    # Distância Euclidiana entre os embeddings
    distance = layers.Lambda(lambda tensors: K.sqrt(K.sum(K.square(tensors[0] - tensors[1]), axis=-1, keepdims=True)))([tower_A, tower_B])
    outputs = layers.Dense(1, activation="sigmoid")(distance)

    siamese_network = models.Model(inputs=[input_A, input_B], outputs=outputs)
    return siamese_network


model = create_siamese_model((112, 112, 3))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def generate_training_pairs(dir_A, dir_B):
    positive_pairs = []
    negative_pairs = []

    images_A = os.listdir(dir_A)
    images_B = os.listdir(dir_B)

    # Gerar pares positivos (mesmo nome em ambas as pastas)
    for img_A in images_A:
        if img_A in images_B:
            img_A_path = os.path.join(dir_A, img_A)
            img_B_path = os.path.join(dir_B, img_A)
            preprocessed_A = preprocess_image(img_A_path)
            preprocessed_B = preprocess_image(img_B_path)
            if preprocessed_A is not None and preprocessed_B is not None:
                positive_pairs.append((preprocessed_A, preprocessed_B))

    # Gerar pares negativos (imagens diferentes)
    for img_A in images_A:
        for img_B in images_B:
            if img_A != img_B:
                img_A_path = os.path.join(dir_A, img_A)
                img_B_path = os.path.join(dir_B, img_B)
                preprocessed_A = preprocess_image(img_A_path)
                preprocessed_B = preprocess_image(img_B_path)
                if preprocessed_A is not None and preprocessed_B is not None:
                    negative_pairs.append((preprocessed_A, preprocessed_B))

    pairs = positive_pairs + negative_pairs
    labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

    return pairs, labels


def load_image_pairs(pairs):
    images_A = []
    images_B = []
    for pair in pairs:
        img_A = preprocess_image(pair[0])
        img_B = preprocess_image(pair[1])
        images_A.append(img_A)
        images_B.append(img_B)

    images_A = np.concatenate(images_A, axis=0)
    images_B = np.concatenate(images_B, axis=0)

    return images_A, images_B

pairs, labels = generate_training_pairs(dir_A, dir_B)
images_A, images_B = load_image_pairs(pairs)
labels = np.array(labels)

train_images_A, val_images_A, train_images_B, val_images_B, train_labels, val_labels = train_test_split(
    images_A, images_B, labels, test_size=0.2, random_state=42
)
train_datagen = ImageDataGenerator(
    rotation_range=20,         # Rotaciona a imagem até 20 graus
    width_shift_range=0.2,     # Deslocamento horizontal
    height_shift_range=0.2,    # Deslocamento vertical
    shear_range=0.2,           # Cisalhamento
    zoom_range=0.2,            # Zoom na imagem
    horizontal_flip=True,      # Flip horizontal
    fill_mode='nearest'        # Preenche com valores próximos
)

train_generator = train_datagen.flow([train_images_A, train_images_B], train_labels, batch_size=16)
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow([val_images_A, val_images_B], val_labels, batch_size=16)

def data_generator(pairs, labels, batch_size):
    while True:
        for start in range(0, len(pairs), batch_size):
            end = min(start + batch_size, len(pairs))
            batch_pairs = pairs[start:end]
            batch_labels = labels[start:end]

            images_A = []
            images_B = []
            for pair in batch_pairs:
                images_A.append(pair[0])
                images_B.append(pair[1])

            yield [np.array(images_A), np.array(images_B)], np.array(batch_labels)

train_pairs, val_pairs, train_labels, val_labels = train_test_split(pairs, labels, test_size=0.2, random_state=42)
train_generator = data_generator(train_pairs, train_labels, batch_size=16)
val_generator = data_generator(val_pairs, val_labels, batch_size=16)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    steps_per_epoch=len(train_pairs) // 16,
    validation_steps=len(val_pairs) // 16
)


plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

predictions = model.predict([images_A, images_B])

plt.figure(figsize=(10, 6))
plt.plot(predictions, label='Predições do Modelo')
plt.title('Predições de Similaridade')
plt.xlabel('Índice da Amostra')
plt.ylabel('Probabilidade de Similaridade')
plt.legend()
plt.show()

def detect_frauds(model, dir_A, dir_B, threshold=0.5):
    for img_A in os.listdir(dir_A):
        img_A_path = os.path.join(dir_A, img_A)
        img_A_array = preprocess_image(img_A_path)

        if img_A_array is not None:
            for img_B in os.listdir(dir_B):
                img_B_path = os.path.join(dir_B, img_B)
                img_B_array = preprocess_image(img_B_path)

                if img_B_array is not None:
                    # Fazer a predição da similaridade
                    prediction = model.predict([img_A_array, img_B_array])[0][0]

                    # Verificar se a similaridade é maior que o threshold
                    if prediction > threshold:
                        print(f"Correspondência detectada: {img_A} e {img_B} (Similaridade: {prediction:.2f})")
                        pesid = get_pesid(img_A)
                        if pesid:
                            insert_suspect(pesid, img_A_path)

detect_frauds(model, dir_A, dir_B)
