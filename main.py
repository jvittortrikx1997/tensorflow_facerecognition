import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import mysql.connector
from datetime import datetime
import cv2

directory_a = r"C:\Users\vitor\PycharmProjects\tensorflow_facerecognition\Solicitantes"
directory_b = r"C:\Users\vitor\PycharmProjects\face_recognition\blacklist"

db_config = {
    'user': 'root',
    'password': '040498',
    'host': '127.0.0.1',
    'port': 3306,
    'database': 'face_recognition'
}

tf.keras.backend.clear_session()

def load_images(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(filename)
    return np.array(images), np.array(labels)

images_a, labels_a = load_images(directory_a)
images_b, labels_b = load_images(directory_b)

images_a = images_a.astype('float32') / 255.0
images_b = images_b.astype('float32') / 255.0

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_model()
history = model.fit(images_a, np.ones(len(images_a)), epochs=10, validation_data=(images_b, np.zeros(len(images_b))))

epochs = history.epoch
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

y_true = np.array([1] * len(images_a) + [0] * len(images_b))  # 1 para solicitantes, 0 para blacklist
y_scores = model.predict(np.concatenate((images_a, images_b)), verbose=0)

auc_roc = roc_auc_score(y_true, y_scores)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, label='Acurácia Treinamento')
plt.plot(epochs, val_acc, label='Acurácia Validação')
plt.title('Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, label='Perda Treinamento')
plt.plot(epochs, val_loss, label='Perda Validação')
plt.title('Perda')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

plt.suptitle(f'AUC-ROC: {auc_roc:.4f}')
plt.show()


def insert_into_db(pesid, image_path):
    pesid_value = str(pesid)
    image_path_value = str(image_path)  # Converte numpy.str_ para str
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    print(f'Tipo de pesid: {type(pesid_value)}')
    print(f'Tipo de image_path: {type(image_path_value)}')  # Atualizado para verificar o tipo após a conversão

    # Formata a data
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    query = "INSERT INTO solicitacao_suspeita (pesid, data, imagem) VALUES (%s, %s, %s)"

    try:
        cursor.execute(query, (pesid_value, now, image_path_value))
        conn.commit()
        print("Dados inseridos com sucesso.")
    except mysql.connector.Error as err:
        print("Erro ao inserir dados:", err)
    finally:
        cursor.close()
        conn.close()


for i, img in enumerate(images_a):
    img_flat = img.reshape(1, 128, 128, 3)
    prediction = model.predict(img_flat)
    if prediction[0][0] > 0.5:
        image_name = labels_a[i]
        query = f"SELECT pesid FROM imagem WHERE caminho_imagem LIKE '%{image_name}%'"

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(query)
        pesid = cursor.fetchone()

        if pesid and pesid[0] is not None:
            print(f'Pesid encontrado: {pesid[0]}')
            insert_into_db(pesid[0], image_name)  # Não precisa de str() aqui, já está como string
        cursor.close()
        conn.close()

print("Processo concluído.")
