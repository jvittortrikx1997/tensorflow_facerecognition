import os
import cv2
import numpy as np
import mysql.connector
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def conectar_bd():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="tcc"
    )

def inserir_suspeito(pesid, nome_suspeito, imagem_path):
    conn = conectar_bd()
    cursor = conn.cursor()
    query = "INSERT INTO solicitacao_suspeita (pesid, data_insercao, imagem) VALUES (%s, NOW(), %s)"
    cursor.execute(query, (pesid, imagem_path))
    conn.commit()
    cursor.close()
    conn.close()

def consultar_blacklist(nome_suspeito):
    conn = conectar_bd()
    cursor = conn.cursor()
    query = "SELECT T1.pesid FROM blacklist T1 INNER JOIN imagem T2 ON T1.pesid = T2.pesid WHERE caminho_imagem LIKE %s"
    cursor.execute(query, (nome_suspeito,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    if result:
        return result[0]
    return None

solicitantes_path = r"C:\Users\joao.mendonca\Desktop\tensorflow_facerecognition\Solicitantes"
blacklist_path = r"C:\Users\joao.mendonca\Desktop\tensorflow_facerecognition\blacklist"

tf.keras.backend.clear_session()

base_model = tf.keras.applications.VGG16(include_top=False, input_shape=(224, 224, 3), pooling='avg')

model = models.Sequential([
    base_model,
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    return img

def load_dataset(folder, label):
    images = []
    labels = []
    for file in os.listdir(folder):
        image_path = os.path.join(folder, file)
        img = load_and_preprocess_image(image_path)
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

solicitantes_images, solicitantes_labels = load_dataset(solicitantes_path, 0)
blacklist_images, blacklist_labels = load_dataset(blacklist_path, 1)

X = np.concatenate([solicitantes_images, blacklist_images], axis=0)
y = np.concatenate([solicitantes_labels, blacklist_labels], axis=0)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

y_pred = (model.predict(X_val) > 0.5).astype("int32")

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc_roc = roc_auc_score(y_val, y_pred)

print(f"Acurácia: {accuracy}")
print(f"Precisão: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC-ROC: {auc_roc}")

def plot_metric(history, metric):
    plt.plot(history.history[metric], label=f'Training {metric}')
    plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.title(f'Training and Validation {metric.capitalize()}')
    plt.show()

plot_metric(history, 'accuracy')
plot_metric(history, 'loss')  # Plotando também a perda

plt.figure()
plt.plot(history.history.get('recall', []), label='Training Recall')
plt.plot(history.history.get('val_recall', []), label='Validation Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.title('Training and Validation Recall')
plt.show()

plt.figure()
plt.plot(history.history.get('f1', []), label='Training F1 Score')
plt.plot(history.history.get('val_f1', []), label='Validation F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.title('Training and Validation F1 Score')
plt.show()

for i in range(len(y_pred)):
    if y_pred[i] == 1:
        nome_suspeito = os.path.splitext(os.listdir(blacklist_path)[i])[0]
        pesid = consultar_blacklist(nome_suspeito)

        if pesid:
            imagem_path = os.path.join(blacklist_path, os.listdir(blacklist_path)[i])
            inserir_suspeito(pesid, nome_suspeito, imagem_path)
            print(f"Suspeito {nome_suspeito} com pesid {pesid} inserido.")
        else:
            print(f"Suspeito {nome_suspeito} não encontrado na blacklist.")
