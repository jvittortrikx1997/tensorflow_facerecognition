import os
from data.preprocess import preprocess_image
from database.repository import get_pesid, insert_suspect

def detect_frauds(model, dir_A, dir_B, threshold=0.5):
    for img_A in os.listdir(dir_A):
        img_A_path = os.path.join(dir_A, img_A)
        img_A_array = preprocess_image(img_A_path)

        for img_B in os.listdir(dir_B):
            img_B_path = os.path.join(dir_B, img_B)
            img_B_array = preprocess_image(img_B_path)

            score = model.predict([img_A_array, img_B_array])[0][0]
            if score > threshold:
                print(f"Suspeita: {img_A} e {img_B} (Simil: {score:.2f})")
                pesid = get_pesid(img_A)
                if pesid:
                    insert_suspect(pesid, img_A_path)