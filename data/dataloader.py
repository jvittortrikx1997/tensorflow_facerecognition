import os
import numpy as np
from data.preprocess import preprocess_image

def generate_training_pairs(dir_A, dir_B):
    pos, neg = [], []
    imgs_A = os.listdir(dir_A)
    imgs_B = os.listdir(dir_B)

    for img in imgs_A:
        if img in imgs_B:
            pos.append((os.path.join(dir_A, img), os.path.join(dir_B, img)))

    for a in imgs_A:
        for b in imgs_B:
            if a != b:
                neg.append((os.path.join(dir_A, a), os.path.join(dir_B, b)))

    pairs = pos + neg
    labels = [1] * len(pos) + [0] * len(neg)
    return pairs, np.array(labels)

def load_image_pairs(pairs):
    A, B = [], []
    for img_A_path, img_B_path in pairs:
        A.append(preprocess_image(img_A_path))
        B.append(preprocess_image(img_B_path))
    return np.concatenate(A, axis=0), np.concatenate(B, axis=0)