from data.dataloader import generate_training_pairs, load_image_pairs
from models.siamese_model import create_siamese_model
from services.training_service import prepare_generators
from services.fraud_detection_service import detect_frauds
from utils.plot import plot_training_history, plot_predictions

dir_A = r"C:\Users\vitor\PycharmProjects\tensorflow_facerecognition\Solicitantes"
dir_B = r"C:\Users\vitor\PycharmProjects\tensorflow_facerecognition\blacklist"

pairs, labels = generate_training_pairs(dir_A, dir_B)
images_A, images_B = load_image_pairs(pairs)

model = create_siamese_model((112, 112, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_gen, val_gen = prepare_generators(images_A, images_B, labels)

history = model.fit(train_gen, validation_data=val_gen, epochs=5)
plot_training_history(history)

predictions = model.predict([images_A, images_B])
plot_predictions(predictions)

detect_frauds(model, dir_A, dir_B)