import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_predictions(predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, label='Predições')
    plt.title('Probabilidade de Similaridade')
    plt.xlabel('Amostra')
    plt.ylabel('Probabilidade')
    plt.legend()
    plt.show()