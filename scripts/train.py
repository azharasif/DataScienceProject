from cnn_model import build_cnn_model
from efficientnet_model import build_efficientnet_model
from data_preprocessing import load_data
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def train_model(model, images_train, labels_train, images_val, labels_val, epochs=10, batch_size=32):
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    images_train = np.array(images_train, dtype=np.float32) / 255.0
    images_val = np.array(images_val, dtype=np.float32) / 255.0

    label_encoder = LabelEncoder()
    labels_train_encoded = label_encoder.fit_transform(labels_train)
    labels_val_encoded = label_encoder.transform(labels_val)

    num_classes = len(label_encoder.classes_)
    print(f"Number of classes detected: {num_classes}")

    labels_train_cat = to_categorical(labels_train_encoded, num_classes=num_classes)
    labels_val_cat = to_categorical(labels_val_encoded, num_classes=num_classes)


    # Train the model
    history = model.fit(images_train, labels_train_cat,
                        validation_data=(images_val, labels_val_cat),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stop])

    return history, num_classes


def main():
    """Main function to load data and train CNN + EfficientNet models."""

    label_column = 'MEDICINE_NAME'

    images_train, labels_train = load_data('data/Training/training_words', 'data/Training/training_labels.csv', label_column)
    images_val, labels_val = load_data('data/Validation/validation_words', 'data/Validation/validation_labels.csv', label_column)
    images_test, labels_test = load_data('data/Testing/testing_words', 'data/Testing/testing_labels.csv', label_column)

    input_shape = (128, 128, 3)

    # print("Training CNN model...")
    # cnn_model = build_cnn_model(input_shape=input_shape)
    # train_model(cnn_model, images_train, labels_train, images_val, labels_val)

    print("Training EfficientNet model...")
    efficientnet_model = build_efficientnet_model(input_shape=input_shape)
    train_model(efficientnet_model, images_train, labels_train, images_val, labels_val)


if __name__ == "__main__":
    main()
