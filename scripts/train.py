from cnn_hog_model import build_cnn_hog_model
from efficientnet_model import build_efficientnet_model
from data_preprocessing import preprocess_cnn_hog_data, preprocess_efficientnet_data
from tensorflow.keras.callbacks import EarlyStopping

def train_cnn_hog():
    """Train CNN + HOG Model."""
    images_train_1, hog_train_1, labels_train_1, images_test_1, hog_test_1, labels_test_1 = preprocess_cnn_hog_data()

    model = build_cnn_hog_model()
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # Training CNN + HOG Model
    model.fit([images_train_1, hog_train_1], labels_train_1, epochs=50, batch_size=32,
              validation_data=([images_test_1, hog_test_1], labels_test_1),
              callbacks=[early_stopping])

    # Save the model
    model.save('models/cnn_hog_model.h5')

def train_efficientnet():
    """Train EfficientNet Model."""
    images_train_2, labels_train_2, images_test_2, labels_test_2 = preprocess_efficientnet_data()

    model = build_efficientnet_model()
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # Training EfficientNet Model
    model.fit(images_train_2, labels_train_2, epochs=50, batch_size=32,
              validation_data=(images_test_2, labels_test_2),
              callbacks=[early_stopping])

    # Save the model
    model.save('models/efficientnet_model.h5')

if __name__ == '__main__':
    # Train each model
    train_cnn_hog()
    train_efficientnet()
