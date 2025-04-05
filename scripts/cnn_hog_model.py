from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

def build_cnn_hog_model():
    """Build the CNN + HOG model."""
    input_image = layers.Input(shape=(128, 128, 1))
    input_hog = layers.Input(shape=(468,))  # Length of HOG features

    # CNN for image input
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_image)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    
    # Combine with HOG features
    x = layers.concatenate([x, input_hog])
    
    # Fully connected layer
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    
    # Output layer
    output = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=[input_image, input_hog], outputs=output)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model
