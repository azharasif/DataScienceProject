from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Create an ImageDataGenerator for testing
datagen = ImageDataGenerator(rescale=1./255)

# Create a test generator from the directory structure
test_generator = datagen.flow_from_directory(
    'data/model_2_efficientnet/test',   # Path to the 'test' folder
    target_size=(224, 224),             # Resize the images to (224, 224)
    batch_size=8,
    class_mode='binary'                 # Binary classification: 'valid' vs. 'invalid'
)

# Load the trained EfficientNet model
efficientnet_model = load_model("models/efficientnet_model.h5")

# Evaluate the model on the test data
loss, accuracy = efficientnet_model.evaluate(test_generator)
print(f"Accuracy for EfficientNet model: {accuracy * 100:.2f}%")
