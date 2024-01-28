import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# Load the trained model
model = keras.models.load_model('mnist_model.h5')

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(28, 28), color_mode='grayscale')
    
    # Invert colors (if needed)
    img_array = 255 - image.img_to_array(img)
    
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def predict_digit(image_path):
    # Preprocess the image
    img_array = preprocess_image(image_path)

    # Make prediction
    prediction = model.predict(img_array.astype(np.float32))
    digit = np.argmax(prediction)

    return digit


# Test the model on a new image
image_path_to_test ="test/d.png"
predicted_digit = predict_digit(image_path_to_test)

print(f'The predicted digit is: {predicted_digit}')
