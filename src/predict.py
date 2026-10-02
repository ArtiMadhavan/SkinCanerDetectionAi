import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from utils.config import IMG_SIZE, CLASSES, CLASS_NAMES_MAP

def predict_image(model, img_path=None, img_array=None):
    """
    Predicts the class of a single image.
    Supports either a file path or a raw PIL/numpy array (from Streamlit).
    """
    if img_path is not None:
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
    elif img_array is not None:
        # Assume it's already resized to IMG_SIZE or do it here
        if img_array.shape[:2] != IMG_SIZE:
            img_array = tf.image.resize(img_array, IMG_SIZE).numpy()
    else:
        raise ValueError("Must provide img_path or img_array")
        
    # Preprocess
    img_array = np.expand_dims(img_array, axis=0)# Same as ImageDataGenerator rescale
    
    # Predict
    predictions = model.predict(img_array)[0]
    
    # Get top 3 predictions
    top_indices = np.argsort(predictions)[-3:][::-1]
    
    results = []
    for idx in top_indices:
        class_short = CLASSES[idx]
        class_full = CLASS_NAMES_MAP[class_short]
        prob = predictions[idx] * 100
        results.append({
            'class_short': class_short,
            'class_full': class_full,
            'probability': prob,
            'index': idx
        })
        
    return results, img_array[0] # Return top 3 results and preprocessed image
