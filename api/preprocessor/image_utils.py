import numpy as np
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    def __init__(self, target_size=(224, 224)):
        """
        Initialize the image preprocessor with notebook settings.
        """
        self.target_size = target_size

    def validate_image(self, image):
        """
        Validate if the image meets notebook requirements.
        """
        try:
            min_size = min(self.target_size)
            if not (image.size[0] >= min_size and image.size[1] >= min_size):
                return False, f"Image must be at least {min_size}x{min_size} pixels"
            return True, ""
        except Exception as e:
            return False, f"Invalid image: {str(e)}"

    def preprocess_image(self, image_data):
        """
        Preprocess image exactly as done in the notebook.
        """
        try:
            # Convert bytes to PIL Image if necessary
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            else:
                image = image_data

            # Validate image
            is_valid, error_message = self.validate_image(image)
            if not is_valid:
                raise ValueError(error_message)

            # Convert to RGB if needed (matching notebook)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize (matching notebook)
            image = image.resize(self.target_size)

            # Convert to numpy array and normalize exactly as notebook does
            img_array = np.array(image)
            img_array = img_array.astype('float32') / 255.0

            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)

            return img_array

        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")

    def decode_prediction(self, prediction, class_labels):
        """
        Decode model prediction into human-readable format.
        """
        try:
            pred_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][pred_idx])
            
            probabilities = {
                label: float(prob) 
                for label, prob in zip(class_labels, prediction[0])
            }
            
            return {
                "classification": class_labels[pred_idx],
                "confidence": confidence,
                "details": {
                    "class_probabilities": probabilities
                }
            }
        except Exception as e:
            logger.error(f"Failed to decode prediction: {str(e)}")
            raise ValueError(f"Failed to decode prediction: {str(e)}")