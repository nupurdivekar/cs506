from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cancer Cell Detection API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model at startup
MODEL_PATH = os.path.join('api', 'models', 'breast_cancer_cnn_model (1).keras')
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

# Class labels from the notebook
CLASS_LABELS = ['High', 'Low', 'Stroma']

async def preprocess_image(image_file: UploadFile):
    """
    Exactly matches the notebook's preprocessing pipeline.
    """
    try:
        # Read the image file
        image_data = await image_file.read()
        img = Image.open(io.BytesIO(image_data))
        
        # Log original image info
        logger.info(f"Processing image: size={img.size}, mode={img.mode}")

        # Match notebook preprocessing exactly
        if img.mode != 'RGB':
            img = img.convert('RGB')
            logger.info("Converted image to RGB")

        # Resize to target size (matching notebook)
        target_size = (224, 224)
        img = img.resize(target_size)
        logger.info(f"Resized image to {target_size}")

        # Convert to numpy and normalize exactly as notebook does
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        logger.info(f"Normalized array: shape={img_array.shape}, min={img_array.min()}, max={img_array.max()}")

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")

@app.post("/api/analyze")
async def analyze_image(file: UploadFile):
    """
    Endpoint to analyze cell images for cancer detection.
    """
    request_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + str(hash(file.filename))[:8]
    logger.info(f"Processing request {request_id} for file: {file.filename}")

    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Preprocess image exactly as notebook does
        processed_image = await preprocess_image(file)

        # Get prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Log raw predictions for debugging
        logger.info("Raw predictions: " + str(predictions[0]))
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        classification = CLASS_LABELS[predicted_class_idx]

        # Log prediction details
        logger.info(f"Predicted class: {classification}")
        logger.info(f"Confidence: {confidence*100:.2f}%")
        logger.info("All class probabilities:")
        for label, prob in zip(CLASS_LABELS, predictions[0]):
            logger.info(f"- {label}: {prob*100:.2f}%")

        # Prepare response
        result = {
            "classification": classification,
            "confidence": confidence,
            "details": {
                "class_probabilities": {
                    label: float(conf) 
                    for label, conf in zip(CLASS_LABELS, predictions[0])
                },
                "message": f"Image classified as {classification} with {confidence*100:.2f}% confidence",
                "analysis_metadata": {
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat(),
                    "model_version": "notebook_trained_v1",
                    "image_size": "224x224"
                }
            }
        }

        logger.info(f"Successfully processed request {request_id}")
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Analysis failed for request {request_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint that also verifies model status.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "available_classes": CLASS_LABELS,
        "target_size": "224x224",
        "version": "notebook_trained_v1"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)