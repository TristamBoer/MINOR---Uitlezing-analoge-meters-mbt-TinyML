from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import List
from MINOR_Stage_Functions import slice_rotate_image, hexdecimal_to_picture_array
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

# Define the input model
class UserInput(BaseModel):
    HEXADECIMAL_BYTES: List[int]  # Changed to accept a list of integers
    height: float
    length: float
    delta_height: float
    delta_length: float
    digit_width: float

    @field_validator("HEXADECIMAL_BYTES")
    def validate_hexadecimal_list(cls, value):
        # Ensure all items are integers and within the valid range for 16-bit values
        if not all(isinstance(item, int) and 0 <= item <= 0xFFFF for item in value):
            raise ValueError("HEXADECIMAL_BYTES must be a list of integers (0 to 65535).")
        return value

app = FastAPI()

# Load the model at startup
model = load_model(r'C:\programeren\jupiter_python\minor\case\smart_meter_model.keras')

@app.post("/predict")
def predict(input: UserInput):
    try:
        # Convert HEX to picture array and normalize
        image = hexdecimal_to_picture_array(input.HEXADECIMAL_BYTES)
        image = (image / image.max() * 255).astype("uint8")
        
        # Process the image
        processed_image = slice_rotate_image(image,height=input.height, 
                                             length=input.length, 
                                             delta_height=input.delta_height,
                                             delta_length=input.delta_length,
                                             digit_width=input.digit_width)
        
        # Prepare for prediction
        processed_image = np.array(processed_image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Convert predictions to a list for JSON serialization
        return {"predictions": predictions.tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

