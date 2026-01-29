import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.components.data_transformation import DataTransformation

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, resume_text):
        try:
            # 1. Load the artifacts
            # We use os.path.join to work on Windows and Linux
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            le_path = os.path.join("artifacts", "label_encoder.pkl")

            print("Loading models...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            le = load_object(file_path=le_path)

            # 2. Clean the input text
            # We reuse the logic from DataTransformation to ensure consistency
            cleaner = DataTransformation()
            cleaned_text = cleaner.clean_resume_text(resume_text)

            # 3. Transform (Vectorize)
            # Note: We use .transform(), NOT .fit_transform()
            vectorized_text = preprocessor.transform([cleaned_text]).toarray()

            # 4. Predict
            prediction_id = model.predict(vectorized_text)
            
            # --- THE FIX: Convert float (6.0) to int (6) ---
            prediction_id = prediction_id.astype(int)

            # 5. Convert ID back to Category Name
            category_name = le.inverse_transform(prediction_id)[0]

            return category_name

        except Exception as e:
            raise CustomException(e, sys)