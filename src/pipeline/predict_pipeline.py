import sys
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity  # <--- NEW IMPORT
from src.exception import CustomException
from src.utils import load_object
from src.components.data_transformation import DataTransformation

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, resume_text, job_description=None):
        try:
            # 1. Load artifacts
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            le_path = os.path.join("artifacts", "label_encoder.pkl")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            le = load_object(file_path=le_path)

            # 2. Clean Resume
            cleaner = DataTransformation()
            cleaned_resume = cleaner.clean_resume_text(resume_text)

            # 3. Vectorize Resume (Transform text to numbers)
            resume_vector = preprocessor.transform([cleaned_resume]).toarray()

            # 4. Predict Category
            prediction_id = model.predict(resume_vector)
            
            # Fix for the Float/Int error
            prediction_id = prediction_id.astype(int)
            
            category_name = le.inverse_transform(prediction_id)[0]

            # 5. Calculate Match Score (If JD is provided)
            match_percentage = "N/A"
            
            if job_description:
                # Clean the JD using the SAME logic as the resume
                cleaned_jd = cleaner.clean_resume_text(job_description)
                
                # Vectorize JD
                jd_vector = preprocessor.transform([cleaned_jd]).toarray()
                
                # Calculate Cosine Similarity (Result is between 0 and 1)
                # We multiply by 100 to get a percentage
                similarity = cosine_similarity(resume_vector, jd_vector)[0][0]
                match_percentage = round(similarity * 100, 2)

            return category_name, match_percentage

        except Exception as e:
            raise CustomException(e, sys)