import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    label_encoder_obj_file_path = os.path.join('artifacts', "label_encoder.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def clean_resume_text(self, text):
        """
        The helper function we perfected in the Notebook.
        """
        try:
            text = re.sub('http\S+\s*', ' ', text)
            text = re.sub('RT|cc', ' ', text)
            text = re.sub('#\S+', '', text)
            text = re.sub('@\S+', '  ', text)
            text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
            text = re.sub(r'[^\x00-\x7f]', r' ', text)
            text = re.sub('\s+', ' ', text)
            return text.lower()
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data Transformation started")
            
            # 1. Load Data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train Dataframe Head: \n{train_df.head().to_string()}")

            # 2. Define Input (X) and Target (y)
            target_column_name = "Category"
            text_column_name = "Resume"

            # 3. Clean the Text (Applying the regex function)
            logging.info("Applying text cleaning...")
            train_df['cleaned_resume'] = train_df[text_column_name].apply(self.clean_resume_text)
            test_df['cleaned_resume'] = test_df[text_column_name].apply(self.clean_resume_text)

            input_feature_train = train_df['cleaned_resume'].values
            target_feature_train = train_df[target_column_name].values

            input_feature_test = test_df['cleaned_resume'].values
            target_feature_test = test_df[target_column_name].values

            # 4. Vectorization (TF-IDF) - The "Preprocessor"
            # We use max_features=3000 to keep the model lightweight but accurate
            logging.info("Applying TF-IDF Vectorization...")
            tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
            
            input_feature_train_arr = tfidf.fit_transform(input_feature_train).toarray()
            input_feature_test_arr = tfidf.transform(input_feature_test).toarray()

            # 5. Label Encoding (Target)
            logging.info("Applying Label Encoding...")
            le = LabelEncoder()
            target_feature_train_arr = le.fit_transform(target_feature_train)
            target_feature_test_arr = le.transform(target_feature_test)

            # 6. Combine features and targets into one array for the Trainer
            # Result: [ [0.1, 0.4, ..., 1], [0.2, 0.5, ..., 6] ]
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            # 7. Save Objects
            logging.info(f"Saving preprocessing objects.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=tfidf
            )
            
            save_object(
                file_path=self.data_transformation_config.label_encoder_obj_file_path,
                obj=le
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e, sys)

# --- TEST BLOCK ---
if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    
    # Run Ingestion First
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Run Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    
    print(f"Transformation Done. Train Shape: {train_arr.shape}")