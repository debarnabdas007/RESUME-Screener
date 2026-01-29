import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            
            # Slicing the numpy arrays
            # All columns EXCEPT the last one are Features (X)
            # The LAST column is the Target (y)
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Define the Model
            # We use the exact same logic as your Notebook to ensure 98% accuracy
            model = OneVsRestClassifier(RandomForestClassifier())

            logging.info(f"Training Model: {model}")
            model.fit(X_train, y_train)

            logging.info("Model training complete. Evaluating...")
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate Accuracy
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"Model Accuracy: {accuracy}")
            
            print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

            # Save the trained model
            logging.info("Saving the model object")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)

# --- THE GRAND TEST ---
# This block runs the ENTIRE pipeline from start to finish.
if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    # 1. Ingestion
    print("--- Starting Ingestion ---")
    ingestion = DataIngestion()
    train_data_path, test_data_path = ingestion.initiate_data_ingestion()

    # 2. Transformation
    print("--- Starting Transformation ---")
    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(train_data_path, test_data_path)

    # 3. Training
    print("--- Starting Training ---")
    trainer = ModelTrainer()
    trainer.initiate_model_trainer(train_arr, test_arr)