import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from SRC.exception import CustomException
from SRC.logger import logging
from SRC.components.data_transformation import DataTransformation, DataTransformationConfig
from SRC.components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', "data.csv")
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting the data ingestion process.")
        try:
            # Reading the data into a DataFrame
            df = pd.read_csv('notebook/data/StudentsPerformance.csv')
            logging.info("Dataset successfully loaded into a DataFrame.")

            # Creating directories if they do not exist
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            # Saving the raw dataset
            df.to_csv(self.config.raw_data_path, index=False, header=True)
            logging.info("Raw dataset saved.")

            # Splitting the dataset into training and test sets
            logging.info("Performing train-test split.")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Saving the split datasets
            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)
            logging.info("Data ingestion process completed successfully.")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Data ingestion
    ingestion = DataIngestion()
    train_data, test_data = ingestion.initiate_data_ingestion()

    # Data transformation
    data_transformer = DataTransformation()
    train_arr, test_arr, _ = data_transformer.initiate_data_transformation(train_data, test_data)

     # Model training
    trainer = ModelTrainer()
    print(trainer.initiate_model_trainer(train_arr, test_arr))
