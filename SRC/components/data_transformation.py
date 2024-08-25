import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from SRC.exception import CustomException
from SRC.logger import logging
from SRC.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer(self):
        """
        This method sets up the preprocessing pipelines for both numerical and categorical features.
        """
        try:
            # Defining the numerical and categorical columns
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]

            # Pipeline for numerical features
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Pipeline for categorical features
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            # Combining both pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        This method handles the data transformation for the training and test datasets.
        """
        try:
            # Loading the datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Loaded training and testing datasets.")

            # Obtaining the preprocessing object
            preprocessor = self.get_data_transformer()

            target_column = "math score"
            numerical_columns = ["writing score", "reading score"]

            # Separating features and target variable for training and test data
            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]

            logging.info("Applying preprocessing on the datasets.")

            # Transforming the datasets using the preprocessor
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Combining the transformed features with the target variable
            train_data = np.c_[X_train_transformed, np.array(y_train)]
            test_data = np.c_[X_test_transformed, np.array(y_test)]

            logging.info("Saving the preprocessing object.")

            # Saving the preprocessor object
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            return train_data, test_data, self.config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)

