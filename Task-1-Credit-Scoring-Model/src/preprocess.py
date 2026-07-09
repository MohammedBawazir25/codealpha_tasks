"""
preprocess.py
-------------
Data preprocessing module.
"""

import pandas as pd

from sklearn.impute import SimpleImputer

from src.config import (
    TARGET,
    PROCESSED_DATA_PATH
)


class DataPreprocessor:

    def __init__(self):

        self.numeric_imputer = SimpleImputer(
            strategy="median"
        )

    def remove_unwanted_columns(self, df):

        if "Unnamed: 0" in df.columns:
            df = df.drop(
                columns=["Unnamed: 0"]
            )

        return df

    def remove_duplicates(self, df):

        before = df.shape[0]

        df = df.drop_duplicates()

        after = df.shape[0]

        print(f"Removed {before-after} duplicate rows")

        return df

    def fill_missing_values(self, df):

        numeric_columns = df.select_dtypes(
            include=["int64", "float64"]
        ).columns

        df[numeric_columns] = self.numeric_imputer.fit_transform(
            df[numeric_columns]
        )

        return df

    def clip_outliers(self, df):

        numeric_columns = df.select_dtypes(
            include=["int64", "float64"]
        ).columns

        for column in numeric_columns:

            if column == TARGET:
                continue

            Q1 = df[column].quantile(0.25)

            Q3 = df[column].quantile(0.75)

            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR

            upper = Q3 + 1.5 * IQR

            df[column] = df[column].clip(
                lower,
                upper
            )

        return df

    def preprocess(self, df):

        print("\nStarting preprocessing...")

        df = self.remove_unwanted_columns(df)

        df = self.remove_duplicates(df)

        df = self.fill_missing_values(df)

        df = self.clip_outliers(df)

        df.to_csv(
            PROCESSED_DATA_PATH,
            index=False
        )

        print("\nProcessed dataset saved to")

        print(PROCESSED_DATA_PATH)

        return df