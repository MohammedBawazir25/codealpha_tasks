"""
data_loader.py
--------------
Loads the credit scoring dataset.
"""

import pandas as pd

from src.config import TRAIN_DATA


def load_data():
    """
    Load training dataset.
    """

    try:

        df = pd.read_csv(TRAIN_DATA)

        print("=" * 60)
        print("Dataset Loaded Successfully")
        print("=" * 60)
        print(f"Shape: {df.shape}")

        return df

    except FileNotFoundError:

        print(f"Dataset not found:\n{TRAIN_DATA}")

        raise

    except Exception as e:

        print(e)

        raise