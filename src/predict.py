"""
predict.py
-----------
Load the trained model and predict credit risk.
"""

import joblib
import pandas as pd

from src.config import MODEL_PATH


class CreditPredictor:

    def __init__(self):

        self.model = joblib.load(MODEL_PATH)

    def predict(self, input_data):

        """
        input_data must be a dictionary.
        """

        df = pd.DataFrame([input_data])

        prediction = self.model.predict(df)[0]

        probability = self.model.predict_proba(df)[0][1]

        return prediction, probability


if __name__ == "__main__":

    sample = {

        "RevolvingUtilizationOfUnsecuredLines": 0.75,

        "age": 35,

        "NumberOfTime30-59DaysPastDueNotWorse": 1,

        "DebtRatio": 0.45,

        "MonthlyIncome": 50000,

        "NumberOfOpenCreditLinesAndLoans": 8,

        "NumberOfTimes90DaysLate": 0,

        "NumberRealEstateLoansOrLines": 1,

        "NumberOfTime60-89DaysPastDueNotWorse": 0,

        "NumberOfDependents": 2,

        "DebtPerIncome": 0.45 / 50001,

        "LatePayments": 1,

        "LoansPerDependent": 8 / 3

    }

    predictor = CreditPredictor()

    prediction, probability = predictor.predict(sample)

    print("\nPrediction:", prediction)

    print("Probability:", probability)