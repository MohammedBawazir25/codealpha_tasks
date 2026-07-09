"""
evaluate_model.py
"""

import os

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    classification_report
)

from src.config import FIGURE_DIR, REPORT_DIR


class ModelEvaluator:

    def __init__(self):

        os.makedirs(FIGURE_DIR, exist_ok=True)

        os.makedirs(REPORT_DIR, exist_ok=True)

    def evaluate(

        self,

        model,

        X_test,

        y_test

    ):

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)[:, 1]

        # -------------------------
        # Classification Report
        # -------------------------

        report = classification_report(

            y_test,

            predictions,

            output_dict=True

        )

        report_df = pd.DataFrame(report).transpose()

        report_df.to_csv(

            os.path.join(

                REPORT_DIR,

                "classification_report.csv"

            )

        )

        print("\nClassification Report Saved")

        # -------------------------
        # Confusion Matrix
        # -------------------------

        fig = plt.figure(figsize=(6, 6))

        ConfusionMatrixDisplay.from_predictions(

            y_test,

            predictions

        )

        plt.savefig(

            os.path.join(

                FIGURE_DIR,

                "confusion_matrix.png"

            )

        )

        plt.close(fig)

        # -------------------------
        # ROC Curve
        # -------------------------

        fig = plt.figure(figsize=(6, 6))

        RocCurveDisplay.from_predictions(

            y_test,

            probabilities

        )

        plt.savefig(

            os.path.join(

                FIGURE_DIR,

                "roc_curve.png"

            )

        )

        plt.close(fig)

        # -------------------------
        # Precision Recall Curve
        # -------------------------

        fig = plt.figure(figsize=(6, 6))

        PrecisionRecallDisplay.from_predictions(

            y_test,

            probabilities

        )

        plt.savefig(

            os.path.join(

                FIGURE_DIR,

                "precision_recall_curve.png"

            )

        )

        plt.close(fig)

        print("\nEvaluation Completed")