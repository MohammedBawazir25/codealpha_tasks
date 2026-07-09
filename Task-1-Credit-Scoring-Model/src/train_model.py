import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score

)

from src.config import (

    RANDOM_STATE,

    MODEL_PATH

)

import joblib


class ModelTrainer:

    def __init__(self):

        self.models = {

            "Logistic Regression":

            LogisticRegression(

                max_iter=1000,

                random_state=RANDOM_STATE

            ),

            "Decision Tree":

            DecisionTreeClassifier(

                random_state=RANDOM_STATE

            ),

            "Random Forest":

            RandomForestClassifier(

                n_estimators=200,

                random_state=RANDOM_STATE,

                n_jobs=-1

            )

        }

    def evaluate(self, model, X, y):

        pred = model.predict(X)

        prob = model.predict_proba(X)[:, 1]

        return {

            "Accuracy": accuracy_score(y, pred),

            "Precision": precision_score(y, pred),

            "Recall": recall_score(y, pred),

            "F1": f1_score(y, pred),

            "ROC_AUC": roc_auc_score(y, prob)

        }

    def train(

        self,

        X_train_scaled,

        X_test_scaled,

        X_train,

        X_test,

        y_train,

        y_test

    ):

        results = []

        best_auc = 0

        best_model = None

        best_name = None

        for name, model in self.models.items():

            print(f"\nTraining {name}")

            if name == "Logistic Regression":

                model.fit(

                    X_train_scaled,

                    y_train

                )

                metrics = self.evaluate(

                    model,

                    X_test_scaled,

                    y_test

                )

            else:

                model.fit(

                    X_train,

                    y_train

                )

                metrics = self.evaluate(

                    model,

                    X_test,

                    y_test

                )

            metrics["Model"] = name

            results.append(metrics)

            if metrics["ROC_AUC"] > best_auc:

                best_auc = metrics["ROC_AUC"]

                best_model = model

                best_name = name

        results = pd.DataFrame(results)

        print(results)

        print(f"\nBest Model : {best_name}")

        joblib.dump(

            best_model,

            MODEL_PATH

        )

        return best_model