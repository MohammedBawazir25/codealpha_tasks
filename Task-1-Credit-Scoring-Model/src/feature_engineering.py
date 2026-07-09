from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from src.config import (
    TARGET,
    RANDOM_STATE,
    TEST_SIZE
)


class FeatureEngineer:

    def __init__(self):
        self.scaler = StandardScaler()

    def create_features(self, df):

        # Debt per income
        df["DebtPerIncome"] = (
            df["DebtRatio"] /
            (df["MonthlyIncome"] + 1)
        )

        # Total late payments
        df["LatePayments"] = (

            df["NumberOfTime30-59DaysPastDueNotWorse"] +

            df["NumberOfTime60-89DaysPastDueNotWorse"] +

            df["NumberOfTimes90DaysLate"]

        )

        # Loans per dependent
        df["LoansPerDependent"] = (

            df["NumberOfOpenCreditLinesAndLoans"] /

            (df["NumberOfDependents"] + 1)

        )

        return df

    def split_data(self, df):

        X = df.drop(columns=[TARGET])

        y = df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(

            X,

            y,

            test_size=TEST_SIZE,

            random_state=RANDOM_STATE,

            stratify=y

        )

        return X_train, X_test, y_train, y_test

    def apply_smote(self, X_train, y_train):

        smote = SMOTE(random_state=RANDOM_STATE)

        X_resampled, y_resampled = smote.fit_resample(
            X_train,
            y_train
        )

        return X_resampled, y_resampled

    def scale_data(self, X_train, X_test):

        X_train_scaled = self.scaler.fit_transform(X_train)

        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled