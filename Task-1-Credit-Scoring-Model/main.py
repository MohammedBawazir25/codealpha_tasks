from src.data_loader import load_data
from src.preprocess import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.train_model import ModelTrainer
from src.evaluate_model import ModelEvaluator
from src.logger import logger


def main():

    logger.info("Project Started")

    print("=" * 60)
    print("Professional Credit Scoring Model")
    print("=" * 60)

    # Load Dataset
    df = load_data()

    # Preprocess
    processor = DataPreprocessor()
    df = processor.preprocess(df)

    # Feature Engineering
    engineer = FeatureEngineer()
    df = engineer.create_features(df)

    # Split
    X_train, X_test, y_train, y_test = engineer.split_data(df)

    # Apply SMOTE
    X_train_balanced, y_train_balanced = engineer.apply_smote(
        X_train,
        y_train
    )

    # Scale data
    X_train_scaled, X_test_scaled = engineer.scale_data(
        X_train_balanced,
        X_test
    )

    # Train Models
    trainer = ModelTrainer()

    best_model = trainer.train(
        X_train_scaled,
        X_test_scaled,
        X_train_balanced,
        X_test,
        y_train_balanced,
        y_test
    )

    # Evaluate Best Model
    evaluator = ModelEvaluator()

    if best_model.__class__.__name__ == "LogisticRegression":
        evaluator.evaluate(
            best_model,
            X_test_scaled,
            y_test
        )
    else:
        evaluator.evaluate(
            best_model,
            X_test,
            y_test
        )

    print("\nProject Completed Successfully")


if __name__ == "__main__":
    main()