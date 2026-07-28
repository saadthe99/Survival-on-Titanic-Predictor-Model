"""
main.py
--------
Orchestrates the full Titanic Survival Prediction pipeline end-to-end,
in the same order as the original notebook:

    1. Load data
    2. Preprocess
    3. Feature engineering (none used - see src/feature_engineering.py)
    4. Encode
    5. Train
    6. Evaluate
    7. Predict on Kaggle's test set
    8. Save submission.csv

Run from the project root with:
    python main.py
"""

from pathlib import Path

from sklearn.model_selection import train_test_split

from src import data_loader, encoding, evaluate, predict, preprocessing
from src import model as model_module
from src.utils import print_step

# ---------------------------------------------------------------------------
# Project paths - all relative to this file's location (via pathlib), so the
# project runs identically on Windows, macOS, and Linux without editing paths.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
SUBMISSION_PATH = OUTPUTS_DIR / "submission.csv"
MODEL_PATH = OUTPUTS_DIR / "model.joblib"


def main() -> None:
    """Run the complete Titanic survival prediction pipeline."""

    # -----------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------
    print_step("Loading data")
    training_data = data_loader.load_training_data(DATA_DIR)
    testing_data = data_loader.load_testing_data(DATA_DIR)

    # -----------------------------------------------------------------
    # 2. Preprocess training data
    # -----------------------------------------------------------------
    print_step("Preprocessing training data")
    training_data = preprocessing.drop_columns(training_data, ["Cabin"])
    training_data = preprocessing.fill_missing_embarked(training_data)
    preprocessing.explore_age_distribution(training_data, figures_dir=FIGURES_DIR)
    training_data = preprocessing.fill_missing_age(training_data)
    print(training_data.head())

    # Medians computed from the (now-cleaned) training data are reused
    # later to fill missing values in the Kaggle test set, so the test
    # set never "leaks" its own statistics into the pipeline.
    age_median = training_data["Age"].median()
    fare_median = training_data["Fare"].median()

    # Separate features from the target
    y = training_data["Survived"]
    X = training_data.drop("Survived", axis=1)

    # Drop columns that aren't useful as model features
    X = preprocessing.drop_columns(X, ["Ticket", "Name"])
    print(X.head())
    print(X["Embarked"].value_counts())

    # -----------------------------------------------------------------
    # 3. Feature engineering
    #    The original notebook does not engineer any additional
    #    features, so this step is a no-op here. New features can be
    #    added by calling functions from src/feature_engineering.py.
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # 4. Encode training data (fitting the encoders here)
    # -----------------------------------------------------------------
    print_step("Encoding training data")
    X, sex_encoder = encoding.encode_sex(X)
    X, embarked_encoder = encoding.encode_embarked(X)
    print(X.head())

    # -----------------------------------------------------------------
    # Preprocess + encode the Kaggle test data, reusing the encoders
    # that were already fit on the training data (never re-fit them).
    # -----------------------------------------------------------------
    print_step("Preprocessing Kaggle test data")
    print(testing_data.describe())
    testing_data.info()
    testing_data = preprocessing.drop_columns(testing_data, ["Cabin", "Ticket", "Name"])
    print(testing_data.head())

    X_test = testing_data
    print(X_test.head())

    print_step("Encoding Kaggle test data")
    X_test, _ = encoding.encode_sex(X_test, encoder=sex_encoder)
    print(X_test.head())
    X_test, _ = encoding.encode_embarked(X_test, encoder=embarked_encoder)
    print(X_test.head())

    # -----------------------------------------------------------------
    # 5. Train (using an internal validation split for evaluation)
    # -----------------------------------------------------------------
    print_step("Splitting data and training the model")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    titanic_model = model_module.create_model(max_iter=10000)
    titanic_model = model_module.train_model(titanic_model, X_train, y_train)

    # -----------------------------------------------------------------
    # 6. Evaluate on the held-out validation split
    # -----------------------------------------------------------------
    print_step("Evaluating model")
    y_pred = titanic_model.predict(X_val)
    evaluate.evaluate_model(y_val, y_pred)

    model_module.save_model(titanic_model, MODEL_PATH)

    print_step("Preparing Kaggle test set for prediction")
    X_test = preprocessing.fill_missing_age(X_test, reference_median=age_median)
    X_test = preprocessing.fill_missing_fare(X_test, reference_median=fare_median)
    X_test.info()

    print_step("Generating Kaggle submission file")
    predict.predict_and_generate_submission(titanic_model, X_test, SUBMISSION_PATH)

    print_step("Pipeline complete")


if __name__ == "__main__":
    main()
