# Titanic Survival Prediction

A beginner-friendly, modular machine learning project that predicts whether a
passenger survived the sinking of the Titanic, using the classic
[Kaggle Titanic dataset](https://www.kaggle.com/c/titanic). Built with
pandas, scikit-learn, and Logistic Regression.

## Project Overview

This project walks through a complete, standard supervised learning workflow:

1. Load the raw Titanic data
2. Clean it (handle missing values, drop unhelpful columns)
3. Encode categorical features
4. Train a Logistic Regression classifier
5. Evaluate it on a held-out validation split
6. Generate predictions on Kaggle's official test set as a `submission.csv`

It started as a single Jupyter notebook and has been refactored into a
modular Python package so it's easier to read, test, extend, and reuse -
while keeping the exact same data-processing logic and results.

## Dataset

The project uses the [Kaggle "Titanic - Machine Learning from Disaster"](https://www.kaggle.com/c/titanic/data)
dataset, which is **not included in this repository** (per Kaggle's terms).
Download the following three files and place them in the `data/` folder:

- `train.csv` - labeled training data (includes the `Survived` column)
- `test.csv` - unlabeled data Kaggle uses to score submissions
- `gender_submission.csv` - an example submission file showing the expected format

| Column        | Description                                   |
|---------------|------------------------------------------------|
| PassengerId   | Unique ID for each passenger                   |
| Survived      | Target: 0 = did not survive, 1 = survived       |
| Pclass        | Ticket class (1st, 2nd, 3rd)                    |
| Name          | Passenger name                                  |
| Sex           | male / female                                   |
| Age           | Age in years                                    |
| SibSp         | # of siblings/spouses aboard                    |
| Parch         | # of parents/children aboard                    |
| Ticket        | Ticket number                                   |
| Fare          | Passenger fare                                  |
| Cabin         | Cabin number                                    |
| Embarked      | Port of embarkation (C, Q, S)                   |

## Folder Structure

```
Titanic-Survival-Prediction/
│
├── data/                       # Place train.csv, test.csv, gender_submission.csv here
│
├── notebooks/
│   └── Titanic_Project.ipynb   # Original exploratory notebook (kept for reference)
│
├── src/
│   ├── data_loader.py          # Loads CSV files into DataFrames
│   ├── preprocessing.py        # Missing-value handling, column drops
│   ├── feature_engineering.py  # Placeholder for future engineered features
│   ├── encoding.py             # Label + one-hot encoding (reusable fitted encoders)
│   ├── model.py                # Model creation, training, save/load
│   ├── evaluate.py             # Accuracy, precision, recall, F1, confusion matrix
│   ├── predict.py              # Predicts on Kaggle's test.csv, writes submission.csv
│   └── utils.py                # Shared helper functions
│
├── outputs/
│   ├── submission.csv          # Generated after running main.py
│   └── figures/                # Saved plots (e.g. Age distribution histogram)
│
├── main.py                     # Runs the entire pipeline end-to-end
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Titanic-Survival-Prediction.git
   cd Titanic-Survival-Prediction
   ```

2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download `train.csv`, `test.csv`, and `gender_submission.csv` from
   [Kaggle](https://www.kaggle.com/c/titanic/data) and place them in `data/`.

## Usage

Run the full pipeline from the project root:

```bash
python main.py
```

This will:
- Load and clean the data
- Print progress and intermediate DataFrame previews to the console
- Save an Age-distribution histogram to `outputs/figures/age_distribution.png`
- Train a Logistic Regression model on 80% of the training data
- Evaluate it on the remaining 20% and print all metrics
- Save the trained model to `outputs/model.joblib`
- Generate `outputs/submission.csv`, ready to upload to Kaggle

## Model Used

**Logistic Regression** (`sklearn.linear_model.LogisticRegression`,
`max_iter=10000`) - a simple, interpretable baseline classifier well suited
to a binary target like survival prediction.

Features used: `Pclass`, `Sex` (label-encoded), `Age`, `SibSp`, `Parch`,
`Fare`, `PassengerId`, and `Embarked` (one-hot encoded into `Embarked_C`,
`Embarked_Q`, `Embarked_S`).

## Evaluation Metrics

The model is evaluated on an 80/20 train/validation split
(`random_state=42`) using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**
- **Full classification report** (per-class precision/recall/F1)

All metrics are printed to the console when `main.py` runs, and returned
as a dictionary from `evaluate.evaluate_model()` for programmatic use.

## Results

Results depend on the exact `train.csv` used, but with the default 80/20
split and `random_state=42`, this Logistic Regression baseline typically
scores **around 80% accuracy** on the validation split. Run `main.py` with
your own data to see the exact numbers printed to the console.

## Future Improvements

- **Feature engineering**: `src/feature_engineering.py` already includes
  ready-to-use templates for `FamilySize` and `IsAlone`; wiring these into
  `main.py` could improve accuracy.
- **Exclude `PassengerId` as a feature**: it's currently passed into the
  model (matching the original notebook) purely because it's needed to
  build the submission file - it carries no predictive signal and could be
  set aside from `X`/`X_test` before training and re-attached only when
  writing `submission.csv`.
- **Try other models**: Random Forest, Gradient Boosting, or SVM often
  outperform plain Logistic Regression on this dataset.
- **Hyperparameter tuning**: e.g. `GridSearchCV` over regularization
  strength (`C`) and solver options.
- **Cross-validation**: replace the single train/validation split with
  k-fold cross-validation for a more robust performance estimate.
- **Extract `Title` from `Name`** (Mr, Mrs, Miss, Master, etc.) before
  dropping the column - a well-known strong predictor for this dataset.
- **Unit tests**: add `pytest` tests for each `src/` module.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
