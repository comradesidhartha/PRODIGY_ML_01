"""Train and use a linear regression model for house prices."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

FEATURE_COLUMNS = ["area", "bedrooms", "bathrooms"]
TARGET_COLUMN = "price"


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate and separate the expected numeric features and target."""
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required columns: {missing}")

    selected = data.loc[:, required_columns].copy()
    for column in required_columns:
        try:
            selected[column] = pd.to_numeric(selected[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"Column '{column}' must contain numeric values.") from error

    selected = selected.dropna(subset=[TARGET_COLUMN])
    features = selected[FEATURE_COLUMNS]
    if features.isna().all().any():
        columns = features.columns[features.isna().all()].tolist()
        raise ValueError(f"Feature columns cannot be entirely empty: {', '.join(columns)}")
    if len(selected) < 5:
        raise ValueError("At least 5 rows with a price are required to train and evaluate the model.")

    return features, selected[TARGET_COLUMN]


def create_model() -> Pipeline:
    """Create a model that fills missing feature values before regression."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", LinearRegression()),
        ]
    )


def train_and_evaluate(
    data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[Pipeline, dict[str, float]]:
    """Evaluate on a held-out split, then fit the returned model on all rows."""
    features, target = prepare_data(data)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )
    if len(y_test) < 2:
        raise ValueError("The test split must contain at least 2 rows to calculate R².")

    evaluation_model = create_model()
    evaluation_model.fit(x_train, y_train)
    predictions = evaluation_model.predict(x_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
    }

    model = create_model()
    model.fit(features, target)
    return model, metrics


def predict_price(model: Pipeline, area: float, bedrooms: float, bathrooms: float) -> float:
    """Predict the price for one house."""
    house = pd.DataFrame(
        [[area, bedrooms, bathrooms]],
        columns=FEATURE_COLUMNS,
    )
    return float(model.predict(house)[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a linear regression model using area, bedrooms, and bathrooms."
    )
    parser.add_argument("--data", required=True, type=Path, help="Path to a Housing.csv file.")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows used for evaluation (default: 0.2).",
    )
    parser.add_argument("--model-output", type=Path, help="Optional path to save the trained model.")
    parser.add_argument("--area", type=float, help="Area of a house to predict.")
    parser.add_argument("--bedrooms", type=float, help="Number of bedrooms to predict.")
    parser.add_argument("--bathrooms", type=float, help="Number of bathrooms to predict.")
    args = parser.parse_args()

    prediction_values = (args.area, args.bedrooms, args.bathrooms)
    if any(value is not None for value in prediction_values) and any(
        value is None for value in prediction_values
    ):
        parser.error("--area, --bedrooms, and --bathrooms must be supplied together.")

    try:
        data = pd.read_csv(args.data)
        model, metrics = train_and_evaluate(data, test_size=args.test_size)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        parser.error(str(error))

    print(f"Mean absolute error (MAE): {metrics['mae']:.2f}")
    print(f"Root mean squared error (RMSE): {metrics['rmse']:.2f}")
    print(f"R-squared (R2): {metrics['r2']:.4f}")

    if args.model_output:
        joblib.dump(model, args.model_output)
        print(f"Saved model to {args.model_output}")

    if args.area is not None:
        price = predict_price(model, args.area, args.bedrooms, args.bathrooms)
        print(f"Predicted price: {price:.2f}")


if __name__ == "__main__":
    main()
