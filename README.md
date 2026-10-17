# PRODIGY_ML_01
Implement a linear regression model to predict the prices of houses based on their square footage and the number of bedrooms and bathrooms.

## House price regression

`house_price_regression.py` trains a scikit-learn linear regression model from a CSV dataset. The CSV must have numeric `area`, `bedrooms`, `bathrooms`, and `price` columns; rows with no price are ignored, and missing feature values are filled with the training-set median.

Install dependencies and evaluate the model on a held-out 20% split:

```bash
python -m pip install -r requirements.txt
python house_price_regression.py --data Housing.csv
```

The script reports MAE, RMSE, and R-squared. It can also predict a house price and save the model:

```bash
python house_price_regression.py --data Housing.csv --area 2000 --bedrooms 3 --bathrooms 2 --model-output house_price_model.joblib
```

Run the tests with `python -m unittest discover -s tests`.
