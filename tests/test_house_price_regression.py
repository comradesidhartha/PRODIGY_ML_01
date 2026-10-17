import unittest

import pandas as pd

from house_price_regression import predict_price, train_and_evaluate


class HousePriceRegressionTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "area": [500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300],
                "bedrooms": [1, 1, 2, 2, 3, 3, 3, 4, 4, 5],
                "bathrooms": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4],
                "price": [100, 120, 160, 190, 240, 260, 300, 340, 360, 420],
            }
        )

    def test_trains_evaluates_and_predicts(self):
        model, metrics = train_and_evaluate(self.data)

        self.assertEqual(set(metrics), {"mae", "rmse", "r2"})
        self.assertTrue(all(value >= 0 for key, value in metrics.items() if key != "r2"))
        self.assertGreater(predict_price(model, 2500, 5, 4), 0)

    def test_missing_features_are_imputed(self):
        self.data.loc[0, "area"] = None

        model, _ = train_and_evaluate(self.data)

        self.assertGreater(predict_price(model, 1000, 2, 2), 0)

    def test_missing_required_column_is_reported(self):
        with self.assertRaisesRegex(ValueError, "bathrooms"):
            train_and_evaluate(self.data.drop(columns=["bathrooms"]))


if __name__ == "__main__":
    unittest.main()
