from BaseModel import BaseModel
from darts.models import ARIMA
from darts import TimeSeries
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

class ARIMAModel(BaseModel):
    def fit_predict(self):
        results = {}
        mse_scores = {}
        all_actual = []
        all_predicted = []

        for column in self.usable_cols:
            series = TimeSeries.from_dataframe(self.data, self.dataset_info['date_col'], column)
            train, test = series.split_after(pd.Timestamp(self.dataset_info['test'][0]))

            model = ARIMA(p=self.dataset_info['lag'])
            model.fit(train)
            prediction = model.predict(len(test))

            # Collect individual predictions for MSE calculation
            actual_values = test.values().flatten()
            predicted_values = prediction.values().flatten()

            # Calculate MSE for the current column
            mse = mean_squared_error(actual_values, predicted_values)
            results[column] = prediction
            mse_scores[column] = mse

            # Append results for consolidated MSE calculation
            all_actual.extend(actual_values)
            all_predicted.extend(predicted_values)

        # Calculate the aggregate MSE across all columns
        aggregate_mse = np.mean(list(mse_scores.values()))

        # Calculate consolidated MSE across all columns
        consolidated_mse = mean_squared_error(all_actual, all_predicted)

        return results, mse_scores, aggregate_mse, consolidated_mse

# Usage example would remain the same, but now you also retrieve aggregate_mse and consolidated_mse from the fit_predict method.
