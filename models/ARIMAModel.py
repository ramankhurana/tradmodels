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
        self.lag = self.dataset_info['lag']
        self.horizon = self.dataset_info['horizon']
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


    def rolling_window_evaluation(self):
        self.lag = self.dataset_info['lag']
        self.horizon = self.dataset_info['horizon']
        results = {}
        mse_scores = {}
        all_actuals = []
        all_forecasts = []

        # Split data into train and test based on predefined dates (can be set in dataset_info)
        series = TimeSeries.from_dataframe(self.data, self.dataset_info['date_col'])
        train, test = series.split_after(pd.Timestamp(self.dataset_info['test'][0]))

        # Loop over each usable column
        for column in self.usable_cols:
            train_series = train[column]
            test_series = test[column]
            model = ARIMA(p=self.lag)
            model.fit(train_series)

            num_windows = len(test_series) - self.horizon + 1
            column_actuals = []
            column_forecasts = []

            # Perform rolling window predictions
            for start in range(num_windows):
                end = start + self.horizon
                test_slice = test_series[start:end]

                prediction = model.predict(self.horizon)
                actual_values = test_slice.values().flatten()
                predicted_values = prediction.values().flatten()

                column_actuals.extend(actual_values)
                column_forecasts.extend(predicted_values)

            mse = mean_squared_error(column_actuals, column_forecasts)
            print ("mse:", mse)
            mse_scores[column] = mse
            results[column] = prediction

            all_actuals.extend(column_actuals)
            all_forecasts.extend(column_forecasts)

        aggregate_mse = np.mean(list(mse_scores.values()))
        consolidated_mse = mean_squared_error(all_actuals, all_forecasts)

        return results, mse_scores, aggregate_mse, consolidated_mse
