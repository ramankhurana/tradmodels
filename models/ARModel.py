'''
The difference between AR and ARIMA is only differencing at this moment.
AR = ARIMA(lag,d=0)
ARIMA = ARIMA(lag,d=1)
in both cases MA is turned off for the moment as it was not improving the MSE score for a few datasets I tried, ETT and Divvy 
'''

from BaseModel import BaseModel
from darts.models import ARIMA
from darts import TimeSeries
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from ForecastMetrics import ForecastMetrics
class ARModel(BaseModel):

    def fit_predict(self):
        results = {}
        mse_scores = {}
        all_actual = []
        all_predicted = []
        self.lag = self.dataset_info['lag']
        self.horizon = self.dataset_info['horizon']

        ## from yaml file
        #train_start,train_end=self.dataset_info["train"]
        #val_start,val_end=self.dataset_info["val"]
        
        train_start,train_end=self.train_
        val_start,val_end=self.val_
        test_start,test_end=self.test_
                
        
        for column in self.usable_cols:
            series = TimeSeries.from_dataframe(self.data, self.dataset_info['date_col'], column,
                                               fill_missing_dates=True)# , freq='10T')
            #train, test = series.split_after(pd.Timestamp(self.dataset_info['test'][0]))

            # Splitting based on indices rather than dates
            train = series[:train_end + 1]  # includes the train_end index
            test = series[val_end + 1:]  # starts just after train_end

            print ("----------------------",train.shape, test.shape)
            
            model = ARIMA(p=self.dataset_info['lag'],
                          d=0)
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

        ## from yaml file
        #train_start,train_end=self.dataset_info["train"]
        #val_start,val_end=self.dataset_info["val"]

        
        train_start,train_end=self.train_
        val_start,val_end=self.val_
        test_start,test_end=self.test_

        results = {}
        mse_scores = {}
        all_actuals = []
        all_forecasts = []
        
        # Split data into train and test based on predefined dates (can be set in dataset_info)
        series = TimeSeries.from_dataframe(self.data, self.dataset_info['date_col'], fill_missing_dates=True) #, freq='10T')
        #train, test = series.split_after(pd.Timestamp(self.dataset_info['test'][0]))
        
        # Splitting based on indices rather than dates
        ''' following is likley buggy when compared to TimeGPT or others'''
        #train = series[:train_end + 1]  # includes the train_end index
        #test = series[val_end + 1:]  # starts just after train_end

        '''this is the fixed one; to be tested'''
        train = series[:val_end + 1]  ## train should include the validation as well otherwise there is unseen data which is equal to val data length. 
        test = series[test_start + 1:] 
        
        # Loop over each usable column
        for column in self.usable_cols:
            print ("column name", column)
            train_series = train[column]
            test_series = test[column]
            model = ARIMA(p=self.lag,
                          d=0)
            #print ("size of train series: ", len(train_series))
            #print(train_series)
            model.fit(train_series)

            num_windows = len(test_series) - self.horizon + 1
            column_actuals = []
            column_forecasts = []

            #print ("num_windows ", num_windows)
            #print ("mean train, test: ",column, train_series.mean(axis=0).pd_dataframe().iloc[0, 0], test_series.mean(axis=0).pd_dataframe().iloc[0, 0])
            
            # Perform rolling window predictions
            for start in range(num_windows):
                end = start + self.horizon
                test_slice = test_series[start:end]

                prediction = model.predict(self.horizon)
                actual_values = test_slice.values().flatten()
                predicted_values = prediction.values().flatten()

                column_actuals.extend(actual_values)
                column_forecasts.extend(predicted_values)

            self.metrics = ForecastMetrics(column_actuals, column_forecasts,column_actuals, column_forecasts)
            
            print ("metric calculation using the class: ", self.metrics.normalised_metrics()) 
            
            #mse = mean_squared_error(column_actuals, column_forecasts)
            metrics_ = self.metrics.normalised_metrics()
            mse = metrics_["MSE"]
            print ("mse:", mse)
            mse_scores[column] = mse
            results[column] = prediction

            all_actuals.extend(column_actuals)
            all_forecasts.extend(column_forecasts)

        aggregate_mse = np.mean(list(mse_scores.values()))
        #consolidated_mse = mean_squared_error(all_actuals, all_forecasts)

        ''' scale the original and forecasted back to un-normalised values '''
        unnormalised = all_actuals, all_forecasts
        self.cons_metrics = ForecastMetrics(all_actuals, all_forecasts,all_actuals, all_forecasts)
        cons_metrics_ = self.cons_metrics.normalised_metrics()
        consolidated_mse = cons_metrics_["MSE"] 

        return results, mse_scores, aggregate_mse, consolidated_mse, cons_metrics_


