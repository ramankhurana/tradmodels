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

    
    def rolling_window_evaluation(self):
        train_start,train_end=self.train_
        val_start,val_end=self.val_
        test_start,test_end=self.test_

        results = {}
        mse_scores = {}
        all_actuals = []
        all_forecasts = []
        
        # Split data into train and test based on predefined dates (can be set in dataset_info)
        series = TimeSeries.from_dataframe(self.data, self.dataset_info['date_col'], fill_missing_dates=True) #, freq='10T')
        
        # Loop over each usable column
        for column in self.usable_cols:
            print ("column name", column)

            
            num_windows = (test_end-test_start) - self.horizon + 1
            step_size = self.getStepSize(num_windows)
            print ("num_windows,step_size: ", num_windows,step_size)
            
            column_actuals = []
            column_forecasts = []
            
            # Perform rolling window predictions
            for start in range(num_windows):

                if ( (self.dataset_info['name'] == "Illness") and (start % 10 !=0)  ) or  (start % step_size != 0)  :
                    continue

                
                train = series[:val_end + 1 + start]   
                #test = series[test_start + 1 + start:]
                test = series[test_start + 1 + start + self.lag:]

                print ("train.shape, test.shape", train.n_samples, train.n_timesteps, train.n_components )
                
                train_series = train[column]
                test_series = test[column]

                model = ARIMA(p=self.lag,
                              d=0)
                model.fit(train_series)
                
                prediction = model.predict(self.horizon)
                predicted_values = prediction.values().flatten()

                
                end = start + self.horizon
                test_slice = test_series[:self.horizon]
                actual_values = test_slice.values().flatten()

                print ("!!!!!!!!!!!!@@@@@@@@@@@@@@################  ------------------- prediction", predicted_values)
                print ("!!!!!!!!!!!!@@@@@@@@@@@@@@################  ------------------- actuals", actual_values)

                if len (actual_values) != self.horizon:
                    continue

                column_actuals.extend(actual_values)
                column_forecasts.extend(predicted_values)

            self.metrics = ForecastMetrics(column_actuals, column_forecasts,column_actuals, column_forecasts)
            

            
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
        df_all_actuals = self.widen_and_rescale_dataframe(all_actuals, self.usable_cols)
        df_all_forecasts = self.widen_and_rescale_dataframe(all_forecasts, self.usable_cols)
        
        
        self.cons_metrics = ForecastMetrics(all_actuals, all_forecasts,df_all_actuals.values, df_all_forecasts.values)
        cons_metrics_ = self.cons_metrics.calculate_all_metrics()
        consolidated_mse = cons_metrics_["MSE"] 

        return results, mse_scores, aggregate_mse, consolidated_mse, cons_metrics_


