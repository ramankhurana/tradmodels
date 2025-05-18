from BaseModel import BaseModel
from darts.models import ARIMA
from darts import TimeSeries
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from ForecastMetrics import ForecastMetrics
class ARIMAModel(BaseModel):

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

            
            train = series[:train_end + 1]  # includes the train_end index
            test = series[val_end + 1:]  # starts just after train_end

            print ("----------------------",train.shape, test.shape)
            
            model = ARIMA(p=self.dataset_info['lag'], d=1, q=0)
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
        train_start,train_end=self.train_
        val_start,val_end=self.val_
        test_start,test_end=self.test_

        results = {}
        mse_scores = {}
        all_actuals = []
        all_forecasts = []
        
        # Split data into train and test based on predefined dates (can be set in dataset_info)
        series = TimeSeries.from_dataframe(self.data, self.dataset_info['date_col'], fill_missing_dates=True) #, freq='10T')
        
        # Loop over each usable column because that is how ARIMA works. 
        for column in self.usable_cols:
            print ("column name", column)

            print ("train_start,train_end,val_start,val_end,test_start,test_end: ",train_start,train_end,val_start,val_end,test_start,test_end)
            num_windows = (test_end-test_start) - self.horizon + 1
            step_size = self.getStepSize(num_windows)
            print ("num_windows,step_size: ", num_windows,step_size)
            
            column_actuals = []
            column_forecasts = []
            
            # Perform rolling window predictions for each column i.e. each time series. 
            for start in range(num_windows):
            #for start in range(1):

                if ( (self.dataset_info['name'] == "Illness") and (start % 10 !=0)  ) or  (start % step_size != 0)  :
                    continue

                print ("start, test_start + 1 + start+self.lag ", start, test_start + 1 + start + self.lag)
                train = series[:val_end + 1 + start]   
                # test = series[test_start + 1 + start:] ## this leads to data leakage. Forecasting horizon is already used in the training of ARIMA model. 
                test = series[test_start + 1 + start + self.lag:]
                

                print ("train.shape, test.shape", train.n_samples, train.n_timesteps, train.n_components, test.n_samples, test.n_timesteps, test.n_components )
                
                train_series = train[column]
                test_series = test[column]

                model = ARIMA(p=self.lag, d=1, q=0)
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



    def rolling_window_multi_horizon_evaluation(self):
        horizons = self.horizon
        train_start, train_end = self.train_
        val_start, val_end = self.val_
        test_start, test_end = self.test_
    
        results = {}         # results[column][window_index][horizon] = prediction
        mse_scores = {}      # mse_scores[column][horizon] = aggregated MSE
        all_results = {}     # all_results[column][horizon] = (all_actuals, all_forecasts)
    
        series = TimeSeries.from_dataframe(
            self.data, self.dataset_info['date_col'], fill_missing_dates=True
        )
    
        for column in self.usable_cols:
            print(f"Processing column: {column}")
            results[column] = {}
            mse_scores[column] = {}
            all_results[column] = {h: ([], []) for h in horizons}
    
            max_horizon = max(horizons)
            num_windows = (test_end - test_start) - max_horizon + 1
            step_size = self.getStepSize(num_windows)
            print("num_windows:", num_windows, "step_size:", step_size)
    
            window_counter = 0
            for start in range(num_windows):
                if ((self.dataset_info['name'] == "Illness") and (start % 10 != 0)) or (start % step_size != 0):
                    continue
                print ("----- fitting for window number: ", start)
                
                train = series[:val_end + 1 + start]
                test = series[test_start + 1 + start + self.lag:]
    
                train_series = train[column]
                test_series = test[column]
    
                model = ARIMA(p=self.lag, d=1,q=0)
                model.fit(train_series)
    
                results[column][window_counter] = {}
    
                for horizon in horizons:
                    if test_series.n_timesteps < horizon:
                        continue  # Skip if not enough future data
    
                    prediction = model.predict(horizon)
                    predicted_values = prediction.values().flatten()
    
                    actual_values = test_series[:horizon].values().flatten()
                    if len(actual_values) != horizon:
                        continue
    
                    # Save prediction for this window & horizon
                    results[column][window_counter][horizon] = prediction
    
                    # Store for metrics
                    all_results[column][horizon][0].extend(actual_values)  # actuals
                    all_results[column][horizon][1].extend(predicted_values)  # forecasts
    
                window_counter += 1
    
            # Calculate MSE for each horizon for this column
            for horizon in horizons:
                actuals, forecasts = all_results[column][horizon]
                if len(actuals) > 0:
                    mse = mean_squared_error(actuals, forecasts)
                    mse_scores[column][horizon] = mse
    
        # Consolidated metrics across all columns per horizon
        consolidated_metrics = {}
        for horizon in horizons:
            total_actuals = []
            total_forecasts = []
    
            for column in self.usable_cols:
                actuals, forecasts = all_results[column][horizon]
                total_actuals.extend(actuals)
                total_forecasts.extend(forecasts)
    
            if total_actuals:
                df_actuals = self.widen_and_rescale_dataframe(total_actuals, self.usable_cols)
                df_forecasts = self.widen_and_rescale_dataframe(total_forecasts, self.usable_cols)
    
                cons_metrics = ForecastMetrics(total_actuals, total_forecasts, df_actuals.values, df_forecasts.values)
                consolidated_metrics[horizon] = cons_metrics.calculate_all_metrics()
    
        print ("results: ",results)
        print ("mse_scores: ", mse_scores)
        print ("consolidated_metrics: ",consolidated_metrics)
        
        return results, mse_scores, consolidated_metrics
    
