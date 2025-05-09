'''
This class use VARIMA from darts and set the parameters to ensure the VAR functionality
VAR = VARIMA (p=lag, d=0, q=0) # d,q=0 by default
'''
from BaseModel import BaseModel
from darts.models import ARIMA
from darts import TimeSeries
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from ForecastMetrics import ForecastMetrics
from darts.models import VARIMA
class VARModel(BaseModel):


    ''' This model does not have fit_predict, However rolling_window_evaluation can be changed to adapt the functionality of fit_predict method, like in ARIMAModel.py
    If needed it can be implemented here as well. 
    '''
    
    def rolling_window_evaluation(self):
        train_start,train_end=self.train_
        val_start,val_end=self.val_
        test_start,test_end=self.test_

        results = {}
        mse_scores = []
        all_actuals = []
        all_forecasts = []
        
        # Split data into train and test based on predefined dates (can be set in dataset_info)
        series = TimeSeries.from_dataframe(self.data, self.dataset_info['date_col'], fill_missing_dates=True) #, freq='10T')
        
        num_windows = (test_end-test_start) - self.horizon + 1
        step_size = self.getStepSize(num_windows)
        print ("num_windows,step_size: ", num_windows,step_size)
            
        column_actuals = []
        column_forecasts = []

        df_actual_list=[]
        df_predicted_list=[]
        
        # Perform rolling window predictions
        for start in range(num_windows):
            
            if ( (self.dataset_info['name'] == "Illness") and (start % 10 !=0)  ) or  (start % step_size != 0)  :
                continue
            
            
            train = series[:val_end + 1 + start]   
            #test = series[test_start + 1 + start:]
            test = series[test_start + 1 + start + self.lag:] 
            
            print ("train.shape, test.shape", train.n_samples, train.n_timesteps, train.n_components )
            
            #train_series = train[column]
            #test_series = test[column]
            
            model = VARIMA(p=self.lag,
                           d=0)
            print ("model initialised")
            model.fit(train)
            print ("model fitted")

            
            prediction = model.predict(self.horizon)
            predicted_values = pd.DataFrame(prediction.values(), columns=self.usable_cols)
            
            
            end = start + self.horizon
            test_slice = test[:self.horizon]
            actual_values = pd.DataFrame(test_slice.values(), columns=self.usable_cols)
            
            #print ("!!!!!!!!!!!!@@@@@@@@@@@@@@################  ------------------- prediction", prediction.values)
            #print ("!!!!!!!!!!!!@@@@@@@@@@@@@@################  ------------------- actuals", actual_values.values) 
            
            df_predicted_list.append(predicted_values)
            df_actual_list.append(actual_values)

            if len (actual_values) != self.horizon:
                continue

            results[str(start)] = predicted_values.values
            
            self.metrics = ForecastMetrics(actual_values.values, predicted_values.values, actual_values.values, predicted_values.values )
            #mse = mean_squared_error(column_actuals, column_forecasts)
            metrics_ = self.metrics.normalised_metrics()
            mse = metrics_["MSE"]
            print ("mse:", mse)
            
            mse_scores.append(mse)
            
            
        all_actuals = pd.concat(df_actual_list) 
        all_forecasts = pd.concat(df_predicted_list)
        
        aggregate_mse = np.mean(mse_scores)
        
        ''' need to fix this for scaled dataframe''' 
        df_all_actuals = self.rescale_dataframe(all_actuals, self.usable_cols)
        df_all_forecasts = self.rescale_dataframe(all_forecasts, self.usable_cols)
        
        
        self.cons_metrics = ForecastMetrics(all_actuals, all_forecasts,
                                            df_all_actuals, df_all_forecasts)
        cons_metrics_ = self.cons_metrics.calculate_all_metrics()
        consolidated_mse = cons_metrics_["MSE"] 
        print ("cons_metrics_: ", cons_metrics_)
        return results, mse_scores, aggregate_mse, consolidated_mse, cons_metrics_
    

