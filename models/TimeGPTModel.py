'''
TimeGPT takes unrolled/stacked dataframe as input, similar to NeuralProphet
Since all columns are stacked and all are forecasted in a single go, it has to be processed differently from ARIMA.
First for a given window it needs to be converted to dataframe, and then concatinated such that all values for each column are placed correctly.

There are two objects for ForecastingMetric class
1. For MSE.
2. For consolodated MSE, in general this is the number we need and hence all metrics are saved in csv file for this setting.
'''

from BaseModel import BaseModel
from nixtlats import TimeGPT
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import math 
from ForecastMetrics import ForecastMetrics
import os

api_token = os.getenv('TIMEGPT_API_TOKEN')
class TimeGPTModel(BaseModel):
    
    def __init__(self, dataset_info):
        super().__init__(dataset_info)
        
        self.token = api_token # set in the backrc environmnet '8ffEwwwvGFnUuJYMygD4xeTZwzprSxlOtXzhnYwNHPvIc0PwP6KPSFTFUuJzxYjuScZa7s2YpOISVjKrVfoIrwAimKpOAwSA5cl1mZ03nwDGY8FfwhnLj4jwsTc55joRN7HqbYIfyeIQfLVqoYHfGl9etgLYw5PTwPK2abfSXnKozhqVN2Z99iFhO9TvcQQRtGzIWTDrsrsAFub1PYKAvTD4knk51C6dOD5MFNTCQNxmolLRBBrYzNfccIfDRaSL'  # Insert your actual TimeGPT token
        self.timegpt = TimeGPT(token=self.token)
        self.timegpt.validate_token()
        print ("token validated ")
        

    def fit_predict(self):

        self.lag = self.dataset_info['lag']
        self.horizon = self.dataset_info['horizon']
        self.freq = 'H'
        self.time_col = self.dataset_info['date_col']

        train_start,train_end=self.train_
        val_start,val_end=self.val_

        # Assuming data is already loaded and prepared
        df = self.data
        target_cols = self.usable_cols

        ## originally this was used but it will make a hole b/w train and test, as there is no val used in such scenario.
        ## the best in this case is to use the train+val as input and  make forecast. 
        #train = df[:train_end + 1]
        #test  = df[val_end + 1:]

        train = df[:val_end + 1]
        test  = df[val_end + 1:]

        #print (df.head(), df.shape)

        train = self.deflate_dataframe(train)


        '''
        print ("shape", train.shape, test.shape)
        print ("head train", train.head())
        print ("head test",  test.head())
        print ("tail train",  train.tail())
        '''

                
        tgpt_results = self.timegpt.forecast(df=train, h=self.horizon, freq=self.freq, 
                                             time_col='ds', target_col='y')
        '''
        print ("time gpt forecasted", tgpt_results.shape)
        print (tgpt_results.head())
        '''
        forecasted_values = tgpt_results 
        
        actual_values = self.deflate_dataframe( test[:self.horizon] )

        # Calculate MSE
        ''' In time GPT case or multivariate case in general, mse, agg mse and consolated mse are same, as they are calculated using same formula
        the difference is needed when underlying model is univariate'''
        mse = mean_squared_error(actual_values["y"].values, forecasted_values["TimeGPT"].values)#, multioutput='raw_values')
        aggregate_mse = np.mean(mse)
        consolidated_mse = mean_squared_error(actual_values["y"].values, forecasted_values["TimeGPT"].values)

        return tgpt_results, mse, aggregate_mse, consolidated_mse

    def rolling_window_evaluation(self):
        self.lag = self.dataset_info['lag']
        self.horizon = self.dataset_info['horizon']
        self.freq = 'W'
        self.time_col = self.dataset_info['date_col']

        df = self.data
        train_start,train_end=self.train_
        val_start,val_end=self.val_
        test_start,test_end=self.test_
        
        print ("train_start,train_end,val_start,val_end,test_start,test_end", train_start,train_end,val_start,val_end,test_start,test_end)
        results = {}
        mse_scores = []
        all_forecasts = []
        all_actuals = []

        
        test_len =  test_end-test_start
        num_windows =  test_len - self.horizon + 1
        print (" val_end, test_len,num_windows: ", val_end, test_len,num_windows)
        
        step_size = 50

        if num_windows>=500:
            step_size = 25 + int( (num_windows-500 )*0.05 )
        if num_windows>=400 and num_windows < 500:
            step_size = 20 
        if num_windows>=300 and	num_windows < 400:
            step_size = 15 
        if num_windows>=200 and	num_windows < 300:
            step_size = 10 
        if num_windows>=100 and	num_windows < 200:
            step_size = 5 
        if num_windows>=50 and	num_windows < 100:
            step_size = 3 
        if num_windows < 50:
            step_size = 1 
        
        print ("num_windows, step_size ",num_windows, step_size)


        df_actual_list=[]
        df_predicted_list=[]

        
        for start in range(num_windows):

        
            if ( (self.dataset_info['name'] == "Illness") and (start % 10 !=0)  ) or  (start % step_size != 0)  :
                #print ("start: ", start)
                continue 
            train_df =  df[:val_end + 1+start] 
            test_df =  df[test_start + 1+start:]
            
            # Convert wide format to long format for TimeGPT consumption, we need to deflate both train and test in the loop.
            train_long_df = self.deflate_dataframe(train_df[-100:])  ## initialise with 100 timestamps; first extract 100 timestamp and then deflate
            
            if self.dataset_info['name']=="Illness":
                train_long_df = self.deflate_dataframe(train_df[-600:])
                
            if self.dataset_info['name']=="M5":
                train_long_df = self.deflate_dataframe(train_df[-300:])
                
            else:
                train_long_df = self.deflate_dataframe(train_df[-1000:])


            print ("size of train_long_df: ", train_long_df.shape)
            print ("tail of df: ",train_long_df.tail)
            # Perform forecasting with TimeGPT
            forecasted_values = self.timegpt.forecast(df=train_long_df, h=self.horizon,
                                                 model='timegpt-1-long-horizon',
                                                 time_col='ds', target_col='y')             # freq=self.freq,  if frequency can not be inferred


            #forecasted_values = tgpt_results
            ## Remeber to first slice the dataframe for mse actual and then deflate, otherwise it will not be correctly alligned. 
            actual_values = self.deflate_dataframe(test_df[:self.horizon])

            print ("******************************", actual_values.shape, forecasted_values.shape)
            
            self.metrics = ForecastMetrics(actual_values["y"].values,forecasted_values["TimeGPT"].values,
                                           actual_values["y"].values, forecasted_values["TimeGPT"].values)
            
            
            metrics_ = self.metrics.normalised_metrics()
            mse = metrics_["MSE"]
            
            print (f"mse for {start} is: {mse}")
            mse_scores.append(mse)
            all_forecasts.extend(forecasted_values["TimeGPT"].values)
            all_actuals.extend(actual_values["y"].values)
            
            ## this has to be done within the loop such that for each iteration it is converted to dataframe and can be concat later on. 
            df_predicted = self.widen_and_rescale_dataframe(forecasted_values["TimeGPT"].values, self.usable_cols)
            df_actuals   = self.widen_and_rescale_dataframe(actual_values["y"].values,  self.usable_cols)
            
            df_actual_list.append(df_actuals)
            df_predicted_list.append(df_predicted)
            
            
                        
        # Calculate overall MSEs
        df_all_actuals = pd.concat(df_actual_list)
        df_all_predicted = pd.concat(df_predicted_list)

        self.cons_metrics = ForecastMetrics(np.array(all_actuals),
                                            np.array(all_forecasts),
                                            df_all_actuals.values,
                                            df_all_predicted.values
                                            )
        #cons_metrics_ = self.cons_metrics.normalised_metrics()
        cons_metrics_ = self.cons_metrics.calculate_all_metrics()
        #consolidated_mse = cons_metrics_["MSE"]
        
        ''' following consolidated_mse is for the case when ONLY mse is calculated, but now I calculate all metrics so I don't need this'''
        consolidated_mse = mean_squared_error(np.array(all_actuals), np.array(all_forecasts))
        aggregate_mse = np.mean(mse_scores)

        return forecasted_values, mse_scores, aggregate_mse, consolidated_mse, cons_metrics_
    
