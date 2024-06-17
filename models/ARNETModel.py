from BaseModel import BaseModel
from ForecastMetrics import ForecastMetrics

import neuralprophet
import torch 
from neuralprophet import NeuralProphet, set_log_level
from neuralprophet import set_random_seed 
set_log_level("ERROR", "INFO")

import numpy as np
set_random_seed(0)
import pandas as pd
from sklearn.preprocessing import StandardScaler

class ARNETModel(BaseModel):
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

        '''
        The training data for NP and other deep-learning model would be just training data
        it will not use vlaidation data.
        ''' 
        test_len =  test_end-test_start
        num_windows =  test_len - self.horizon + 1
        print (" val_end, test_len,num_windows: ", val_end, test_len,num_windows)

        step_size = self.getStepSize(num_windows)

        df_actual_list=[]
        df_predicted_list=[]

        
        train_df =  df[train_start:train_end] ## notice change here w.r.t statistical models. Validate that this is correct, train, val and test would be fix 
        val_df   = df[val_start:val_end]
        test_df =  df[test_start:test_end]
        
        # Convert wide format to long format for NP consumption, we need to deflate both train and val outside loop and test inside the loop 
        train_long_df = self.deflate_dataframe_NP(train_df)
        val_long_df   = self.deflate_dataframe_NP(val_df)

        m = NeuralProphet( #growth='off',
                           #yearly_seasonality=False,
                           #weekly_seasonality=False,
                           #daily_seasonality=False,
            #n_lags=self.lag,
            n_forecasts=self.horizon,
            trainer_config={"accelerator": "mps", "devices":1}
        )                                                                                                                                                                             
        print ("neural prophet instance created ")
        
        metrics = m.fit(train_long_df,validation_df=train_long_df,freq='auto',progress=None)
        
        print ("NP model fitted to train data")
        
        for start in range(num_windows):
            if ( (self.dataset_info['name'] == "Illness") and (start % 10 !=0)  ) or  (start % step_size != 0)  :
                continue
            
            test_long_df =  self.deflate_dataframe_NP ( df[test_start + 1+start: test_start + 1+start+self.lag]  ) # it should be start point to start point + lag; this is the data we want to send to model to make the next prediction from whatever it has learned during the traning

            
            df_future = m.make_future_dataframe(test_long_df, n_historic_predictions=True, periods=self.horizon)
            forecast = m.predict(df_future)
            
            forecast_skim = forecast[["ds","ID","yhat1"]]
            
            deflate_df = self.widen_dataframe(forecast_skim, self.usable_cols,
                                              True,
                                              self.lag)
            
            
            actual_df = df[test_start + 1+start+self.lag : test_start + 1+start+self.lag + self.horizon] [self.usable_cols]
            
            print ("lagged df: ", df[test_start + 1+start: test_start + 1+start+self.lag])
            print ("horizon df: ", actual_df )
            print ("deflate_df: ", deflate_df)

            
            
            if len (actual_df.values) != self.horizon:
                continue

            all_actuals.append(actual_df)
            all_forecasts.append(deflate_df)
            
            self.metrics = ForecastMetrics(actual_df.values, deflate_df.values,
                                           actual_df.values, deflate_df.values)

            metrics_ = self.metrics.normalised_metrics()
            mse = metrics_["MSE"]
            
            print (f"mse for {start} is: {mse}")
            mse_scores.append(mse)

            
            deflate_df_scaled = self.rescale_dataframe(deflate_df, self.usable_cols)
            actual_df_scaled  = self.rescale_dataframe(actual_df, self.usable_cols)
            df_predicted_list.append(deflate_df_scaled)
            df_actual_list.append(actual_df_scaled)

            results[str(start)] = deflate_df.values
            
        ## end of for loop
            
        df_all_actuals = pd.concat(df_actual_list)
        df_all_predicted = pd.concat(df_predicted_list)

        df_all_actuals_unNorm = pd.concat(all_actuals)
        df_all_predicted_unNorm = pd.concat(all_forecasts)
        

        self.cons_metrics = ForecastMetrics(df_all_actuals_unNorm.values,
                                            df_all_predicted_unNorm.values,
                                            df_all_actuals,
                                            df_all_predicted
                                            )
        #cons_metrics_ = self.cons_metrics.normalised_metrics()                                                                                                                                                       
        cons_metrics_ = self.cons_metrics.calculate_all_metrics()
        #consolidated_mse = cons_metrics_["MSE"]                                                                                                                                                                      

        ''' following consolidated_mse is for the case when ONLY mse is calculated, but now I calculate all metrics so I don't need this'''
        consolidated_mse = cons_metrics_["MSE"]
        aggregate_mse = np.mean(mse_scores)

        return results, mse_scores, aggregate_mse, consolidated_mse, cons_metrics_

            
