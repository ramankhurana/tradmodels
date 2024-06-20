'''

Nomenclature to be used
all_actuals=[] : for list of all actual dataframes
all_forecasts=[]: for list of all forecasted dataframes

df_all_actuals: all actual dataframes; after standard scaler transformation 
df_all_forecasts: all forecasted dataframes; after standard scaler transformation

df_all_actuals_unnorm: all actual dataframes; after inverse transform 
df_all_forecasts_unnorm:  all forecast dataframes; after inverse transform


'''

import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from BaseModel import BaseModel
from ForecastMetrics import ForecastMetrics



class ChronosModel(BaseModel):
    def getquantiles(self,pred):
        return pred

    def to_tensor_list(self,df):
        tensor_list =  [ torch.tensor(df[icol].values)  for icol in self.usable_cols]
        return tensor_list
    
    def rolling_window_evaluation(self):
        self.lag = self.dataset_info['lag']
        self.horizon = self.dataset_info['horizon']
        self.freq = 'W'
        self.time_col = self.dataset_info['date_col']
        
        
        df = self.data
        train_start,train_end=self.train_
        val_start,val_end=self.val_
        test_start,test_end=self.test_
        
        results = {}
        mse_scores = []
        all_actuals = []
        all_forecasts = []
        
        test_len =  test_end-test_start
        num_windows =  test_len - self.horizon + 1
        #print (" val_end, test_len,num_windows: ", val_end, test_len,num_windows)

        step_size = self.getStepSize(num_windows)

        df_actual_list=[]
        df_predicted_list=[]


        train_df =  df[train_start:val_end] ## should be same like stats; or TimeGPT model
        test_df =  df[test_start:test_end]
        

        train_col_list = [ torch.tensor(train_df[icol])  for icol in self.usable_cols]
        #print ("train_col_list: ", train_col_list)
        
        test_col_list  = [ torch.tensor(train_df[icol]) for icol in self.usable_cols]
        #print ("test_col_list: ", test_col_list)

        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny",
            #device_map="cuda",
            #device_map="mps",
            torch_dtype=torch.bfloat16,
            
        )
        
        for start in range(num_windows):
            if ( (self.dataset_info['name'] == "Illness") and (start % 10 !=0)  ) or  (start % step_size != 0)  :
                continue

            train = df[:val_end + 1 + start]
            test = df[test_start + 1 + start:test_start + 1 + start+self.horizon][self.usable_cols]
            
            #print ("shapes: ", train.shape, test.shape)
        
            train_tensor_list = self.to_tensor_list ( train )
            test_tensor_list  = self.to_tensor_list ( test )

            context           = train_tensor_list
            prediction_length = self.horizon

            #print ("type before forecast: ", type(context) )
            forecast = pipeline.predict(context, prediction_length)

            #print ("forecast ", forecast)
            #print ("type(forecast): ", type(forecast), forecast.shape )

            forecast_median = np.quantile(forecast, 0.5, axis=1)
            forecast_numpy = torch.tensor(forecast_median).permute(1,0).numpy()
            
            df_forecast = pd.DataFrame(forecast_numpy, columns=self.usable_cols)
            
            if len (test.values) != self.horizon:
                continue

            
            all_actuals.append(df_forecast)
            all_forecasts.append(test)

            print (test.values.shape, df_forecast.values.shape)
            self.metrics = ForecastMetrics(test.values, df_forecast.values,
                                           test.values, df_forecast.values)
            
            metrics_ = self.metrics.normalised_metrics()
            mse = metrics_["MSE"]

            print (f"mse for {start} is: {mse}")
            mse_scores.append(mse)
            results[str(start)] = df_forecast.values



        df_all_actuals = pd.concat(all_actuals)
        df_all_forecasts = pd.concat(all_forecasts)

        
        
        
        df_all_actuals_unnorm = self.rescale_dataframe(df_all_actuals, self.usable_cols)
        df_all_forecasts_unnorm = self.rescale_dataframe(df_all_forecasts, self.usable_cols)


        self.cons_metrics = ForecastMetrics(df_all_actuals.values,
                                            df_all_forecasts.values,
                                            df_all_actuals_unnorm.values,
                                            df_all_forecasts_unnorm.values
                                            )
        cons_metrics_ = self.cons_metrics.calculate_all_metrics()
        ''' following consolidated_mse is for the case when ONLY mse is calculated, but now I calculate all metrics so I don't need this'''
        consolidated_mse = cons_metrics_["MSE"]
        aggregate_mse = np.mean(mse_scores)

        return results, mse_scores, aggregate_mse, consolidated_mse, cons_metrics_
