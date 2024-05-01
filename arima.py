## To do
# convert it into class so that it can be called for each dataset
# model config not needed here
# 
# train, vali, test
# metric
# record train time
# record inference time
# record memory? How?


import pandas as pd 
from darts.models import ARIMA
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts import TimeSeries 

## local packages

from ModelConfig import ModelConfig
from DatasetConfig import DatasetConfig

## read the model config 
path_to_yaml_file = 'runschema.yaml'
config = ModelConfig(path_to_yaml_file, 'AR')
print(config)


## read the dataset config
datasetconfig = DatasetConfig('datasetschema.yaml')
print (datasetconfig.datasets["ETTh2"])
#dataset_info = datasetconfig.get_dataset_info('Exchange')
#print(dataset_info)

dataset = datasetconfig.datasets["ETTh2"]
    
lag=dataset["lag"]
horizon=dataset["horizon"]
datasetpath=dataset["dataset_path"]
date_col=dataset["date_col"]




df = pd.read_csv(datasetpath)
columns = df.columns
usable_cols = list(set(columns) - set([date_col]))


for icol in usable_cols:

    series = TimeSeries.from_dataframe(df,date_col, icol)
    model = ARIMA(p=lag)
    model.fit(series)
    pred = model.predict(horizon)
    print (icol, ": ", pred.values())

print (series.values())


import pandas as pd
from darts.models import ARIMA
from darts import TimeSeries

class ARIMAModel:
    def __init__(self, dataset_info):
        self.dataset_info = dataset_info
        self.lag          =
        self.horizon      = 

    def load_data(self):
        df = pd.read_csv(self.dataset_info['dataset_path'])
        self.columns = df.columns
        self.usable_cols = list(set(self.columns) - set([self.dataset_info['date_col']]))

    def fit_predict(self):
        results = {}
        for icol in self.usable_cols:
            series = TimeSeries.from_dataframe(df, self.dataset_info['date_col'], icol)
            model = ARIMA(p=self.dataset_info['lag'])
            model.fit(series)
            prediction = model.predict(self.dataset_info['horizon'])
            results[icol] = prediction.values()
        return results
