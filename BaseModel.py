import pandas as pd
from sklearn.preprocessing import StandardScaler
from darts import TimeSeries

import sys
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.Print_time_boundaries import DataSplitter

class BaseModel:
    def __init__(self, dataset_info, scale_data=True):
        self.dataset_info = dataset_info
        self.data = None
        self.scale_data = scale_data
        self.scaler = StandardScaler() if scale_data else None
        self.train_=None
        self.val_=None
        self.test_=None
        
    def load_data(self):
        #self.data = pd.read_csv(self.dataset_info['cloud_path'])
        self.data = pd.read_csv(self.dataset_info['dataset_path'])

        ## get the time boundaries for train, val, test
        splitter = DataSplitter(len(self.data), self.dataset_info['name']  )
        self.train_, self.val_, self.test_ = splitter.get_train_val_test_ranges()
        #print (self.train_, self.val_, self.test_)
        
        self.data[self.dataset_info['date_col']] = pd.to_datetime(self.data[self.dataset_info['date_col']])
        self.usable_cols = list(set(self.data.columns) - {self.dataset_info['date_col']})

        if self.scale_data:
            self.apply_scaling()

    def apply_scaling(self):
        # Assuming the 'train' key in dataset_info contains the start and end dates for the training period
        #train_start, train_end = self.dataset_info['train'] ## using yaml 
        train_start, train_end = self.train_  ## by passing yaml to ease the book-keeping 
        print (type(train_start), train_end)

        
        
        #train_mask = (self.data[self.dataset_info['date_col']] >= train_start) & \
        #    (self.data[self.dataset_info['date_col']] <= train_end)

        # Fit the scaler on training data only using iloc for row indexing
        self.scaler.fit(self.data.iloc[train_start:train_end + 1][self.usable_cols])

        # Fit the scaler on training data only
        #self.scaler.fit(self.data.loc[train_mask, self.usable_cols])

        # Apply the scaler to all the data
        self.data[self.usable_cols] = self.scaler.transform(self.data[self.usable_cols])

