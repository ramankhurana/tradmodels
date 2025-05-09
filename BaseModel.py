import pandas as pd
from sklearn.preprocessing import StandardScaler
from darts import TimeSeries

import sys
from ForecastMetrics import ForecastMetrics

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
        
        self.metrics = None
        self.cons_metrics = None

        ''' following are common for all models '''
        self.lag = self.dataset_info['lag']
        self.horizon = self.dataset_info['horizon']
        
        ''' common variable list over '''
        
    def load_data(self):
        #self.data = pd.read_csv(self.dataset_info['cloud_path'])
        self.data = pd.read_csv(self.dataset_info['dataset_path'])
        #[["OT","date","HULL"]]
        
        if self.dataset_info["name"] == "M5":
            # Generate a date range
            date_range = pd.date_range(start='2011-01-28', end='2016-04-23')
            print (date_range.shape)
            self.data['date'] = date_range
            

        ## get the time boundaries for train, val, test
        splitter = DataSplitter(len(self.data), self.dataset_info['name'] , self.dataset_info['lag'] )
        self.train_, self.val_, self.test_ = splitter.get_train_val_test_ranges()
        
        
        
        self.data[self.dataset_info['date_col']] = pd.to_datetime(self.data[self.dataset_info['date_col']])
        self.usable_cols = list(set(self.data.columns) - {self.dataset_info['date_col']})
        #self.usable_cols = ["OT", "HUFL"]
        self.usable_cols.sort()
        ''' this sorting is needed to make sure that the order is always same''' 
        
        if self.scale_data:
            self.apply_scaling()

    def apply_scaling(self):
        # Assuming the 'train' key in dataset_info contains the start and end dates for the training period
        #train_start, train_end = self.dataset_info['train'] ## using yaml 
        train_start, train_end = self.train_  ## by passing yaml to ease the book-keeping 
        print (type(train_start), train_end)
        
        
        
        #train_mask = (self.data[self.dataset_info['date_col']] >= train_start) & \
        #    (self.data[self.dataset_info['date_col']] <= train_end)
        
        
        ## before fitting let's remove duplicates first
        self.data = self.data.drop_duplicates(subset=[self.dataset_info['date_col']], keep='first')

        print (self.usable_cols)
        # Fit the scaler on training data only using iloc for row indexing
        self.scaler.fit(self.data.iloc[train_start:train_end + 1][self.usable_cols])
        
        # Fit the scaler on training data only
        #self.scaler.fit(self.data.loc[train_mask, self.usable_cols])
        
        # Apply the scaler to all the data
        self.data[self.usable_cols] = self.scaler.transform(self.data[self.usable_cols])
        #print ("------", self.data)
        
        

    def deflate_dataframe(self, df):
        """
        Transform a wide format DataFrame to a long format DataFrame where each row 
        represents a single time point for each ID.
        
        Parameters:
        df (pd.DataFrame): DataFrame to transform. Expected to have 'date' and other columns 
                       where each column name after 'date' represents a unique ID.
    
        Returns:
        pd.DataFrame: Transformed DataFrame in long format with columns 'unique_id', 'ds' (date), and 'y' (values).
        """
        # Ensure the date column is in the correct format (optional but recommended)
        df['date'] = pd.to_datetime(df['date'])
        
        # Melting the DataFrame
        # 'id_vars' is the column(s) to keep (not melt), everything else is considered a value var
        melted_df = df.melt(id_vars=['date'], var_name='unique_id', value_name='y')
        
        # Renaming the columns to match your output specification
        melted_df = melted_df.rename(columns={'date': 'ds'})
        
        return melted_df





    def deflate_dataframe_NP(self, df):
        """
        This implementation is slighly different from deflate_dataframe becuase Neural Prophet need spefic column names and hence needs to be hard-coded 
        Transform a wide format DataFrame to a long format DataFrame where each row 
        represents a single time point for each ID.
        
        Parameters:
        df (pd.DataFrame): DataFrame to transform. Expected to have 'date' and other columns 
                       where each column name after 'date' represents a unique ID.
    
        Returns:
        pd.DataFrame: Transformed DataFrame in long format with columns 'unique_id', 'ds' (date), and 'y' (values).
        """
        # Ensure the date column is in the correct format (optional but recommended)
        df['date'] = pd.to_datetime(df['date'])
        
        # Melting the DataFrame
        # 'id_vars' is the column(s) to keep (not melt), everything else is considered a value var
        melted_df = df.melt(id_vars=['date'], var_name='ID', value_name='y')
        
        # Renaming the columns to match your output specification
        melted_df = melted_df.rename(columns={'date': 'ds'})
        
        return melted_df



    def widen_and_rescale_dataframe(self, value_list, column_names, rescale=True):
        stacked_df = pd.DataFrame({
            'value': value_list})
        
        num_columns = len(column_names)
        rows_per_column = len(stacked_df) / num_columns

        if (num_columns * rows_per_column) != len(stacked_df):
            raise ValueError ("the num_columns * rows_per_column does not match with the stacked dataframw size, check the shape of each one of these before running again")
        
        deflated_data = {}
        
        for i, col_name in enumerate(column_names):
            start_idx = int(i * rows_per_column)
            end_idx = int(start_idx + rows_per_column)
            deflated_data[col_name] = stacked_df['value'].iloc[start_idx:end_idx].values
        
        deflated_df = pd.DataFrame(deflated_data)
        if rescale:
            deflated_df = pd.DataFrame ( self.scaler.inverse_transform(deflated_df)  ,
                                         columns = column_names
                                        )
        return deflated_df


    def widen_dataframe(self, value_list, column_names, remove_lag=True, size=0):
        stacked_df = value_list        
        num_columns = len(column_names)
        rows_per_column = len(stacked_df) / num_columns

        if (num_columns * rows_per_column) != len(stacked_df):
            raise ValueError ("the num_columns * rows_per_column does not match with the stacked dataframw size, check the shape of each one of these before running again")
        
        deflated_data = {}

        for i, col_name in enumerate(column_names):
            print ("column name: ", col_name)
            deflated_data[col_name] = stacked_df[ (stacked_df['ID'] == col_name) ]["yhat1"].values
            #print (deflated_data[col_name])
            #print (type(deflated_data[col_name]))
        deflated_df = pd.DataFrame(deflated_data)
        
        if remove_lag==True:
            deflated_df = deflated_df[size:]
            
        return deflated_df
    

    def rescale_dataframe(self, value_list, column_names):
        df = value_list
        print ("columns before rescaling: ", df.columns, df.shape)
        unscaled_df = pd.DataFrame ( self.scaler.inverse_transform(df)  ,
                                     columns = column_names
                                    )
        print ("columns after rescaling: ", type(unscaled_df))
        return unscaled_df
    
    def getStepSize(self,num_windows):

        step_size = 50

        if num_windows>=500:
            step_size = 25 + int( (num_windows-500 )*0.05 )
        if num_windows>=400 and num_windows < 500:
            step_size = 20
        if num_windows>=300 and num_windows < 400:
            step_size = 15
        if num_windows>=200 and num_windows < 300:
            step_size = 10
        if num_windows>=100 and num_windows < 200:
            step_size = 5
        if num_windows>=50 and  num_windows < 100:
            step_size = 3
        if num_windows < 50:
            step_size = 1

        return step_size*8
