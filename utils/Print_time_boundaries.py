import pandas as pd
import numpy as np

class DataSplitter:
    def __init__(self, dataset_length, dataset_name, lag):
        self.dataset_length = dataset_length
        self.dataset_name = dataset_name
        self.seq_len = lag

    def border_to_range(self, b1, b2):
        return [(b1[0], b2[0]), (b1[1], b2[1]), (b1[2], b2[2])]

    def get_range(self):
        num_train = int(self.dataset_length * 0.7)
        num_test = int(self.dataset_length * 0.2)
        num_vali = self.dataset_length - num_train - num_test
        border1s = [0, num_train - self.seq_len, self.dataset_length - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, self.dataset_length]
        return self.border_to_range(border1s, border2s)

    def get_range_etth(self):
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        return self.border_to_range(border1s, border2s)

    def get_range_ettm(self):
        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        return self.border_to_range(border1s, border2s) 

    def get_train_val_test_ranges(self):
        if self.dataset_name in ["ETTh1", "ETTh2"]:
            return self.get_range_etth()
        if self.dataset_name in ["ETTm1", "ETTm2"]:
            return self.get_range_ettm()
        else:
            return self.get_range()


# Example of how to use the DataSplitter class
#df = pd.read_csv("/path/to/your/dataset.csv")
#splitter = DataSplitter(len(df), "ETTh1")  # Use dataset_name as needed
#print(splitter.get_train_val_test_ranges())
