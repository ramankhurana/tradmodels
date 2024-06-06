import importlib
from DatasetConfig import DatasetConfig
from ModelConfig import ModelConfig
import os 
import numpy as np
import argparse
import pandas as pd

from BaseModel import BaseModel

def parseargs():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset to use')
    parser.add_argument('--model', type=str, required=True, help='The name of the model to use')
    args = parser.parse_args()
    dataset = args.dataset
    model = args.model
    return (dataset, model)



class EvaluateModel:
    def __init__(self, dataset_file, model_file, dataset_name, model_name):
        self.dataset_config = DatasetConfig(dataset_file)
        self.model_config = ModelConfig(model_file, model_name)
        self.dataset_info = self.dataset_config.get_dataset_info(dataset_name)
        self.model = self.load_model(model_name)
        self.dataset = dataset_name
        self.model_name = model_name
        
        self.predictions=None
        self.mse_scores=None
        self.aggregate_mse=None
        self.consolidated_mse=None

        #print ("dataset path in  Eval: ", self.dataset_info["dataset_path"], self.get_path_till_dataset(self.dataset_info["dataset_path"]), self.dataset_config.run_environment)
        print (f'This job is running in {self.dataset_config.run_environment} environment')
        
        if self.dataset_config.run_environment == "local":
            self.results_base_dir = 'results/'
        else:
            self.results_base_dir = self.get_path_till_dataset(self.dataset_info["dataset_path"])  ## there should be results dir appended here 

        
        self.results_dir = f'{self.results_base_dir}/{model_name}'

        #self.results_base_dir = '/mnt-gluster/all-data/khurana/dataset-tradmodels/dataset/results/'
        #self.results_dir = f'/mnt-gluster/all-data/khurana/dataset-tradmodels/dataset/results/{model_name}'
        
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def load_model(self, model_name):
        try:
            # Dynamically import the model module based on the model_name provided
            model_module = importlib.import_module("models."+model_name + 'Model')
            # Dynamically create an instance of the model class
            model_class = getattr(model_module, model_name + 'Model')
            print ("model_class: ", model_class)
            ''' this is the return statement which initialise the model class ''' 
            return model_class(self.dataset_info)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Model class {model_name + 'Model'} could not be loaded: {e}")

    def evaluate(self):
        if not self.model:
            raise Exception("Model is not initialized or not found.")
        self.model.load_data()
        #self.predictions, self.mse_scores, self.aggregate_mse, self.consolidated_mse = self.model.fit_predict()
        self.predictions, self.mse_scores, self.aggregate_mse, self.consolidated_mse = self.model.rolling_window_evaluation()
        

    def save_predictions(self):
        if self.model_name == "TimeGPT":
            np.save(os.path.join(self.results_dir, f'{self.model_name}.npy'), self.predictions["TimeGPT"].values)
        else:
            for column, prediction in self.predictions.items():
                np.save(os.path.join(self.results_dir, f'{column}.npy'), prediction.values())

    def save_results_to_csv(self):
        results_path = f'{self.results_base_dir}/results_v03.csv' # f'/mnt-gluster/all-data/khurana/dataset-tradmodels/dataset/results/results.csv'
        results_data = {
            'Model': [self.model_name],
            'Dataset': [self.dataset],
            'Lag': [self.dataset_info['lag']],
            'Horizon': [self.dataset_info['horizon']],
            'Consolidated MSE': [self.consolidated_mse]
        }
        df = pd.DataFrame(results_data)
        if os.path.exists(results_path):
            df.to_csv(results_path, mode='a', header=False, index=False)
        else:
            df.to_csv(results_path, index=False)
    
    def printresults(self):
        if self.model_name=="TimeGPT":
            print ("consolidated_mse: ",self.consolidated_mse)
        else:
            for col, mse in self.mse_scores.items():
                print(f"MSE for {col}: {mse}")
            print ("consolidated_mse: ",self.consolidated_mse )
  
    def get_path_till_dataset(self, full_path):
        parts = full_path.split(os.sep)
        # Find the index of 'dataset' and slice the list up to and including this index
        if 'dataset' in parts:
            dataset_index = parts.index('dataset') + 1
            return os.sep.join(parts[:dataset_index])
        return None

# Usage example
if __name__ == "__main__":
    (dataset, model) = parseargs()
    evaluator = EvaluateModel('datasetschema.yaml', 'runschema.yaml', dataset, model)
    evaluator.evaluate()
    evaluator.save_predictions()
    evaluator.save_results_to_csv()
    evaluator.printresults()
