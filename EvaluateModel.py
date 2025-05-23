import importlib
from DatasetConfig import DatasetConfig
from ModelConfig import ModelConfig
import os 
import numpy as np
import argparse
import pandas as pd

from BaseModel import BaseModel

''' This function is to parse the arguments that are passed via commandline while runing this python script
This needs two inputs; datasets and model: 
1. Which dataset should be used for evaluation. 
2. This needs the model for which evaluation needs to e performed.

In addition to these init of class require two yaml files. 
1. 'datasetschema.yaml': This has all details about each of the dataset
2. 'runschema.yaml': This has all details on which datasets to run on evaluation

''' 
def parseargs():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset to use')
    parser.add_argument('--model', type=str, required=True, help='The name of the model to use')
    args = parser.parse_args()
    dataset = args.dataset
    model = args.model
    return (dataset, model)


'''
This is the main class to do the evaluation. Note that only one dataset cna e evaluated for one model at a time. This is to ensure minimal reprocessing in case of errors during evaluation.
This class need 4 inputs:
1. dataset_file: this is yaml file
2. model_file: this is yaml fiel
3. dataset_name: name of the dataste
4. model_name: name of the model 
'''
class EvaluateModel:
    def __init__(self, dataset_file, model_file, dataset_name, model_name):
        self.dataset_config = DatasetConfig(dataset_file)
        self.model_config = ModelConfig(model_file, model_name)

        ## Get the dataset information 
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
            self.results_base_dir = 'results2025/'
        else:
            self.results_base_dir = self.get_path_till_dataset(self.dataset_info["dataset_path"])  ## there should be results dir appended here 


        
        self.results_dir = f'{self.results_base_dir}/{model_name}'

        #self.results_base_dir = '/mnt-gluster/all-data/khurana/dataset-tradmodels/dataset/results/'
        #self.results_dir = f'/mnt-gluster/all-data/khurana/dataset-tradmodels/dataset/results/{model_name}'
        
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)


    ''' Load the model to be evaluated and create an instance for the model. For each model there is a dedicated class that is inherited from Base class name BaseModel.py
    Note that the model is imported in the initization step to ensure it can be used in rest of the class  '''
    
    
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

    ''' This is to load the dataset the needs to be used for evaluation. 
    self.model.load_data is a function implemented in BaseModel to avoid implementation for each model and dataset.
    Just before evaluation, dataset is loaded and scaled (if scale flag is not set to false).
    This also takes care of training or fitting the model like ARIMA if needed'''
    
    def evaluate(self):
        if not self.model:
            raise Exception("Model is not initialized or not found.")
        self.model.load_data()
        #self.predictions, self.mse_scores, self.aggregate_mse, self.consolidated_mse = self.model.fit_predict()
        #self.predictions, self.mse_scores, self.aggregate_mse, self.consolidated_mse, self.metrics = self.model.rolling_window_evaluation()
        self.predictions, self.mse_scores, self.metrics = self.model.rolling_window_multi_horizon_evaluation() 
        

    def save_predictions(self):
        if self.model_name == "TimeGPT":
            np.save(os.path.join(self.results_dir, f'{self.model_name}.npy'), self.predictions["TimeGPT"].values)
        else:
            for column, prediction in self.predictions.items():
                np.save(os.path.join(self.results_dir, f'{column}.npy'), prediction.values())

    #def save_results_to_csv(self):
    #    print ("results: ", self.metrics)
    #    results_path = f'{self.results_base_dir}/results_v05.csv' # f'/mnt-gluster/all-data/khurana/dataset-tradmodels/dataset/results/results.csv'
    #    results_data = {
    #        'Model': [self.model_name],
    #        'Dataset': [self.dataset],
    #        'Lag': [self.dataset_info['lag']],
    #        'Horizon': [self.dataset_info['horizon']],
    #        'Consolidated MSE': [self.consolidated_mse],
    #        'MAE':[self.metrics["MAE"]],
    #        'MASE':[self.metrics["MASE"]],
    #        'ZI-MAE':[self.metrics["ZI-MAE"]],
    #        'ZI-MSE':[self.metrics["ZI-MSE"]],
    #        'MAPE':[self.metrics['MAPE']],
    #        'SMAPE':[self.metrics['SMAPE']],
    #        'ZI-MAPE':[self.metrics['ZI-MAPE']],
    #        'ZI-SMAPE':[self.metrics['ZI-SMAPE']]
    #        #'':[self.metrics['']]
    #        
    #    }
    #    
    #    df = pd.DataFrame(results_data)
    #    if os.path.exists(results_path):
    #        df.to_csv(results_path, mode='a', header=False, index=False)
    #    else:
    #        df.to_csv(results_path, index=False)
    #

    def save_results_to_csv(self):
        print("Saving results for multiple horizons:", self.metrics)
    
        results_path = f'{self.results_base_dir}/results_v05.csv'
        rows = []
    
        for horizon, metrics in self.metrics.items():
            row = {
                'Model': self.model_name,
                'Dataset': self.dataset,
                'Lag': self.dataset_info['lag'],
                'Horizon': horizon,
                'Consolidated MSE': metrics.get("MSE"),
                'MAE': metrics.get("MAE"),
                'MASE': metrics.get("MASE"),
                'ZI-MAE': metrics.get("ZI-MAE"),
                'ZI-MSE': metrics.get("ZI-MSE"),
                'MAPE': metrics.get("MAPE"),
                'SMAPE': metrics.get("SMAPE"),
                'ZI-MAPE': metrics.get("ZI-MAPE"),
                'ZI-SMAPE': metrics.get("ZI-SMAPE"),
            }
            rows.append(row)
    
        df = pd.DataFrame(rows)
        # Append to CSV if file exists, otherwise create it
        try:
            existing = pd.read_csv(results_path)
            df = pd.concat([existing, df], ignore_index=True)
        except FileNotFoundError:
            pass
        
        df.to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")

    
    
    def printresults(self):
        if self.model_name=="TimeGPT":
            print ("consolidated_mse: ",self.consolidated_mse)
        else:
            #for col, mse in self.mse_scores.items():
            #    print(f"MSE for {col}: {mse}")
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
    #evaluator.save_predictions()
    evaluator.save_results_to_csv()
    evaluator.printresults()
