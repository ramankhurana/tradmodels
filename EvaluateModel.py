import importlib
from DatasetConfig import DatasetConfig
from ModelConfig import ModelConfig
import os 
import numpy as np
import argparse
import pandas as pd
def parseargs():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset to use')
    parser.add_argument('--model', type=str, required=True, help='The name of the model to use')
    args = parser.parse_args()
    dataset = args.dataset
    model = args.model
    



class EvaluateModel:
    def __init__(self, dataset_file, model_file, dataset_name, model_name):
        self.dataset_config = DatasetConfig(dataset_file)
        self.model_config = ModelConfig(model_file, model_name)
        self.dataset_info = self.dataset_config.get_dataset_info(dataset_name)
        self.model = self.load_model(model_name)
        self.dataset = dataset_name
        
        self.predictions=None
        self.mse_scores=None
        self.aggregate_mse=None
        self.consolidated_mse=None

        self.results_dir = f'results/{model_name}'
        
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def load_model(self, model_name):
        try:
            # Dynamically import the model module based on the model_name provided
            model_module = importlib.import_module("models."+model_name + 'Model')
            # Dynamically create an instance of the model class
            model_class = getattr(model_module, model_name + 'Model')
            return model_class(self.dataset_info)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Model class {model_name + 'Model'} could not be loaded: {e}")

    def evaluate(self):
        if not self.model:
            raise Exception("Model is not initialized or not found.")
        self.model.load_data()
        self.predictions, self.mse_scores, self.aggregate_mse, self.consolidated_mse = self.model.fit_predict()


    def save_predictions(self):
        for column, prediction in self.predictions.items():
            np.save(os.path.join(self.results_dir, f'{column}.npy'), prediction.values())

    def save_results_to_csv(self):
        results_path = f'results/results.csv'
        results_data = {
            'Model': [self.model],
            'Dataset': [self.dataset],
            'Consolidated MSE': [self.consolidated_mse]
        }
        df = pd.DataFrame(results_data)
        if os.path.exists(results_path):
            df.to_csv(results_path, mode='a', header=False, index=False)
        else:
            df.to_csv(results_path, index=False)
    
    def printresults(self):
        for col, mse in self.mse_scores.items():
            print(f"MSE for {col}: {mse}")
        print ("consolidated_mse: ",self.consolidated_mse )

# Usage example
if __name__ == "__main__":
    parseargs()
    evaluator = EvaluateModel('datasetschema.yaml', 'runschema.yaml', 'ETTh2', 'ARIMA')
    evaluator.evaluate()
    evaluator.save_predictions()
    evaluator.save_results_to_csv()
    evaluator.printresults()