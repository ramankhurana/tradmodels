## To do
# add column masking
# add # of columns to be used
# 

import yaml

class DatasetConfig:
    def __init__(self, filepath):
        self.filepath = filepath
        self.datasets = {}
        self.read_yaml()

    def read_yaml(self):
        with open(self.filepath, 'r') as file:
            data = yaml.safe_load(file)  # Load the YAML file

            # Store base paths
            local_base_path = data['local_base_path']
            cloud_base_path = data['cloud_base_path']

            # Iterate over each dataset and store the necessary details
            for dataset_name, details in data['datasets'].items():
                self.datasets[dataset_name] = {
                    'name': details['name'],
                    'date_col': details['date_col'],
                    'train': details['train'],
                    'test': details['test'],
                    'val': details['val'],
                    'lag': details['lag'],
                    'horizon': details['horizon'],
                    'dataset_path': details['dataset_path'].replace('{{ local_base_path }}', local_base_path),
                    'cloud_path': details['cloud_path'].replace('{{ cloud_base_path }}', cloud_base_path)
                }

    def get_dataset_info(self, dataset_name):
        return self.datasets.get(dataset_name, None)

# Example usage
if __name__ == "__main__":
    config = DatasetConfig('datasetschema.yaml')
    dataset_info = config.get_dataset_info('Exchange')  # Retrieve info for the 'Exchange' dataset
    print(dataset_info)
    print (config.datasets["ETTh1"])
