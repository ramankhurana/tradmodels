import yaml

class ModelConfig:
    def __init__(self, filepath, model_name):
        self.filepath = filepath
        self.model_name = model_name
        self.validate = None
        self.runmodel = None
        self.filtered_datasets = []

        self.read_config()

    def read_config(self):
        with open(self.filepath, 'r') as file:
            data = yaml.safe_load(file)
            model_data = data['runschema'][self.model_name]

            self.validate = model_data['validate']
            self.runmodel = model_data['runmodel']
            all_datasets = set(model_data['dataset'])
            mask_datasets = set(model_data['mask'])

            # Filter datasets by removing those listed in the mask
            self.filtered_datasets = list(all_datasets - mask_datasets)

    def get_filtered_datasets(self):
        return self.filtered_datasets

    def __str__(self):
        return (f"Model: {self.model_name}\n"
                f"Validate: {self.validate}\n"
                f"Run Model: {self.runmodel}\n"
                f"Filtered Datasets: {self.filtered_datasets}")

# Usage example
if __name__ == "__main__":
    path_to_yaml_file = 'runschema.yaml'
    config = ModelConfig(path_to_yaml_file, 'AR')
    print(config)
