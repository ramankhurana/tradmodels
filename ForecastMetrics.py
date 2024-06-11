import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

class ForecastMetrics:
    def __init__(self, y_true, y_pred,y_true_orig, y_pred_orig):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_true_orig = y_true_orig # data before normalisation 
        self.y_pred_orig = y_pred_orig # data before normalisation 
        
    def mse(self):
        n = len(self.y_true)
        print ("type of the true", type(self.y_true) )
        diff = self.y_true - self.y_pred
        return np.sum(np.square(diff)) / n 
    
    def mape(self):
        return mean_absolute_percentage_error(self.y_true_orig, self.y_pred_orig)
    
    def smape(self):
        n = len(self.y_true_orig)
        errors = np.abs(self.y_true_orig - self.y_pred_orig)
        denominator = (np.abs(self.y_true_orig) + np.abs(self.y_pred_orig)) / 2
        smape = np.sum(errors / denominator) * (1. / n)
        return smape
    
    def mae(self):
        n = len(self.y_true)
        errors = np.abs(self.y_true - self.y_pred)
        mae = np.sum(errors) / n
        return mae
    
    def mase(self, naive_forecast):
        n = len(self.y_true)
        errors = np.abs(self.y_true - self.y_pred)
        mean_absolute_error = np.sum(errors) / n
        denominator = np.sum(np.abs(self.y_true - naive_forecast)) / n
        mase = mean_absolute_error / denominator
        return mase
    
    def zi_mape(self):
        n = len(self.y_true_orig)
        errors = np.where(self.y_true_orig != 0, np.abs((self.y_true_orig - self.y_pred_orig) / self.y_true_orig), np.abs(self.y_pred_orig))
        zi_mape = np.sum(errors) / n
        return zi_mape
    
    def zi_smape(self):
        n = len(self.y_true_orig)
        errors = np.where(self.y_true_orig != 0, (np.abs(self.y_true_orig - self.y_pred_orig) * 2) / (np.abs(self.y_true_orig) + np.abs(self.y_pred_orig)), np.abs(self.y_pred_orig))
        zi_smape = np.sum(errors) / n
        return zi_smape
    
    def zi_mae(self):
        n = len(self.y_true)
        errors = np.where(self.y_true != 0, np.abs(self.y_true - self.y_pred), 0)
        zi_mae = np.sum(errors) / n
        return zi_mae
    
    def zi_mse(self):
        n = len(self.y_true)
        errors = np.where(self.y_true != 0, np.square(self.y_true - self.y_pred), 0)
        zi_mse = np.sum(errors) / n
        return zi_mse

    def normalised_metrics(self):
        metrics = {
            "MSE": self.mse(),
            "MAE": self.mae(),
            "MASE": self.mase(np.roll(self.y_true, 1)),  # Naïve forecast: Shift previous values
            "ZI-MAE": self.zi_mae(),
            "ZI-MSE": self.zi_mse()
        }
        return metrics

    def unnormalised_metrics(self):
        metrics = {
            "MAPE": self.mape(),
            "SMAPE": self.smape(),
            "ZI-MAPE": self.zi_mape(),
            "ZI-SMAPE": self.zi_smape(),
        }
        return metrics

    
    def calculate_all_metrics(self):
        metrics = {
            "MSE": self.mse(),
            "MAPE": self.mape(),
            "SMAPE": self.smape(),
            "MAE": self.mae(),
            "MASE": self.mase(np.roll(self.y_true, 1)),  # Naïve forecast: Shift previous values
            "ZI-MAPE": self.zi_mape(),
            "ZI-SMAPE": self.zi_smape(),
            "ZI-MAE": self.zi_mae(),
            "ZI-MSE": self.zi_mse()
        }
        return metrics
    
    def print_metrics(self):
        metrics = self.calculate_all_metrics()
        for metric, value in metrics.items():
            print(f"{metric}: {value}")



'''
# Example usage
y_true = np.array([3, 0, 1, 5, 0, 2])
y_pred = np.array([3, 0, 2, 4, 1, 3])

metrics = ForecastMetrics(y_true, y_pred)
metrics.print_metrics()
'''
