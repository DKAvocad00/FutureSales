import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import catboost as cb
from typing import Union


class ModelErrorAnalysis:
    """

    """

    def __init__(self, model: cb.CatBoostRegressor, df: pd.DataFrame, validation_dict: dict) -> None:
        """

        """
        self.model: cb.CatBoostRegressor = model
        self.df: pd.DataFrame = df
        self.validation_dict: dict = validation_dict
        self.val: pd.DataFrame = self.df[
            self.df['date_block_num'].isin([row['val'] for row in self.validation_dict['validation_indexes']][0])]

        self.X_val = self.val.drop(columns=['item_cnt_month'], axis=1)
        self.y_val = self.val['item_cnt_month']

        self.predictions: np.array = self.model.predict(self.X_val)
        self.errors: np.array = self.predictions - self.y_val

    def calculate_metrics(self) -> dict:
        """

        """
        mae = np.mean(np.abs(self.errors))
        mse = mean_squared_error(self.y_val, self.predictions)
        rmse = np.sqrt(mse)
        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

    def plot_residuals(self) -> None:
        """

        """
        plt.figure(figsize=(10, 6))
        sns.residplot(x=self.predictions, y=self.errors, lowess=True, line_kws={'color': 'red', 'lw': 1})
        plt.title('Residuals Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.show()

    def analyze_big_target(self, target_threshold) -> np.float64:
        """

        """
        big_target_indices = self.y_val >= target_threshold
        big_target_errors = self.errors[big_target_indices]
        big_target_mae = np.mean(np.abs(big_target_errors))
        return big_target_mae

    def analyze_small_dynamic(self, dynamic_threshold) -> np.float64:
        """

        """
        dynamic_indices = np.abs(self.y_val) <= dynamic_threshold
        dynamic_errors = self.errors[dynamic_indices]
        dynamic_mae = np.mean(np.abs(dynamic_errors))
        return dynamic_mae

    def find_influential_samples(self, threshold) -> tuple:
        """

        """
        influential_samples = np.abs(self.errors) > threshold
        return self.y_val[self.y_val['item_cnt_month'] == influential_samples], self.errors[influential_samples]
