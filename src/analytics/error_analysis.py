import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import catboost as cb
from typing import Union


class ModelErrorAnalysis:
    """
    Class for analyzing errors of a trained CatBoostRegressor model.
    """

    def __init__(self, model: cb.CatBoostRegressor, df: pd.DataFrame, predictions: list, validation_dict: dict) -> None:
        """
        **Initialize ModelErrorAnalysis object.**

        :param model: Trained CatBoostRegressor model.
        :param df: DataFrame containing data.
        :param predictions: List of model predictions.
        :param validation_dict: Dictionary containing validation data details.
        """

        self.model: cb.CatBoostRegressor = model
        self.df: pd.DataFrame = df
        self.validation_dict: dict = validation_dict
        self.predictions: list = predictions

        self.val: pd.DataFrame = self.df[
            self.df['date_block_num'].isin([row['val'] for row in self.validation_dict['validation_indexes']][0])]

        self.y_val = self.val['item_cnt_month']

        self.errors: np.array = self.predictions - self.y_val

    def calculate_metrics(self) -> dict:
        """
        **Calculate error metrics.**

        This method calculates error metrics including Mean Absolute Error (MAE),
        Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) based on the
        model's predictions and actual target values.

        :return: Dictionary containing error metrics with keys 'MAE', 'MSE', and 'RMSE'.
        """

        mae = np.mean(np.abs(self.errors))
        mse = mean_squared_error(self.y_val, self.predictions)
        rmse = np.sqrt(mse)
        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

    def plot_residuals(self) -> None:
        """
        **Plot residuals.**

        This method generates a residual plot to visualize the errors between the
        predicted values and the actual target values.

        :return: None
        """
        plt.figure(figsize=(10, 6))
        sns.residplot(x=self.predictions, y=self.errors, lowess=True, line_kws={'color': 'red', 'lw': 1})
        plt.title('Errors Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Errors')
        plt.show()

    def analyze_big_target(self, target_threshold) -> np.float64:
        """
        **Analyze errors for samples with big target values.**

        This method calculates the mean absolute error (MAE) for samples with target
        values greater than or equal to the specified threshold.

        :param target_threshold: Threshold for defining big target values.
        :return: Mean absolute error for samples with big target values.
        """
        big_target_indices = self.y_val >= target_threshold
        big_target_errors = self.errors[big_target_indices]
        big_target_mae = np.mean(np.abs(big_target_errors))
        return big_target_mae

    def analyze_small_dynamic(self, dynamic_threshold) -> np.float64:
        """
        **Analyze errors for samples with small dynamic values.**

        This method calculates the mean absolute error (MAE) for samples with target
        values less than or equal to the specified threshold.

        :param dynamic_threshold: Threshold for defining small dynamic values.
        :return: Mean absolute error for samples with small dynamic values.
        """
        dynamic_indices = np.abs(self.y_val) <= dynamic_threshold
        dynamic_errors = self.errors[dynamic_indices]
        dynamic_mae = np.mean(np.abs(dynamic_errors))
        return dynamic_mae

    def find_influential_samples(self, threshold) -> tuple:
        """
        **Find influential samples.**

        This method identifies influential samples based on the absolute errors exceeding
        the specified threshold.

        :param threshold: Threshold for defining influential samples.
        :return: Tuple containing influential sample target values and errors.
        """
        influential_samples = np.abs(self.errors) > threshold
        return self.y_val[influential_samples], self.errors[influential_samples]
