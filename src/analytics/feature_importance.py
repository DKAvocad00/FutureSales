import pandas as pd
from matplotlib import pyplot as plt
import catboost as cb


class FeatureImportance:
    """
    Class for visualizing feature importance using a trained CatBoostRegressor model.
    """

    def __init__(self, model: cb.CatBoostRegressor, df: pd.DataFrame, validation_dict: dict) -> None:
        """
        **Initialize FeatureImportance object.**

        :param model: Trained CatBoostRegressor model.
        :param df: DataFrame containing data.
        :param validation_dict: Dictionary containing validation data details.
        """
        self.model: cb.CatBoostRegressor = model
        self.df: pd.DataFrame = df
        self.validation_dict: dict = validation_dict

    def feature_importance_visualize(self) -> None:
        """
        **Visualize feature importance using the trained model.**

        This method calculates and visualizes feature importance based on PredictionValuesChange.

        :return: None
        """
        val = self.df[
            self.df['date_block_num'].isin([row['val'] for row in self.validation_dict['validation_indexes']][0])]

        X_val = val.drop(columns='item_cnt_month', axis=1)

        importances = self.model.get_feature_importance(type='PredictionValuesChange')
        feature_importances = pd.Series(importances, index=X_val.columns).sort_values()

        plt.figure(figsize=(10, 6))
        plt.barh(feature_importances.index, feature_importances.values)
        plt.title('CatBoost Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.show()
