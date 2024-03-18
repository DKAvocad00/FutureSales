import shap

import numpy as np
import pandas as pd
import catboost as cb


class Explainability:
    """
    Class for generating SHAP (SHapley Additive exPlanations) plots to explain model predictions.
    """

    def __init__(self, model: cb.CatBoostRegressor, val_data: pd.DataFrame, validation_dict: dict) -> None:
        """
        **Initialize Explainability object.**

        :param model: Trained CatBoostRegressor model.
        :param val_data: DataFrame containing validation data.
        :param validation_dict: Dictionary containing validation data details.
        """
        self.model: cb.CatBoostRegressor = model
        self.validation_dict: dict = validation_dict
        self.val_data: pd.DataFrame = val_data

        self.X: pd.DataFrame = self.val_data.drop(columns=['item_cnt_month'], axis=1)

        self.explainer: shap.TreeExplainer = shap.TreeExplainer(self.model)
        self.shap_values: np.array = self.explainer.shap_values(self.X)

    def shap_summary(self) -> None:
        """
        *8Generate summary SHAP plot.**

        This method generates a summary plot of SHAP values.

        :return: None
        """
        print("\nSummary plot:\n")
        shap.summary_plot(self.shap_values, features=self.X, feature_names=self.X.columns, color="auto")

    def shap_dependece(self, dep_features_name: list) -> None:
        """
        **Generate SHAP dependence plots.*8

        This method generates SHAP dependence plots for specified features.

        :param dep_features_name: List of feature names for which dependence plots are generated.
        :return: None
        """

        print("\nDependence plot:\n")
        for feature in dep_features_name:
            shap.dependence_plot(feature, self.shap_values,
                                 self.X, feature_names=self.X.columns.to_list())
