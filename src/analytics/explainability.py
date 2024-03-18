import shap

import numpy as np
import pandas as pd
import catboost as cb


class Explainability:
    """

    """

    def __init__(self, model: cb.CatBoostRegressor, df: pd.DataFrame, validation_dict: dict) -> None:
        """

        """

        self.model: cb.CatBoostRegressor = model
        self.df: pd.DataFrame = df
        self.validation_dict: dict = validation_dict

        self.val = self.df[
            self.df['date_block_num'].isin([row['val'] for row in self.validation_dict['validation_indexes']][0])]
        self.explainer: shap.TreeExplainer = shap.TreeExplainer(self.model,
                                                                self.val.drop(columns=['item_cnt_month'], axis=1))
        self.shap_values: np.array = self.explainer.shap_values(self.val.drop(columns=['item_cnt_month'], axis=1))

    def shap_visualization(self, dep_feature_name: str, force_sample_idx: int = 0) -> None:
        """

        """

        print("\nSummary plot:\n")
        shap.summary_plot(self.shap_values[0])

        print("\nForce plot:\n")
        shap.force_plot(self.explainer.expected_value[0], self.shap_values[0][force_sample_idx, :],
                        self.val[force_sample_idx, :],
                        feature_names=self.val.drop(columns=['item_cnt_month'], axis=1).columns.to_list())

        print("\nDependence plot:\n")
        shap.dependence_plot(dep_feature_name, self.shap_values[0],
                             self.val.drop(columns=['item_cnt_month'], axis=1),
                             feature_names=self.val.columns.to_list())
