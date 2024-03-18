import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import catboost as cb
from boruta import BorutaPy

from typing import Union, List


class FeatureImportance:
    """

    """

    def __init__(self, model: cb.CatBoostRegressor, df: pd.DataFrame, validation_dict: dict) -> None:
        """

        """
        self.model: cb.CatBoostRegressor = model
        self.df: pd.DataFrame = df
        self.validation_dict: dict = validation_dict

    def feature_importance_visualize(self) -> None:
        """

        """
        val = self.df[
            self.df['date_block_num'].isin([row['val'] for row in self.validation_dict['validation_indexes']][0])]

        feature_importance = self.model.feature_importances_
        sorted_idx = np.argsort(feature_importance)

        fig = plt.figure(figsize=(12, 6))
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), np.array(val.columns.difference(['item_cnt_month']))[sorted_idx])
        plt.title('Feature Importance')

    def boruta_importance(self, **boruta_params: dict) -> BorutaPy:
        """

        """

        train = self.df[
            self.df['date_block_num'].isin([row['train'] for row in self.validation_dict['validation_indexes']][0])]

        boruta_selector: BorutaPy = BorutaPy(self.model, **boruta_params)

        boruta_selector.fit(train.drop(columns=['item_cnt_month'], axis=1), train['item_cnt_month'])

        feature_ranks: list = list(zip(train.drop(columns='item_cnt_month', axis=1).columns.to_list(),
                                       boruta_selector.ranking_,
                                       boruta_selector.support_))

        for feat in feature_ranks:
            print('Feature: {:<25} Rank: {},  Keep: {}'.format(feat[0], feat[1], feat[2]))

        return boruta_selector
