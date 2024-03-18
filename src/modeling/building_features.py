from typing import Callable, Any
import pandas as pd
import itertools
import numpy as np
from src.path_utils import save_final_to
import os
from src.modeling.utils import downcast, write_to_json


class FeatureModeling:
    """
    Class for feature engineering and modeling.
    """

    def __init__(self, sales_path: str, shop_path: str, item_path: str, item_categories_path: str, test_path) -> None:
        """
        Initialize FeatureModeling object with file paths for sales, shop, item, item categories, and test data.

        :param sales_path: Path to the sales data file.
        :param shop_path: Path to the shop data file.
        :param item_path: Path to the item data file.
        :param item_categories_path: Path to the item categories data file.
        :param test_path: Path to the test data file.
        """
        self.sales_path: str = sales_path
        self.shop_path: str = shop_path
        self.item_path: str = item_path
        self.item_categories_path: str = item_categories_path
        self.test_path: str = test_path

        self.df: pd.DataFrame | None = None
        self.mean_features: list = []
        self.features_dict: dict = {"in_features": [], "target": [], "cat_features": [], "lag_features": {}}

    def create_final_data(self, make_big: bool = False) -> None:

        """
        **Creates the final DataFrame for training based on the provided train and test DataFrames.**

        This method creates the final DataFrame for analysis by aggregating the train DataFrame
        based on date_block_num, shop_id, and item_id and merging it with the test DataFrame if provided.
        Optionally, it can expand the DataFrame to include all possible combinations of date_block_num,
        shop_id, and item_id if make_big is set to True.

        :param make_big: If True, expands the DataFrame to include all possible combinations of
                         date_block_num, shop_id, and item_id. Default is False.
        :return: The final DataFrame for training.
        """

        # Read DataFrames
        train = pd.read_csv(self.sales_path)
        test = pd.read_csv(self.test_path)

        # Aggregate train DataFrame based on date_block_num, shop_id, and item_id
        self.df = train.groupby(["date_block_num", "shop_id", "item_id"], as_index=False).agg(
            item_cnt_month=pd.NamedAgg(column="item_cnt_day", aggfunc="sum")
        )

        # Optionally, expand the DataFrame to include all possible combinations of date_block_num,
        # shop_id, and item_id if make_big is True
        if make_big:
            indexlist: list = []
            for i in train['date_block_num'].unique():
                x = itertools.product([i],
                                      train.loc[train['date_block_num'] == i]['shop_id'].unique(),
                                      train.loc[train['date_block_num'] == i]['item_id'].unique(),
                                      )
                indexlist.append(np.array(list(x)))

            sales_big: pd.DataFrame = pd.DataFrame(data=np.concatenate(indexlist, axis=0),
                                                   columns=['date_block_num', 'shop_id', 'item_id'])

            self.df = sales_big.merge(self.df, how="left", on=['date_block_num', 'shop_id', 'item_id'])

        # If test DataFrame is provided, append it to the final DataFrame
        if test is not None:
            test["date_block_num"] = self.df['date_block_num'].max() + 1
            test = test.drop(columns="ID")

            self.df = pd.concat([self.df, test[["date_block_num", "shop_id", "item_id"]]])

        # Fill NaN values with 0 for item_cnt_month and item_revenue_month columns
        self.df['item_cnt_month'] = self.df['item_cnt_month'].fillna(0)

        # Add month feature
        self.df['month'] = self.df['date_block_num'] % 12

        # Downcast data
        self.df = downcast(self.df, verbose=False)

        self.features_dict['in_features'].extend(self.df.columns.difference(['item_cnt_month']))
        self.features_dict['target'].extend(['item_cnt_month'])
        self.features_dict['cat_features'].extend(self.df.columns.difference(['item_cnt_month']))

    def add_city_features(self) -> None:
        """
        **Add city-related features to the DataFrame.**

        This method reads shop data from the specified file path and adds city-related features to the DataFrame,
        including the shop category and city name.

        :return: None
        """

        shop_data = pd.read_csv(self.shop_path)

        # Extract shop category from shop name
        shop_data["shop_category"] = shop_data['shop_name'].str.split(" ").map(lambda x: x[1])

        # Merge shop data with the DataFrame on shop_id
        self.df = self.df.merge(shop_data[['city', 'shop_category', 'shop_id']], how="left", on=['shop_id'])

        self.features_dict['in_features'].extend(['city', 'shop_category'])
        self.features_dict['cat_features'].extend(['city', 'shop_category'])

    def add_item_features(self) -> None:
        """
        **Add features related to items to the DataFrame.**

        This method reads item data and sales data from the specified file paths and adds features related to items
        to the DataFrame, such as the item category ID, and the duration after the first sale of each item.

        :return: None
        """

        item_data = pd.read_csv(self.item_path)
        sales_data = pd.read_csv(self.sales_path)

        item_data['first_sale_date'] = sales_data.groupby('item_id').agg({'date_block_num': 'min'})['date_block_num']
        item_data['first_sale_date'] = item_data['first_sale_date'].fillna(self.df['date_block_num'].max())

        # Merge item data with the DataFrame on item_id
        self.df = self.df.merge(item_data[['item_id', 'item_category_id', 'first_sale_date']], how="left",
                                on=['item_id'])

        # Compute the duration after the first sale for each item
        self.df['duration_after_first_sale'] = self.df['date_block_num'] - self.df['first_sale_date']
        self.df.drop('first_sale_date', axis=1, inplace=True)

        self.features_dict['in_features'].extend(['item_category_id', 'duration_after_first_sale'])
        self.features_dict['cat_features'].extend(['item_category_id', 'duration_after_first_sale'])

    def add_item_categories_features(self, threshold: int = 5) -> None:
        """
        **Add features related to item categories to the DataFrame.**

        This method reads item categories data from the specified file path
        and adds features related to item categories to the DataFrame.
        It categorizes item categories based on the threshold parameter and merges them with the DataFrame.

        :param threshold: The minimum count threshold for categorizing item categories. Default is 5.
        :return: None
        """

        # Define a function to categorize item categories based on the threshold
        def _make_etc(x: str) -> str:
            if len(item_categories_data[item_categories_data['category'] == x]) >= threshold:
                return x
            else:
                return 'etc'

        item_categories_data = pd.read_csv(self.item_categories_path)

        # Apply the _make_etc function to categorize item categories
        item_categories_data['category'] = item_categories_data['category'].apply(_make_etc)

        # Merge item categories data with the DataFrame on item_category_id
        self.df = self.df.merge(item_categories_data[['category', 'item_category_id']], how="left",
                                on=['item_category_id'])

        self.features_dict['in_features'].extend(['category'])
        self.features_dict['cat_features'].extend(['category'])

    def add_mean_price(self) -> None:
        """
        **Add mean price features to the DataFrame.**

        This method calculates the mean item price for each (date_block_num, shop_id, item_id) group
        from the sales data and merges it into the DataFrame.

        :return: None
        """

        sales_data = pd.read_csv(self.sales_path)

        # Group the sales data by date_block_num, shop_id, and item_id, and calculate the mean item price
        group = sales_data.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False).agg({'item_price': 'mean'})
        group.rename(columns={'item_price': 'item_mean_price'}, inplace=True)

        # Merge the calculated mean item price with the main DataFrame based on date_block_num, shop_id, and item_id
        self.df = self.df.merge(group, how="left", on=['date_block_num', 'shop_id', 'item_id'])

        self.df['item_mean_price'] = self.df['item_mean_price'].fillna(0)

        self.df = downcast(self.df, verbose=False)

    def add_mean_features(self, idx_features: list) -> None:
        """
        **Add mean sales features to the DataFrame based on the specified index features.**

        This method calculates the mean monthly sales for each unique combination of the specified index features
        and merges it into the DataFrame.

        :param idx_features: List of features used for grouping the data.
                             Should include 'date_block_num' as the first element and one or two additional features.
        :return: None
        """

        # Ensure that the first index feature is 'date_block_num' and the length is valid
        assert (idx_features[0] == 'date_block_num') and len(idx_features) in [2, 3]

        # Determine the feature name based on the number of index features
        if len(idx_features) == 2:
            feature_name = idx_features[1] + '_mean_sales'
        else:
            feature_name = idx_features[1] + '_' + idx_features[2] + '_mean_sales'

        # Group the DataFrame by the specified index features and calculate the mean monthly sales
        group = self.df.groupby(idx_features, as_index=False).agg({'item_cnt_month': 'mean'})
        group.rename(columns={'item_cnt_month': feature_name}, inplace=True)

        # Merge the calculated mean sales feature with the main DataFrame based on the index features
        self.df = self.df.merge(group, on=idx_features, how='left')

        self.mean_features.append(feature_name)

        self.df = downcast(self.df, verbose=False)

    def add_lag_features(self, idx_features: list, lag_feature: str, validation_dict: dict, nlags: int = 3) -> None:
        """
        **Add lagged features to the DataFrame based on the specified lag feature and index features.**

        This method calculates lagged features for the given lag feature and adds them to the DataFrame.
        The number of lagged features to create can be specified by the 'nlags' parameter.

        :param idx_features: List of features used for grouping the data.
        :param lag_feature: The feature for which lagged features will be created.
        :param validation_dict: Dictionary containing validation indexes.
        :param nlags: Number of lagged features to create. Default is 3.
        :return: None
        """
        assert idx_features[0] == 'date_block_num'

        # Initialize lag_features dictionary for each fold
        self.features_dict["lag_features"] = {fold: self.features_dict["lag_features"].get(fold, []) for fold, _ in
                                              enumerate(validation_dict['validation_indexes'])}

        df_temp = self.df[idx_features + [lag_feature]].copy()

        for fold, row in enumerate(validation_dict['validation_indexes']):

            indexes_to_calculating = np.concatenate((row['train'], row['val'], validation_dict['test_indexes']))

            for i in range(1, nlags + 1):
                lag_feature_name = lag_feature + '_lag' + str(i) + "_" + str(fold)

                df_temp.columns = idx_features + [lag_feature_name]

                df_temp.loc[df_temp['date_block_num'].isin(indexes_to_calculating), 'date_block_num'] += 1

                self.df = self.df.merge(df_temp.drop_duplicates(), on=idx_features, how='left')

                self.df[lag_feature_name] = self.df[lag_feature_name].fillna(0)

                self.features_dict['lag_features'][fold].append(lag_feature_name)

        del df_temp
        self.df = downcast(self.df, verbose=False)

    def add_lag_mean_features(self, idx_features: list, validation_dict: dict, nlags: int = 3,
                              drop_mean_features: bool = False) -> None:
        """
        **Add lagged mean features to the DataFrame based on the specified index features.**

        This method calculates lagged mean features for each existing mean feature in the DataFrame and adds them as new
        features. Optionally, it can drop the original mean features after creating lagged features.

        :param idx_features: List of features used for grouping the data.
        :param validation_dict: Dictionary containing validation indexes.
        :param nlags: Number of lagged features to create. Default is 3.
        :param drop_mean_features: Whether to drop the original mean features after creating lagged features. Default is False.
        :return: None
        """

        for item_mean_feature in self.mean_features:
            # Add lagged features for the current mean feature
            self.add_lag_features(idx_features=idx_features, lag_feature=item_mean_feature,
                                  validation_dict=validation_dict, nlags=nlags)

        # Optionally drop the original mean features after creating lagged features
        if drop_mean_features:
            self.df.drop(self.mean_features, axis=1, inplace=True)
            self.mean_features = []

    def final_process(self):
        """
        **Perform final processing steps on the DataFrame.**

        This method performs the final processing steps on the DataFrame, including dropping the 'item_mean_price' column
        and downcasting the DataFrame.

        :return: None
        """
        # Drop item_mean
        self.df = self.df.drop('item_mean_price', axis=1)

        # Downcast data
        self.df = downcast(self.df, verbose=False)

        write_to_json(key="training_feature", value=self.features_dict)

    @staticmethod
    def add_features(feature_functions: list[Callable], **kwargs: Any) -> None:
        """
        **Apply a list of feature engineering functions to the data.**

        This method takes a list of feature engineering functions and applies each function
        to the data. It allows for flexible addition of multiple features using different functions.

        :param feature_functions: A list of feature engineering functions.
        :param kwargs: Additional keyword arguments to be passed to each feature function.
        :return: None
        """
        for func in feature_functions:
            func(**kwargs)
            print('[INFO]: {} function was successfully processed'.format(func.__name__))

    def load_data(self, file_name: str, save_to: str = save_final_to) -> None:
        """
        **Save the DataFrame to a CSV file.**

        This method saves the DataFrame to a CSV file with the specified file name.

        :param file_name: The name of the CSV file to save.
        :param save_to: Path to save final data.
        :return: None
        """

        if not os.path.exists(save_final_to):
            os.makedirs(save_final_to)

        self.df.to_csv(save_to + file_name + '.csv',
                       index=False, date_format='%d.%m.%Y')
        print("[INFO]: File {0} was successfully saved".format(file_name))

    def get_data(self) -> pd.DataFrame:
        """
        **Retrieves the  DataFrame.**

        This method returns the DataFrame.

        :return: DataFrame.
        """
        return self.df

    def set_data(self, df: pd.DataFrame) -> None:
        """
        Set the DataFrame for the FeatureModeling object.

        This method allows setting the DataFrame to be used for feature engineering and modeling.

        :param df: The DataFrame containing the data for feature engineering and modeling.
        :return: None
        """
        self.df = df
