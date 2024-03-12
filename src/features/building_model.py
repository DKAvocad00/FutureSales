import pandas as pd
import itertools
import numpy as np
from path_utils import save_final_to
import catboost as cb
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import adfuller, kpss
import os


class ValidationSchema:
    """

    """

    def __init__(self, df: pd.DataFrame) -> None:
        """

        """
        self.final_data: pd.DataFrame = df

    def train_test_spliter(self, val_size: float = 0.04) -> dict:
        """

        """
        test_months = (self.final_data['date_block_num'].unique()).max()
        train_months = test_months - 1

        train = self.final_data[self.final_data['date_block_num'] <= round((1 - val_size) * train_months)]

        val = self.final_data[(self.final_data['date_block_num'] > train_months - round(val_size * train_months)) & (
                self.final_data['date_block_num'] < test_months)]

        test = self.final_data[self.final_data['date_block_num'].isin([test_months])]

        return {'train': train, 'val': val, 'test': test}

    def cv_spliter(self, max_train_size: int = 12) -> dict:
        """

        """
        test_data = (self.final_data['date_block_num'].unique()).max()
        to_train = self.final_data[self.final_data['date_block_num'] <= (test_data - 1)]

        tscv = TimeSeriesSplit(n_splits=-2 * max_train_size + 46, max_train_size=max_train_size, test_size=1, gap=0)

        validation_data = [{'train': train_indexes, 'val': val_indexes} for train_indexes, val_indexes in
                           tscv.split(to_train['date_block_num'].unique())]

        return {'validation_indexes': validation_data, 'test_indexes': [test_data]}


class FeatureExtraction:
    """

    """

    def __init__(self) -> None:
        """

        """
        self.df: pd.DataFrame = None
        self.mean_features: list = None
        self.lag_features_to_clip: list = None

    def create_final_data(self, train: pd.DataFrame, test: pd.DataFrame | None = None,
                          make_big: bool = False) -> None:

        """
        **Creates the final DataFrame for training based on the provided train and test DataFrames.**

        This method creates the final DataFrame for analysis by aggregating the train DataFrame
        based on date_block_num, shop_id, and item_id and merging it with the test DataFrame if provided.
        Optionally, it can expand the DataFrame to include all possible combinations of date_block_num,
        shop_id, and item_id if make_big is set to True.

        :param train: The train DataFrame containing historical sales data.
        :param test: The test DataFrame containing data for prediction. Default is None.
        :param make_big: If True, expands the DataFrame to include all possible combinations of
                         date_block_num, shop_id, and item_id. Default is False.
        :return: The final DataFrame for training.
        """

        # Convert data types to optimize memory usage
        train["date_block_num"] = train["date_block_num"].astype(np.int8)
        train["shop_id"] = train['shop_id'].astype(np.int8)
        train["item_id"] = train['item_id'].astype(np.int16)

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
            test["date_block_num"] = 34
            test["date_block_num"] = test["date_block_num"].astype(np.int8)
            test["shop_id"] = test['shop_id'].astype(np.int8)
            test["item_id"] = test['item_id'].astype(np.int16)
            test = test.drop(columns="ID")

            self.df = pd.concat([self.df, test[["date_block_num", "shop_id", "item_id"]]])

        # Fill NaN values with 0 for item_cnt_month and item_revenue_month columns
        self.df['item_cnt_month'] = self.df['item_cnt_month'].fillna(0)

    def add_city_features(self, shop_path: str) -> None:
        """

        """
        shop_data = pd.read_csv(shop_path)
        shop_data["shop_category"] = shop_data['shop_name'].str.split(" ").map(lambda x: x[1])

        self.df = self.df.merge(shop_data[['city', 'shop_category', 'shop_id']], how="left", on=['shop_id'])

    def add_item_categories_features(self, item_categories_path: str, threshold: int = 5) -> None:
        """

        """

        def _make_etc(x: str) -> str:
            if len(item_categories_data[item_categories_data['category'] == x]) >= threshold:
                return x
            else:
                return 'etc'

        item_categories_data = pd.read_csv(item_categories_path)

        item_categories_data['category'] = item_categories_data['category'].apply(_make_etc)

        self.df = self.df.merge(item_categories_data[['category', 'item_category_id']], how="left",
                                on=['item_category_id'])

    def add_mean_features(self, idx_features: list, with_cv_schema: bool = False,
                          validation_indexes: list | None = None) -> None:
        """

        """
        assert (idx_features[0] == 'date_block_num') and len(idx_features) in [2, 3]

        if len(idx_features) == 2:
            feature_name = idx_features[1] + '_mean_sales'
        else:
            feature_name = idx_features[1] + '_' + idx_features[2] + '_mean_sales'

        group = self.df.groupby(idx_features).agg({'item_cnt_month': 'mean'})
        group = group.reset_index()
        group = group.rename(columns={'item_cnt_month': feature_name})

        self.df = self.df.merge(group, on=idx_features, how='left')

        self.mean_features.append(feature_name)

    def add_lag_features(self, idx_features: list, lag_feature: list, nlags: int = 3, clip: bool = False,
                         with_cv_schema: bool = False, validation_indexes: list | None = None) -> None:
        """

        """
        df_temp = self.df[idx_features + [lag_feature]].copy()

        for i in range(1, nlags + 1):
            lag_feature_name = lag_feature + '_lag' + str(i)

            df_temp.columns = idx_features + [lag_feature_name]

            df_temp['date_block_num'] += 1
            self.df = self.df.merge(df_temp.drop_duplicates(), on=idx_features, how='left')
            self.df[lag_feature_name] = self.df[lag_feature_name].fillna(0)

            if clip:
                self.lag_features_to_clip.append(lag_feature_name)

    def add_mean_price(self, sales_path: str, with_cv_schema: bool = False,
                       validation_indexes: list | None = None) -> None:
        """

        """
        sales_data = pd.read_csv(sales_path)
        if with_cv_schema:
            for row in validation_indexes:
                indexes_to_calculating = np.append(row['train'], row['val'])

                group = sales_data[sales_data['date_block_num'].isin(indexes_to_calculating)].groupby(
                    ['date_block_num', 'shop_id', 'item_id'], as_index=False).agg(
                    item_mean_price=pd.NamedAgg(column="item_price", aggfunc="mean"))

                self.df = self.df.merge(group, how="left", on=['date_block_num', 'shop_id', 'item_id'])

                # self.df['item_mean_price'] = self.df['item_mean_price'].fillna(0)
                self.df = self.df.rename(
                    columns={'item_mean_price': 'item_mean_price_' + str(indexes_to_calculating.max())})

        else:
            group = sales_data.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False).agg(
                item_mean_price=pd.NamedAgg(column="item_price", aggfunc="mean"))

            self.df = self.df.merge(group, how="left", on=['date_block_num', 'shop_id', 'item_id'])

            self.df['item_mean_price'] = self.df['item_mean_price'].fillna(0)

    @staticmethod
    def _ts_stationarity_check(df: pd.DataFrame, column: str, significance_level: float = 0.05) -> str:
        """

        """
        stationarity: str = "not defined"

        try:
            adf_stats: tuple = adfuller(df[column])
        except ValueError:
            print("Invalid input, x is constant")
            stationarity = "stationary"
            return stationarity

        try:
            kpss_stats: tuple = kpss(df[column])
        except OverflowError:
            print("OverflowError: cannot convert float infinity to integer.")
            return stationarity

        significance_level_perc: str = str(int(significance_level * 100)) + "%"

        try:
            if (kpss_stats[0] < kpss_stats[3][significance_level_perc]) and (
                    adf_stats[0] < adf_stats[4][significance_level_perc]) and \
                    (kpss_stats[1] > significance_level) and (adf_stats[1] < significance_level):
                stationarity = "stationary"
            elif (kpss_stats[0] > kpss_stats[3][significance_level_perc]) and (
                    adf_stats[0] > adf_stats[4][significance_level_perc]) and \
                    (kpss_stats[1] < significance_level) and (adf_stats[1] > significance_level):
                stationarity = "non-stationary"
            elif (kpss_stats[0] < kpss_stats[3][significance_level_perc]) and (
                    adf_stats[0] > adf_stats[4][significance_level_perc]) and \
                    (kpss_stats[1] > significance_level) and (adf_stats[1] > significance_level):
                stationarity = "trend stationary"
            elif (kpss_stats[0] > kpss_stats[3][significance_level_perc]) and (
                    adf_stats[0] < adf_stats[4][significance_level_perc]) and \
                    (kpss_stats[1] < significance_level) and (adf_stats[1] < significance_level):
                stationarity = "difference stationary"
            else:
                print("The stationarity of the series cannot be verified.")

        except KeyError:
            if (kpss_stats[1] > significance_level) and (adf_stats[1] < significance_level):
                stationarity = "stationary"
            elif (kpss_stats[1] < significance_level) and (adf_stats[1] > significance_level):
                stationarity = "non-stationary"
            elif (kpss_stats[1] > significance_level) and (adf_stats[1] > significance_level):
                stationarity = "trend stationary"
            elif (kpss_stats[1] < significance_level) and (adf_stats[1] < significance_level):
                stationarity = "difference stationary"
            else:
                print("The stationarity of the series cannot be verified.")

        except Exception:
            print("The stationarity of the series cannot be verified with the given parameters.")

        return stationarity

    def ts_nonstatinarity_processing(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """

        """
        while self._ts_stationarity_check(df, column) != "stationary":
            diff_column = np.diff(df[column])
            diff_column = np.append(diff_column, np.mean(diff_column))
            df[column] = diff_column

        return df

    def load_data(self, file_name: str) -> None:
        """

        """

        if not os.path.exists(save_final_to):
            os.makedirs(save_final_to)

        self.df.to_csv(save_final_to + file_name + '.csv',
                       index=False, date_format='%d.%m.%Y')
        print("INFO: File {0} was successfully saved".format(file_name))

    def get_data(self) -> pd.DataFrame:
        """
        **Retrieves the  DataFrame.**

        This method returns the DataFrame.

        :return: DataFrame.
        """
        return self.df


def train_model(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, in_features: list[str], target: list[str],
                cat_features: list[str] | None = None) -> list[float]:
    """

    """
    train_data = cb.Pool(train[in_features],
                         train[target],
                         cat_features=cat_features)
    val_data = cb.Pool(val[in_features],
                       val[target],
                       cat_features=cat_features)

    model = cb.CatBoostRegressor(cat_features=cat_features, task_type="GPU", random_seed=42)
    model.fit(train_data, eval_set=val_data, use_best_model=True, verbose=True, early_stopping_rounds=70)
    return model.predict(test[in_features])


def train_cv_model(data: pd.DataFrame, validation_indexes: list, test_indexes: list, in_features: list[str],
                   target: list[str], cat_features: list[str] | None = None) -> None:
    """

    """
    test = data[data['date_block_num'].isin(test_indexes)]

    model = cb.CatBoostRegressor(task_type="GPU", random_seed=42)

    for idx, row in validation_indexes:
        print("Iteration {} of {}".format(idx, len(validation_indexes)))
        train_df = data[data['date_block_num'].isin(row['train'])]
        val_df = data[data['date_block_num'].isin(row['val'])]

        train_data = cb.Pool(train_df[in_features],
                             train_df[target],
                             cat_features=cat_features)
        val_data = cb.Pool(val_df[in_features],
                           val_df[target],
                           cat_features=cat_features)

        model.fit(train_data, eval_set=val_data, use_best_model=True, verbose=True, early_stopping_rounds=70)

    return model.predict(test[in_features])
