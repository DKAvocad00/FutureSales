import pandas as pd
import itertools
import numpy as np
from path_utils import save_final_to


class FeatureExtraction:
    def __init__(self):
        self.df = None

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
            item_cnt_month=pd.NamedAgg(column="item_cnt_day", aggfunc="sum"),
            item_revenue_month=pd.NamedAgg(column="revenue", aggfunc="sum"),
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
        self.df['item_revenue_month'] = self.df['item_revenue_month'].fillna(0)

    def load_data(self, file_name: str) -> None:
        """
        **Saves the transformed DataFrame to a CSV file.**

        The method stores the transformed DataFrame into a new CSV file at the specified location.

        :param file_name: The name of the CSV file to save.
        :return: None
        """

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
