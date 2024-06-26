import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import seaborn as sns
import numpy as np
import re
import os
from src.path_utils import save_interm_to


class ETL:
    """
    **Class for performing the Extract, Transform, and Load (ETL) process on a given dataset.**

    This class encapsulates the functionality to read a dataset from a CSV file, transform it based on various options,
    and save the transformed data to a new CSV file.
    """

    def __init__(self, data_path: str) -> None:
        """
        **Initializes the ETL class with the provided data path.**

        :param data_path: The path to the CSV file containing the dataset.
        """

        self.df: pd.DataFrame = pd.read_csv(data_path)
        self.option: list[str] = ['shops', 'sales', 'items', 'test', 'item_categories']
        print("[INFO]: {0} rows and {1} columns has been read from {2}".format(self.df.shape[0],
                                                                               self.df.shape[1],
                                                                               os.path.basename(
                                                                                   data_path)
                                                                               ))

    def transform_data(self, option: str) -> None:
        """
        **Transforms the dataset based on the specified option.**

        The method applies a series of transformations to the dataset, tailored to the specific needs
        identified by the provided option.

        :param option: The transformation option ('shop', 'train', 'item', 'test', 'category').
        :return: None
        """
        if option.lower() not in self.option:
            raise ValueError(
                "Invalid option. Please choose one of: {}".format(self.option))

        if option.lower() == 'sales':
            self.train_fix(drop_extreme=True)
            self.isolation_forest(['item_price', 'item_cnt_day'], info=False, change=True)

        if option.lower() == 'test':
            self.test_fix()

        if option.lower() == 'items':
            self.item_fix()

        if option.lower() == 'shops':
            self.shop_fix()

        if option.lower() == 'item_categories':
            self.item_category_fix()

    # Items fix
    def item_fix(self) -> None:
        """
        **Fixes issues related to item data.**

        This method cleans item names by removing special characters, brackets, and extra spaces,
        and converts them to lowercase.
        """

        # fix item name
        def _clean_name(string: str) -> str:
            """
            Cleans the item name.

            :param string: The item name to clean.
            :return: The cleaned item name.
            """

            string = re.sub(r"\[.*?\]", "", string)
            string = re.sub(r"\(.*?\)", "", string)
            string = re.sub(r"[^A-ZА-Яa-zа-я0-9 ]", "", string)
            string = re.sub(r"\s{2,}", " ", string)
            string = string.lower()
            return string

        self.df["cleaned_name"] = self.df["item_name"].apply(_clean_name)

    # Shops fix
    def shop_fix(self) -> None:
        """
        **Fixes issues related to shop data.**

        This method corrects shop IDs and adds a new column 'city' based on the shop name.
        """

        # shop id fix
        shops_id_fix: dict[int] = {0: 57, 1: 58, 10: 11, 40: 39}
        self.df.drop(index=shops_id_fix.keys(), inplace=True)

        # add new column 'city'
        self.df.loc[self.df.shop_name == 'Сергиев Посад ТЦ "7Я"',
        "shop_name"] = 'СергиевПосад ТЦ "7Я"'
        self.df['city'] = self.df['shop_name'].str.split(
            ' ').map(lambda x: x[0])

    # train fix
    def train_fix(self, drop_extreme: bool = False) -> None:
        """
        **Fixes issues related to training data.**

        This method drops duplicates, fixes datetime format, corrects shop IDs, and optionally drops extreme values.

        :param drop_extreme: Whether to drop extreme values.
        """

        # drop duplicates
        self.df.drop_duplicates(inplace=True)

        # datetime fix
        self.df['date'] = pd.to_datetime(self.df['date'], format='%d.%m.%Y')

        # add revenue column
        self.df['revenue'] = self.df['item_price'] * self.df['item_cnt_day']

        # shop id fix
        shops_id_fix: dict[int] = {0: 57, 1: 58, 10: 11, 40: 39}
        self.df = self.df.replace({'shop_id': shops_id_fix})

        # drop extreme or negative values
        if drop_extreme:
            self.df = self.df.loc[(self.df['item_price'] < 50000) & (
                    self.df['item_price'] > 0)]
            self.df = self.df.loc[(self.df['item_cnt_day'] < 700) & (
                    self.df['item_cnt_day'] > 0)]

    # test fix
    def test_fix(self) -> None:
        """
        **Fixes issues related to test data.**

        This method corrects shop IDs in the test data.
        """

        # shop id fix
        shops_id_fix: dict[int] = {0: 57, 1: 58, 10: 11, 40: 39}
        self.df = self.df.replace({'shop_id': shops_id_fix})

    # item category fix
    def item_category_fix(self) -> None:
        """
        **Fixes issues related to item category data.**

        This method adds a new column 'category' based on the item category name.
        """

        # add new column global category
        self.df['category'] = self.df['item_category_name'].apply(
            lambda x: x.split()[0])

    # Outliers
    def z_score(self, column: str, info: bool = True, change: bool = False) -> None:
        """
        **Detects outliers using z-score method.**

        :param column: The column name to detect outliers for.
        :param info: Whether to display information about detected outliers.
        :param change: Whether to remove outliers from the DataFrame.
        """

        self.df['z_score'] = (self.df[column] - self.df[column].mean()) / self.df[column].std()
        outliers = self.df[(self.df['z_score'] < -3) | (self.df['z_score'] > 3)]

        if info:
            print(f"{'=' * 50}\n\nZ_score outliers detection:\n")
            print("Number of outliers in {0}: {1}".format(
                column, outliers.shape[0]))
            print("Outlier share in {0}: {1}%".format(column, round(
                (outliers.shape[0] / self.df.shape[0] * 100), 3)))
            print(f"\n{'=' * 50}")

        if change:
            self.df = self.df[(self.df['z_score'] > -3) &
                              (self.df['z_score'] < 3)]
            self.df.drop('z_score', axis=1, inplace=True)

    def outlier_detect_IQR(self, columns: list[str], threshold: float = 3, info: bool = True,
                           change: bool = False) -> None:
        """
        **Detects outliers using the IQR (InterQuartile Range) method.**

        :param columns: List of column names to detect outliers for.
        :param threshold: Threshold value for detecting outliers.
        :param info: Whether to display information about detected outliers.
        :param change: Whether to remove outliers from the DataFrame.
        """

        outlier_indices: list = []
        outliers: list = []
        for col in columns:
            IQR = self.df[col].quantile(0.75) - self.df[col].quantile(0.25)
            lower_fence = self.df[col].quantile(0.25) - (IQR * threshold)
            upper_fence = self.df[col].quantile(0.75) + (IQR * threshold)
            tmp = pd.concat([self.df[col] > upper_fence,
                             self.df[col] < lower_fence], axis=1)
            outlier_index = tmp.any(axis=1)
            outlier_indices.extend(outlier_index[outlier_index].index)
            outliers.extend(self.df.loc[outlier_index, col])

            if info:
                print(f"{'=' * 50}\n\nIQR outliers detection:\n")
                print("Number of outliers in {0}: {1}".format(
                    col, outlier_index.sum()))
                print("Outlier share in {0}: {1}%".format(col,
                                                          round((outlier_index.sum() / len(outlier_index) * 100), 3)))
                print(f"\n{'=' * 50}\n")

        if change:
            self.df = self.df[~self.df.index.isin(outlier_indices)]

    def isolation_forest(self, columns: list[str], info: bool = True, change: bool = False) -> None:
        """
        **Detects outliers using the Isolation Forest algorithm.**

        :param columns: List of column names to detect outliers for.
        :param info: Whether to display information about detected outliers.
        :param change: Whether to remove outliers from the DataFrame.
        """

        clf = IsolationForest(n_jobs=-1)
        labels = clf.fit_predict(self.df[columns].to_numpy())

        if info:
            print(f"{'=' * 50}\n\nIsolation forest outliers detection:\n")
            print("Number of outliers: {0}".format(
                np.count_nonzero(labels == -1)))
            print("Outlier share: {0}%".format(
                round((np.count_nonzero(labels == -1) / len(labels) * 100), 3)))
            print(f"\n{'=' * 50}")

        if change:
            outliers = [i for i in range(0, len(labels)) if labels[i] == -1]
            self.df = self.df.drop(
                self.df.iloc[outliers].index, axis=0).copy(deep=True)

    def get_data(self) -> pd.DataFrame:
        """
        **Retrieves the  DataFrame.**

        This method returns the DataFrame.

        :return: DataFrame.
        """

        return self.df

    def load_data(self, file_name: str) -> None:
        """
        **Save the DataFrame to a CSV file.**

        This method saves the DataFrame to a CSV file with the specified file name.

        :param file_name: The name of the CSV file to save.
        :return: None
        """

        if not os.path.exists(save_interm_to):
            os.makedirs(save_interm_to)

        self.df.to_csv(save_interm_to + file_name + '.csv',
                       index=False, date_format='%d.%m.%Y')
        print("[INFO]: File {0} was successfully saved".format(file_name))


class DQC:
    """
    Data Quality Check (DQC) class for performing  quality assurance operations on a given DataFrame.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        **Initializes the DQC class with the provided DataFrame.**

        :param df: The DataFrame to perform quality control operations on.
        """

        self.df: pd.DataFrame = df

    # Statistic
    def statistics(self) -> None:
        """
        **Displays basic statistics and information about the DataFrame**.

        This method provides information about the DataFrame, including its structure, example data,
        number of unique values, and number of duplicated rows.
        """

        # Print information about data
        print("Information about data:\n")
        print(f"{self.df.info()}\n\n{'=' * 50}\n")

        # Print some examples of data
        print("Some examples of data:\n")
        print(f"{self.df.head(10)}\n\n{'=' * 50}\n")

        # Print number of unique data
        print("Number of unique data:\n")
        print(f"{self.df.nunique()}\n\n{'=' * 50}\n")

        # Print number of duplicated data
        print(f"Number of dublicated data: {self.df.duplicated().sum()}")

    def describe_matrix(self, column_list: list[str]) -> None:
        """
        **Displays descriptive statistics for specified columns.**

        :param column_list: List of column names.
        """

        print(self.df[column_list].describe().T)

    # Graphics
    def boxplots(self, columns: dict[str]) -> None:
        """
        **Displays box plots for specified columns.**

        :param columns: List of column names.
        """

        _ = plt.figure(figsize=(10, 5))
        for n, i in enumerate(columns):
            plt.subplot(1, len(columns), n + 1)
            plt.xlabel(i)
            sns.boxplot(data=self.df, x=i)
        plt.tight_layout()
        plt.show()

    def histplots(self, columns: dict[str]) -> None:
        """
        **Displays histograms for specified columns.**

        :param columns: List of column names.
        """

        _ = plt.figure(figsize=(10, 5))
        for n, i in enumerate(columns):
            plt.subplot(1, len(columns), n + 1)
            plt.xlabel(i)
            sns.histplot(data=self.df, x=i, kde=True)
        plt.tight_layout()
        plt.show()

    def get_data(self) -> pd.DataFrame:
        """
        **Retrieves the  DataFrame.**

        This method returns the DataFrame.

        :return: DataFrame.
        """

        return self.df
