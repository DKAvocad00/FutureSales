import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


class ValidationSchema:
    """

    """

    def __init__(self, data: str | pd.DataFrame) -> None:
        """

        """
        if isinstance(data, str):
            self.final_data: pd.DataFrame = pd.read_csv(data)
        else:
            self.final_data: pd.DataFrame = data

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
