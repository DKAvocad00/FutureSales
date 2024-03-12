import math
import pandas as pd


def asMinutes(s: float) -> str:
    """
    **Converts time in seconds to a string representation in minutes and seconds.**

    This function takes a time duration in seconds and converts it into a human-readable string
    representation in minutes and seconds format.

    :param s: The time duration in seconds.
    :return: A string representing the time duration in minutes and seconds format (e.g., '5m 30s').
    """
    m = math.floor(s / 60)
    s -= m * 60
    return '{}m {}s'.format(round(m), round(s))


def create_kaggle_data(predictions: list[float], file_name: str, save_path: str) -> None:
    kaggle = pd.DataFrame({'item_cnt_month': predictions})
    kaggle.to_csv(save_path + file_name + '.csv',
                  index=True, index_label="ID")

    print("INFO: File {0} was successfully saved".format(file_name))
