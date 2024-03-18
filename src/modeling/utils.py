from typing import Any

import pandas as pd
import os
import json
from src.path_utils import json_path


def read_json_file(file_path: str = json_path) -> dict:
    """
    **Read data from a JSON file.**

    This function reads data from a JSON file and returns it as a dictionary.

    :param file_path: The path to the JSON file.
    :return: A dictionary containing the data read from the JSON file.
             Returns None if the file is not found or if there is an error decoding JSON.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Unable to decode JSON from file '{file_path}'.")
        return None


def write_to_json(key: Any, value: Any, save_to: str = json_path) -> None:
    """
    **Write data to a JSON file.**

    This function updates a JSON file with a new key-value pair or creates a new file if it doesn't exist.
    The key parameter can accept specific string values such as "model_params", "training_type", or "training_feature",
    or it can be a dictionary key.

    :param key: The key to be added or updated in the JSON file. It can be a specific string value such as
                "model_params", "training_type", or "training_feature", or a custom dictionary key.
    :param value: The value corresponding to the key.
    :param save_to: The path to the JSON file. Default is "data.json".
    :return: None
    """

    # Load existing data from JSON file if it exists, otherwise initialize as empty dictionary
    try:
        with open(save_to, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}

    # Update data with new key-value pair
    data[key] = value

    # Write updated data back to JSON file
    with open(save_to, 'w') as file:
        json.dump(data, file, indent=4)

    print('[INFO]: Json file was successfully written.')


def downcast(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    **Downcast the DataFrame's numeric columns to reduce memory usage.**

    This function downcasts the numeric columns of the DataFrame to the smallest
    possible data type that can accurately represent the data. It helps reduce
    memory usage of the DataFrame.

    :param df: The DataFrame to downcast.
    :param verbose: Whether to print the percentage compression achieved. Default is True.
    :return: The DataFrame with downcasted columns.
    """

    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        dtype_name = df[col].dtype.name
        if dtype_name == 'object':
            pass
        elif dtype_name == 'bool':
            df[col] = df[col].astype('int8')
        elif dtype_name.startswith('int') or (df[col].round() == df[col]).all():
            df[col] = pd.to_numeric(df[col], downcast='integer')
        else:
            df[col] = pd.to_numeric(df[col], downcast='float')
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('{:.1f}% compressed'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def create_kaggle_data(predictions: list[float], file_name: str, save_path: str) -> None:
    """
    **Create a Kaggle submission file from the predictions.**

    This function creates a Kaggle submission file from the provided predictions
    and saves it to the specified path with the given file name.

    :param predictions: A list of predicted values to be included in the submission.
    :param file_name: The name of the submission file.
    :param save_path: The path where the submission file will be saved.
    :return: None
    """
    kaggle = pd.DataFrame({'item_cnt_month': predictions})

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    kaggle.to_csv(save_path + file_name + '.csv',
                  index=True, index_label="ID")

    print("[INFO]: File {0} was successfully saved".format(file_name))
