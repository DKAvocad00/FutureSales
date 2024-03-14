import pandas as pd
import os


def downcast(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """

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

    """
    kaggle = pd.DataFrame({'item_cnt_month': predictions})

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    kaggle.to_csv(save_path + file_name + '.csv',
                  index=True, index_label="ID")

    print("INFO: File {0} was successfully saved".format(file_name))
