import argparse
from sys import path
import os

path.append('.')
from src.path_utils import save_final_to, catboost_model_path, final_full_data_path
from src.modeling.validation_schema import ValidationSchema
from src.modeling.training_schema import TrainingModel
from src.modeling.utils import create_kaggle_data, read_json_file, downcast
import pandas as pd
import catboost as cb

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script for generating predictions using a trained CatBoost model.")

    parser.add_argument("--final_data_path",
                        default=final_full_data_path,
                        type=str,
                        help="Path to the file containing the final processed data (default: final_full_data_path).")

    parser.add_argument("--catboost_model_path",
                        default=catboost_model_path,
                        type=str,
                        help="Path to the file containing the trained CatBoost model (default: catboost_model_path).")

    parser.add_argument("--save_kaggle_path",
                        default=save_final_to,
                        type=str,
                        help=f"Path to save the prediction files in a format "
                             f"suitable for Kaggle submission (default: save_final_to).")
    parser.add_argument("--kaggle_file_name",
                        default="kaggle_predictions",
                        type=str,
                        help="Name of the file to save Kaggle predictions (default: 'kaggle_predictions').")

    args = parser.parse_args()

    DATA_PATH = args.final_data_path
    MODEL_PATH = args.catboost_model_path
    KAGGLE_PATH = args.save_kaggle_path
    KAGGLE_FILE_NAME = args.kaggle_file_name

    try:
        with os.scandir(save_final_to) as entries:
            pass
    except FileNotFoundError:
        print("[ERROR]: The directory does not exist. Please run train_model.py before running this script.")
        raise

    try:
        with open(DATA_PATH, 'r'):
            pass
    except FileNotFoundError:
        print(f"[ERROR]: The file does not exist."
              f" Please choose another directory and try again or run train_model.py before running this script.")
        raise

    try:
        with open(MODEL_PATH, 'r'):
            pass
    except FileNotFoundError:
        print(f"[ERROR]: The model does not exist."
              f" Please choose another directory and try again or run train_model.py before running this script.")
        raise

    print("[INFO]: Reading final data...")
    df = pd.read_csv(DATA_PATH)
    df = downcast(df)
    print("[INFO]: Final data was successfully read.")

    model_params = read_json_file()
    vs = ValidationSchema(data=df)

    if model_params["training_type"] == "full":
        validation_dict = vs.train_test_spliter()
    else:
        validation_dict = vs.cv_spliter()

    tm = TrainingModel(data=df, validation_dict=validation_dict, params=model_params["model_params"])

    model = cb.CatBoostRegressor()

    print("[INFO]: Loading model...")
    model.load_model(MODEL_PATH)
    print("[INFO]: Model was successfully loaded.")

    predictions = tm.make_predictions(model=model, in_features=model_params['training_feature']['in_features'],
                                      lag_features=model_params['training_feature']['lag_features'])
    print("[INFO]: Predictions was successfully generated.")

    create_kaggle_data(predictions=predictions, file_name=KAGGLE_FILE_NAME, save_path=KAGGLE_PATH)
