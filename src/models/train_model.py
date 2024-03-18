from sys import path

path.append('.')
from src.path_utils import test_path, shops_path, sales_path, items_path, item_categories_path
from src.path_utils import sales_fix_path, test_fix_path, shops_fix_path, item_categories_fix_path, items_fix_path
import argparse
from src.modeling.building_features import FeatureModeling
from src.modeling.validation_schema import ValidationSchema
from src.modeling.training_schema import TrainingModel
from src.modeling.utils import read_json_file
from src.data.make_dataset import ETL

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script for data processing and model training.")

    parser.add_argument("--make_data_big",
                        default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Flag indicating whether to process data in full or just a subset (default: True).")

    parser.add_argument("--validation_type",
                        default="full",
                        type=str,
                        help=f"Type of validation to perform. Options: 'full' (train-test split)"
                             f" or 'cv' (cross-validation).")

    parser.add_argument("--final_data_name",
                        default="final_full_data",
                        type=str,
                        help="Name for the final processed data file (default: 'final_full_data').")

    parser.add_argument("--save_model",
                        default=True,
                        type=argparse.BooleanOptionalAction,
                        help="Flag indicating whether to save the trained model (default: True).")

    parser.add_argument("--final_model_name",
                        default="catboost_full",
                        type=str,
                        help="Name for the final trained model file (default: 'catboost_full').")

    args = parser.parse_args()

    MAKE_BIG = args.make_data_big
    VALIDATION_TYPE = args.validation_type
    FILE_NAME = args.final_data_name
    SAVE_MODEL = args.save_model
    FINAL_MODEL_NAME = args.final_model_name

    if VALIDATION_TYPE.lower() not in ["full", "cv"]:
        raise ValueError("Invalid validation type option. Please choose 'full' or 'cv'")

    file_paths = [test_path, shops_path, sales_path, items_path, item_categories_path]
    file_fix_paths = [sales_fix_path, test_fix_path, shops_fix_path, item_categories_fix_path, items_fix_path]

    for file_fix_path in file_fix_paths:
        try:
            with open(file_fix_path, 'r'):
                pass
        except FileNotFoundError:

            print("[ERROR]: Required fix file not found. Trying to process the missing file...")

            try:
                for file_path in file_paths:
                    with open(file_path, 'r'):
                        pass
            except FileNotFoundError:
                print("[ERROR]: One or more required row files not found."
                      "Please check if you have data in data/row/ path.")
                raise
            prefix_to_search = file_fix_path.split("/")[-1]
            prefix_to_search = prefix_to_search.split("_fix.csv")[0]

            matching_paths = [path for path in file_paths if path.split("/")[-1].startswith(prefix_to_search)]

            file = ETL(matching_paths[0])
            file.transform_data(prefix_to_search)
            file.load_data(file_name=prefix_to_search + "_fix")

            print("[INFO]: Successfully processing missing file.")

    fe = FeatureModeling(sales_path=sales_fix_path, shop_path=shops_fix_path, item_path=items_fix_path,
                         item_categories_path=item_categories_fix_path, test_path=test_fix_path)

    fe.create_final_data(make_big=MAKE_BIG)

    vs = ValidationSchema(fe.get_data())

    if VALIDATION_TYPE.lower() == "full":
        validation_dict = vs.train_test_spliter()
    else:
        validation_dict = vs.cv_spliter()

    feature_functions = [
        fe.add_mean_price,
        fe.add_city_features,
        fe.add_item_features,
        fe.add_item_categories_features,
        lambda: fe.add_mean_features(idx_features=['date_block_num', 'item_id']),
        lambda: fe.add_mean_features(idx_features=['date_block_num', 'item_id', 'city']),
        lambda: fe.add_lag_features(idx_features=['date_block_num', 'shop_id', 'item_id'], lag_feature='item_cnt_month',
                                    validation_dict=validation_dict, nlags=3),
        lambda: fe.add_lag_features(idx_features=['date_block_num', 'shop_id', 'item_id'],
                                    lag_feature='item_mean_price', validation_dict=validation_dict, nlags=3),
        lambda: fe.add_lag_mean_features(idx_features=['date_block_num', 'shop_id', 'item_id'],
                                         validation_dict=validation_dict, nlags=3, drop_mean_features=True),
        lambda: fe.add_mean_features(idx_features=['date_block_num', 'shop_id', 'item_category_id']),
        lambda: fe.add_lag_mean_features(idx_features=['date_block_num', 'shop_id', 'item_category_id'],
                                         validation_dict=validation_dict, nlags=3, drop_mean_features=True),
        fe.final_process
    ]

    fe.add_features(feature_functions)

    fe.load_data(file_name=FILE_NAME)

    model_params = read_json_file()

    model = TrainingModel(data=fe.get_data(), validation_dict=validation_dict, params=model_params["model_params"])

    print("[INFO]: Training model...")

    if VALIDATION_TYPE.lower() == "full":
        model.train_model(in_features=model_params["training_feature"]["in_features"],
                          lag_features=model_params["training_feature"]["lag_features"],
                          target=model_params["training_feature"]["target"],
                          cat_features=model_params["training_feature"]["cat_features"],
                          save_model=SAVE_MODEL,
                          model_name=FINAL_MODEL_NAME)

    else:
        model.train_cv_model(in_features=model_params["training_feature"]["in_features"],
                             lag_features=model_params["training_feature"]["lag_features"],
                             target=model_params["training_feature"]["target"],
                             cat_features=model_params["training_feature"]["cat_features"],
                             save_model=SAVE_MODEL,
                             model_name=FINAL_MODEL_NAME)

    print("[INFO]: Training of the model was successfully completed")
