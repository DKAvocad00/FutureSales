import os
from typing import Callable
from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval
import pandas as pd
import catboost as cb
from src.path_utils import save_model_to
from sklearn.metrics import mean_squared_error
import numpy as np
from functools import wraps
from catboost.utils import get_gpu_device_count
from src.modeling.utils import write_to_json


def validate_validation_type(expected_type: str) -> Callable:
    """
    **Decorator to validate the validation type technique used in a method.**

    This decorator checks if the validation type technique used in the method matches the expected type.

    :param expected_type: The expected validation type technique.
    :return: The decorator function.
    """

    def decorator(func: Callable) -> Callable:
        """
        **Inner decorator function.**

        :param func: The function to be decorated.
        :return: The wrapper function.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            """
            **Wrapper function to perform validation of validation type technique.**

            :param self: The class instance.
            :param args: Positional arguments passed to the method.
            :param kwargs: Keyword arguments passed to the method.
            :raises ValueError: If the validation type technique does not match the expected type.
            :return: The result of the decorated function.
            """
            if self.validation_dict['validation_type'] != expected_type:
                raise ValueError("Invalid validation type technique. Requires {} but got {}".format(
                    expected_type, self.validation_dict["validation_type"])
                )
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class TrainingModel:
    """
    A class for training CatBoostRegressor models and performing hyperparameter search.
    """

    def __init__(self, data: pd.DataFrame, validation_dict: dict, params: dict) -> None:
        """
        **Initialize the TrainingModel object.**

        :param data: The DataFrame containing the training data.
        :param validation_dict: A dictionary containing validation indexes.
        :param params: A dictionary containing CatBoost parameters.
        """
        self.data: pd.DataFrame = data
        self.validation_dict: dict = validation_dict
        self.params: dict = params
        self.task_type: dict = self._check_device()

    @validate_validation_type('full')
    def train_model(self, in_features: list[str], lag_features: dict[list], target: list[str], cat_features: list[str],
                    save_model: bool = False, model_name: str | None = None) -> list[float]:
        """
        **Train a CatBoostRegressor model.**

        This method trains a CatBoostRegressor model using the specified input features, lagged features, and target variable.
        The lagged features are applied to the input features before training the model. Optionally, the trained model can be
        saved to a file.

        :param in_features: A list of input features used for training the model.
        :param lag_features: A dictionary containing lagged features for each fold.
        :param target: The target variable used for training.
        :param cat_features: A list of categorical features.
        :param save_model: A boolean indicating whether to save the trained model. Default is False.
        :param model_name: The name of the file to save the trained model. If None, a default name will be used. Default is None.
        :return: A list of predicted values for the test dataset.
        """

        # Combine input features with lagged features
        in_features = in_features + lag_features["0"]

        model = cb.CatBoostRegressor(**self.params, **self.task_type, random_seed=42)

        # Split the data into train, validation, and test sets
        train = self.data[
            self.data['date_block_num'].isin([row['train'] for row in self.validation_dict['validation_indexes']][0])]
        val = self.data[
            self.data['date_block_num'].isin([row['val'] for row in self.validation_dict['validation_indexes']][0])]

        test = self.data[self.data['date_block_num'].isin(self.validation_dict['test_indexes'])]

        # Prepare train and validation data
        train_data = cb.Pool(train[in_features],
                             train[target],
                             cat_features=cat_features)
        val_data = cb.Pool(val[in_features],
                           val[target],
                           cat_features=cat_features)

        # Train the model
        model.fit(train_data, eval_set=val_data, use_best_model=True, verbose=True, early_stopping_rounds=30)

        if save_model:
            self._save_training_model(model=model, model_name=model_name)

        return model.predict(test[in_features])

    @validate_validation_type('cv')
    def train_cv_model(self, in_features: list[str], lag_features: dict[list], target: list[str],
                       cat_features: list[str], save_model: bool = False, model_name: str | None = None) -> list[float]:
        """
        **Train a CatBoostRegressor model using cross-validation.**

        This method trains a CatBoostRegressor model using cross-validation with the specified input features, lagged features,
        and target variable. The lagged features are applied to the input features for each fold before training the model.
        Optionally, the trained model can be saved to a file.

        :param in_features: A list of input features used for training the model.
        :param lag_features: A dictionary containing lagged features for each fold.
        :param target: The target variable used for training.
        :param cat_features: A list of categorical features.
        :param save_model: A boolean indicating whether to save the trained model. Default is False.
        :param model_name: The name of the file to save the trained model. If None, a default name will be used. Default is None.
        :return: A list of predicted values for the test dataset.
        """

        model = cb.CatBoostRegressor(**self.params, **self.task_type, random_seed=42)

        # Select the test dataset
        test = self.data[self.data['date_block_num'].isin(self.validation_dict['test_indexes'])]

        # Determine the number of folds
        folds_number = len(self.validation_dict['validation_indexes'])

        # Iterate over each fold in the cross-validation
        for fold, row in self.validation_dict['validation_indexes']:
            print("Iteration {} of {}".format(fold, folds_number))

            # Combine input features with lagged features for the current fold
            in_features_fold = in_features + lag_features[fold]

            # Split the data into train and validation sets for the current fol
            train = self.data[self.data['date_block_num'].isin(row['train'])]
            val = self.data[self.data['date_block_num'].isin(row['val'])]

            # Prepare train and validation data for the current fold
            train_data = cb.Pool(train[in_features_fold],
                                 train[target],
                                 cat_features=cat_features)
            val_data = cb.Pool(val[in_features_fold],
                               val[target],
                               cat_features=cat_features)

            # Train the model for the current fold
            model.fit(train_data, eval_set=val_data, use_best_model=True, verbose=True, early_stopping_rounds=30)

        # Save the trained model if requested
        if save_model:
            self._save_training_model(model=model, model_name=model_name)

        # Get the lag features for the last fold
        _, last_value = lag_features.popitem()

        return model.predict(test[in_features + last_value])

    @validate_validation_type('full')
    def parameter_search(self, param_space: dict, in_features: list[str], lag_features: dict[list], target: list[str],
                         cat_features: list[str]) -> dict:
        """
        **Perform hyperparameter search for the CatBoost model.**

        This method performs hyperparameter search for the CatBoost model using the specified parameter space.

        :param param_space: Dictionary defining the hyperparameter space.
        :param in_features: List of input features.
        :param lag_features: Dict of lag input features
        :param target: List of target features.
        :param cat_features: List of categorical features.
        :return: Dictionary containing the best hyperparameters found during the search.
        """

        in_features = in_features + lag_features[0]

        # Initialize a Trials object to keep track of the hyperparameter optimization process
        trials = Trials()

        # Split the data into train and test sets based on the predefined validation indexes
        train = self.data[
            self.data['date_block_num'].isin([row['train'] for row in self.validation_dict['validation_indexes']][0])]

        test = self.data[
            self.data['date_block_num'].isin([row['val'] for row in self.validation_dict['validation_indexes']][0])]

        # Create CatBoost Pool objects for training and validation data
        train_data = cb.Pool(data=train[in_features],
                             label=train[target],
                             cat_features=cat_features)

        # Define the hyperparameter tuning function
        def _hyperparameter_tuning(params: dict) -> dict:
            """
            **Perform hyperparameter tuning for the CatBoost model.**

            This function trains a CatBoost model with the given hyperparameters and calculates the RMSE.

            :param params: Hyperparameters to tune.
            :return: Dictionary containing the loss (RMSE) and the optimization status.
            """
            model = cb.CatBoostRegressor(**params, **self.task_type, random_seed=42)
            model.fit(train_data, early_stopping_rounds=30, verbose=True)
            preds = model.predict(test[in_features])

            # Calculate the root mean squared error (RMSE) between the predictions and the actual target values
            rmse = np.sqrt(mean_squared_error(test[target], preds))

            return {'loss': rmse, 'status': STATUS_OK}

        # Use hyperopt library to perform hyperparameter optimization
        params = fmin(fn=_hyperparameter_tuning, space=param_space, algo=tpe.suggest, max_evals=20, trials=trials,
                      verbose=True)

        self.params = space_eval(param_space, params)

        write_to_json(key="model_params", value=self.params)

        return self.params

    def make_predictions(self, model: cb.CatBoostRegressor, in_features: list[str],
                         lag_features: dict[list]) -> list[float]:
        """
        **Generate predictions using the trained model.**

        This method generates predictions using the provided CatBoostRegressor model for the test dataset.
        It adjusts the input features according to the validation type specified in the validation dictionary.
        If the validation type is 'full', it uses lagged features from the first fold. Otherwise, it uses lagged features
        from the last fold.

        :param model: The trained CatBoostRegressor model used for prediction.
        :param in_features: A list of input features used for prediction.
        :param lag_features: A dictionary containing lagged features for each fold.
        :return: A list of predicted target values for the test dataset.
        """

        test = self.data[self.data['date_block_num'].isin(self.validation_dict['test_indexes'])]

        if self.validation_dict["validation_type"] == "full":
            in_features = in_features + lag_features["0"]

        else:

            _, last_value = lag_features.popitem()
            in_features = in_features + last_value

        return model.predict(test[in_features])

    @staticmethod
    def _save_training_model(model: cb.CatBoostRegressor, model_name: str) -> None:
        """
        **Save the trained model**.

        This method saves the trained CatBoost model to a file.

        :param model: The trained CatBoost model to be saved.
        :param model_name: The name of the file to which the model will be saved.
        :return: None
        """
        if not os.path.exists(save_model_to):
            os.makedirs(save_model_to)

        model.save_model(save_model_to + model_name)

        print(f"[INFO]: Model {model} was successfully saved")

    @staticmethod
    def _check_device() -> dict:
        """
        **Check the available devices and return the task type.**

        :return: If a GPU is available, return {'task_type': "GPU"}. Otherwise,
                 return {'task_type': "CPU", 'thread_count': -1}.
        """
        gpu_device_count = get_gpu_device_count()
        if gpu_device_count != 0:
            return {'task_type': "GPU"}

        return {'task_type': "CPU",
                'thread_count': -1}
