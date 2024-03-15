import os
from typing import Callable
from hyperopt import fmin, tpe, Trials, STATUS_OK
import pandas as pd
import catboost as cb
from path_utils import save_model_to
from sklearn.metrics import mean_squared_error
import numpy as np
from functools import wraps
from catboost.utils import get_gpu_device_count


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

    """

    def __init__(self, data: pd.DataFrame, validation_dict: dict, params: dict) -> None:
        """

        """
        self.data: pd.DataFrame = data
        self.validation_dict: dict = validation_dict
        self.params: dict = params
        self.task_type: dict = self._check_device()

    @validate_validation_type('full')
    def train_model(self, in_features: list[str], target: list[str], cat_features: list[str],
                    save_model: bool = False, model_name: str | None = None) -> list[float]:
        """

        """

        model = cb.CatBoostRegressor(**self.params, **self.task_type, random_seed=42)

        train = self.data[
            self.data['date_block_num'].isin([row['train'] for row in self.validation_dict['validation_indexes']][0])]
        val = self.data[
            self.data['date_block_num'].isin([row['val'] for row in self.validation_dict['validation_indexes']][0])]

        test = self.data[self.data['date_block_num'].isin(self.validation_dict['test_indexes'])]

        train_data = cb.Pool(train[in_features],
                             train[target],
                             cat_features=cat_features)
        val_data = cb.Pool(val[in_features],
                           val[target],
                           cat_features=cat_features)

        model.fit(train_data, eval_set=val_data, use_best_model=True, verbose=True, early_stopping_rounds=15)

        if save_model:
            self._save_training_model(model=model, model_name=model_name)

        return model.predict(test[in_features])

    @validate_validation_type('cv')
    def train_cv_model(self, in_features: list[str], target: list[str], cat_features: list[str],
                       save_model: bool = False, model_name: str | None = None) -> list[float]:
        """

        """

        model = cb.CatBoostRegressor(**self.params, **self.task_type, random_seed=42)

        test = self.data[self.data['date_block_num'].isin(self.validation_dict['test_indexes'])]

        folds_number = len(self.validation_dict['validation_indexes'])

        for idx, row in self.validation_dict['validation_indexes']:
            print("Iteration {} of {}".format(idx, folds_number))

            train = self.data[self.data['date_block_num'].isin(row['train'])]
            val = self.data[self.data['date_block_num'].isin(row['val'])]

            train_data = cb.Pool(train[in_features],
                                 train[target],
                                 cat_features=cat_features)
            val_data = cb.Pool(val[in_features],
                               val[target],
                               cat_features=cat_features)

            model.fit(train_data, eval_set=val_data, use_best_model=True, verbose=True, early_stopping_rounds=70)

        if save_model:
            model = cb.CatBoostRegressor(task_type="GPU", random_seed=42)
            self._save_training_model(model=model, model_name=model_name)

        return model.predict(test[in_features])

    @validate_validation_type('full')
    def parameter_search(self, param_space: dict, in_features: list[str], target: list[str],
                         cat_features: list[str]) -> dict:
        """
        **Perform hyperparameter search for the CatBoost model.**

        This method performs hyperparameter search for the CatBoost model using the specified parameter space.

        :param param_space: Dictionary defining the hyperparameter space.
        :param in_features: List of input features.
        :param target: List of target features.
        :param cat_features: List of categorical features.
        :return: Dictionary containing the best hyperparameters found during the search.
        """

        # Initialize a Trials object to keep track of the hyperparameter optimization process
        trials = Trials()

        # Split the data into train, validation, and test sets based on the predefined validation indexes
        to_train = self.data[self.data['date_block_num'].isin(
            [row['train'] for row in self.validation_dict['validation_indexes']][0])]

        train = to_train[to_train['date_block_num'] != to_train['date_block_num'].max()]
        val = to_train[to_train['date_block_num'] == to_train['date_block_num'].max()]

        test = self.data[
            self.data['date_block_num'].isin([row['val'] for row in self.validation_dict['validation_indexes']][0])]

        # Create CatBoost Pool objects for training and validation data
        train_data = cb.Pool(data=train[in_features],
                             label=train[target],
                             cat_features=cat_features)

        val_data = cb.Pool(data=val[in_features],
                           label=val[target],
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
            model.fit(train_data, eval_set=val_data, use_best_model=True, early_stopping_rounds=15)
            preds = model.predict(test[in_features])

            # Calculate the root mean squared error (RMSE) between the predictions and the actual target values
            rmse = np.sqrt(mean_squared_error(test[target], preds))

            return {'loss': rmse, 'status': STATUS_OK}

        # Use hyperopt library to perform hyperparameter optimization
        self.params = fmin(fn=_hyperparameter_tuning, space=param_space, algo=tpe.suggest, max_evals=100, trials=trials,
                           verbose=True)

        return self.params

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

        print("INFO: Model was successfully saved")

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
