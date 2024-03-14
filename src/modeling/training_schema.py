import pandas as pd
import catboost as cb


class TrainingModel:
    """

    """

    def __init__(self,
                 train: pd.DataFrame | None = None,
                 val: pd.DataFrame | None = None,
                 test: pd.DataFrame | None = None,
                 data: pd.DataFrame | None = None,
                 validation_indexes: list | None = None,
                 test_indexes: list | None = None) -> None:
        """

        """
        self.train: pd.DataFrame | None = train
        self.val: pd.DataFrame | None = val
        self.test: pd.DataFrame | None = test
        self.data: pd.DataFrame | None = data
        self.validation_indexes: list | None = validation_indexes
        self.test_indexes: list | None = test_indexes

    def train_model(self,
                    in_features: list[str],
                    target: list[str],
                    cat_features: list[str],
                    train: pd.DataFrame | None = None,
                    val: pd.DataFrame | None = None,
                    test: pd.DataFrame | None = None) -> list[float]:

        """

        """

        train = self.train if train is None else train

        if train is None:
            raise TypeError("Invalid train type. Must be pd.DataFrame but got {}".format(type(train)))

        val = self.val if val is None else val

        if val is None:
            raise TypeError("Invalid val type. Must be pd.DataFrame but got {}".format(type(val)))

        test = self.test if test is None else test

        if test is None:
            raise TypeError("Invalid test type. Must be pd.DataFrame but got {}".format(type(test)))

        train_data = cb.Pool(train[in_features],
                             train[target],
                             cat_features=cat_features)
        val_data = cb.Pool(val[in_features],
                           val[target],
                           cat_features=cat_features)

        model = cb.CatBoostRegressor(cat_features=cat_features, task_type="GPU", random_seed=42)
        model.fit(train_data, eval_set=val_data, use_best_model=True, verbose=True, early_stopping_rounds=70)
        return model.predict(test[in_features])

    def train_cv_model(self,
                       in_features: list[str],
                       target: list[str],
                       cat_features: list[str],
                       data: pd.DataFrame | None = None,
                       validation_indexes: list | None = None,
                       test_indexes: list | None = None) -> list[float]:
        """

        """
        data = self.data if data is None else data

        if data is None:
            raise TypeError("Invalid data type. Must be pd.DataFrame but got {}".format(type(data)))

        validation_indexes = self.validation_indexes if validation_indexes is None else validation_indexes

        if validation_indexes is None:
            raise TypeError(
                "Invalid validation indexes type. Must be pd.DataFrame but got {}".format(type(validation_indexes)))

        test_indexes = self.test_indexes if test_indexes is None else test_indexes

        if isinstance(test_indexes, None):
            raise TypeError("Invalid test indexes type. Must be pd.DataFrame but got {}".format(type(test_indexes)))

        test = data[data['date_block_num'].isin(test_indexes)]

        model = cb.CatBoostRegressor(task_type="GPU", random_seed=42)

        for idx, row in validation_indexes:
            print("Iteration {} of {}".format(idx, len(validation_indexes)))
            train_df = data[data['date_block_num'].isin(row['train'])]
            val_df = data[data['date_block_num'].isin(row['val'])]

            train_data = cb.Pool(train_df[in_features],
                                 train_df[target],
                                 cat_features=cat_features)
            val_data = cb.Pool(val_df[in_features],
                               val_df[target],
                               cat_features=cat_features)

            model.fit(train_data, eval_set=val_data, use_best_model=True, verbose=True, early_stopping_rounds=70)

        return model.predict(test[in_features])
