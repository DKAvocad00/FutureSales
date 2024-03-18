# FutureSales

## Overview

The FutureSales project is an initiative aimed at participating in the Kaggle competition titled "Predict Future Sales."
This competition challenges participants to forecast future sales for a software company, using historical sales
data provided. The goal is to predict total sales for every product and store in the following month.

## Project Structure

The project directory is structured as follows:

- **data/**
    - **raw/**: Contains the raw CSV files with the initial dataset.
    - **processed/**: Directory for storing processed datasets.
- **notebooks/**: Holds Jupyter notebooks for data analysis, preprocessing, and visualization.
    - **baseline.ipynb**: Jupyter notebook where data overview, preprocessing are performed using ETL, DQC processes.
    - **EDA.ipynb**: Jupyter notebook where data visualization are performed using EDA processes.
    - **modeling.ipynb**: Jupyter notebook for visualization feature importance and visualization with SHAP methods
- **src/**
    - **analytics/**: Contains py scripts to analyse model like feature importance, expandability, error_analysis
    - **data/**: Contains py scripts with ETL and DQC classes to proccess and chech row data
    - **modeling/**: Contains py scripts with classes and methods to create final data and training model, like feature
      modeling, validation schema and training schema
    - **models/**: Contains py scripts that can be used for training and prediction
        - **predict_model.py**: Py script for predict future sales by training model
        - **train_model.py**: Py script for preprocessing data and training model on this process data
    - **visualize/**: Contains py scripts to analyses and visualize statistic on row data

## Installation

To run the project locally, follow these steps:

1. Clone the repository: `git clone <repository-url>`
2. Install the required Python dependencies: `pip install -r requirements.txt`
3. Ensure that the raw CSV files are placed in the `data/raw/` directory.

## Usage

1. Run the `train_model.py` file to process row/intermediate data and training model

   #### Example usage

   To process data and train model with default settings, you can run the following command:

          python src/models/train_model.py

   Alternatively, you can use the additional flags to process and train model by running:

          python src/models/train_model.py --make_data_big --final_data_name="final_subdata" --final_model_name="catboost_sub"

   Use `python src/models/train_model.py --help` to learn more about flags.


2. Run the `predict_model.py` file to make prediction on test data and load kaggle data to .csv

   #### Example usage

   To use training model for prediction, you can run the following command:

        python src/models/predict_model.py

   Alternatively, you can use the additional flags to run your script:

        python src/models/predict_model.py --kaggle_file_name="kaggle_predictions_data"

   Use `python src/models/predict_model.py --help` to learn more about flags.

