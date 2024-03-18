import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt
from src.visualization.statistics_utils import adfuller_stationarity, kpss_stationarity


def barplots(data: pd.DataFrame, group_column: list[str], agg_column: str, x: str, y: str,
             agg_func: list[str] | None = None) -> None:
    """
    **Creates bar plots based on grouping and aggregation of columns.**

    This method creates bar plots based on the specified group column and aggregation column.
    Optionally, multiple aggregation functions can be applied to the aggregation column.

    :param data: The data used for grouping and visualization
    :param group_column: The column used for grouping.
    :param agg_column: The column used for aggregation.
    :param agg_func: list of aggregation functions to apply. Default is ['sum'].
                     Possible values include 'sum', 'mean', 'median', 'min', 'max', etc.
    :param x: The column to be used as the x-axis in the bar plot.
    :param y: The column to be used as the y-axis in the bar plot.
    :return: None
    """

    # If agg_func is None, set it to default ['sum']
    if agg_func is None:
        agg_func = ['sum']

    # Create subplots based on the number of aggregation functions
    _, ax = plt.subplots(len(agg_func), 1, figsize=(24, 9))

    # If ax is not an array, convert it to a list
    if not isinstance(ax, np.ndarray):
        ax = [ax]

    # Iterate over aggregation functions and create bar plots
    for i, func in enumerate(agg_func):
        sns.barplot(data=data.groupby(group_column, as_index=False)[agg_column].agg(func), x=x,
                    y=y, palette='Set2', ax=ax[i])
        ax[i].set_xlabel(group_column)
        ax[i].set_ylabel(func + ' ' + agg_column)

    plt.tight_layout()


def lineplots(data: pd.DataFrame, group_column: list[str], agg_column: str, x: str, y: str,
              agg_func: list[str] | None = None) -> None:
    """
    **Creates line plots based on grouping and aggregation of columns.**

    This method creates line plots based on the specified group column and aggregation column.
    Optionally, multiple aggregation functions can be applied to the aggregation column.

    :param data: The data used for grouping and visualization
    :param group_column: The column used for grouping.
    :param agg_column: The column used for aggregation.
    :param agg_func: list of aggregation functions to apply. Default is ['sum'].
                     Possible values include 'sum', 'mean', 'median', 'min', 'max', etc.
    :param x: The column to be used as the x-axis in the line plot.
    :param y: The column to be used as the y-axis in the line plot.
    :return: None
    """

    # If agg_func is None, set it to default ['sum']
    if agg_func is None:
        agg_func = ['sum']

    # Create subplots based on the number of aggregation functions
    _, ax = plt.subplots(len(agg_func), 1, figsize=(24, 9))

    # If ax is not an array, convert it to a list
    if not isinstance(ax, np.ndarray):
        ax = [ax]

    # Iterate over aggregation functions and create line plots
    for i, func in enumerate(agg_func):
        sns.lineplot(data=data.groupby(group_column, as_index=False)[agg_column].agg(func), x=x,
                     y=y, palette='Set2', ax=ax[i])
        ax[i].set_xlabel(group_column)
        ax[i].set_ylabel(func + ' ' + agg_column)

    plt.tight_layout()


def tsplot(y: pd.Series, lags: int | None = None) -> None:
    """
    **Generates a time series plot along with ACF and PACF plots, and performs stationarity tests.**

    This method plots the original time series data and the autocorrelation function (ACF) and
    partial autocorrelation function (PACF) plots. It also performs stationarity tests using the
    Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests.

    :param y: The time series data to plot and test for stationarity.
    :param lags: The number of lags to include in the ACF and PACF plots. Default is None.
    :return: None
    """

    # Perform ADF and KPSS stationarity tests
    adfuller_stationarity(y)
    kpss_stationarity(y)

    # Create subplots for time series, ACF, and PACF plots
    with plt.style.context('bmh'):
        plt.figure(figsize=(14, 8))
        layout = (4, 1)
        ts_ax = plt.subplot2grid(layout, (0, 0), rowspan=2)
        acf_ax = plt.subplot2grid(layout, (2, 0))
        pacf_ax = plt.subplot2grid(layout, (3, 0))

        # Plot original time series data
        y.plot(ax=ts_ax, color='blue', label='Or')
        ts_ax.set_title('Original')

        # Plot ACF
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)

        # Plot PACF
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)

        plt.tight_layout()


def series_decompose(series: pd.Series, period: int = 30) -> None:
    """
    **Decomposes a time series into its trend, seasonal, and residual components using seasonal decomposition.**

    This function decomposes a time series into its trend, seasonal, and residual components using the
    seasonal decomposition method. It plots the original series along with its trend, seasonal, and
    residual components for visualization.

    :param series: The time series data to decompose.
    :param period: The period of seasonality in the time series. Default is 30.
    :return: None
    """

    # Perform seasonal decomposition
    result = seasonal_decompose(series, model='multiplicative', period=period)

    # Create subplots for original series, trend, seasonal, and residuals
    plt.figure(figsize=(12, 8))

    plt.subplot(4, 1, 1)
    plt.plot(series, label='Original')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(result.trend, label='Trend')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(result.seasonal, label='Seasonal')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(result.resid, label='Residuals')
    plt.legend()

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()
