import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def adfuller_stationarity(timeseries: pd.Series) -> None:
    """
    **Performs the Augmented Dickey-Fuller (ADF) test for stationarity on a given time series.**

    The Augmented Dickey-Fuller (ADF) test is used to determine whether a time series is stationary
    or non-stationary based on the presence of unit roots. A stationary time series has a constant mean,
    variance, and autocovariance over time.

    :param timeseries: The time series data to perform the ADF test on.
    :return: None
    """

    # Perform ADF test
    dftest = adfuller(timeseries, autolag='AIC')

    # Create a Series to store ADF test results
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

    # Add critical values to the Series
    for [key, value] in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value

    # Print ADF test results
    print('Results of Dickey-Fuller Test:\n{}'.format(dfoutput))


def kpss_stationarity(timeseries: pd.Series) -> None:
    """
    **Performs the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for stationarity on a given time series.**

    The KPSS test is used to determine whether a time series is stationary or non-stationary by testing
    for the presence of trends in the data.

    :param timeseries: The time series data to perform the KPSS test on.
    :return: None
    """

    # Perform KPSS test
    kpsstest = kpss(timeseries, regression='c', nlags="auto")

    # Create a Series to store KPSS test results
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', '#Lags Used'])

    # Add critical values to the Series
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value

    # Print KPSS test results
    print('Results of KPSS Test:\n{}'.format(kpss_output))
