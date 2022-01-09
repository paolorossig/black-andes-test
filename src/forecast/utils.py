import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


### Functions for getting and handling the data.
def get_data(path: str) -> pd.DataFrame:
    """
    Preprocess the data to make it suitable for prophet.
    """
    df = pd.read_csv(path, sep=";")
    df = df.iloc[:, :2]
    df.columns = ["ds", "y"]
    df.ds = pd.to_datetime(df.ds, format="%d/%m/%Y")

    return df


def time_series_split(data: pd.DataFrame, n_val: int, n_test: int) -> tuple:
    """
    Split the data into training, validation and testing sets.
    """
    df_to_np = data.set_index("ds").to_numpy()
    n_aux = n_val + n_test
    train, val, test = df_to_np[:-n_aux], df_to_np[-n_aux:-n_test], df_to_np[-n_test:]

    return train, val, test


### Functions for plotting the data.
def tsplot(y: pd.Series, lags: int = None, figsize: tuple = (12, 7)) -> None:
    """
    Plot time series, its ACF and PACF, calculate Dickey–Fuller test

    y - timeseries
    lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    p_value = adfuller(y)[1]
    ts_ax.set_title(
        "Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}".format(p_value)
    )
    plot_acf(y, lags=lags, ax=acf_ax)
    plot_pacf(y, lags=lags, ax=pacf_ax)
    plt.tight_layout()
    plt.show()


### Functions for calculating metrics.
def rmse(y_true, y_pred):
    """
    Calculate the root mean squared error.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    """
    Calculate the mean absolute percentage error.
    """
    return np.mean(np.abs((y_true - y_pred) / y_true))


def mase(y_true, y_pred):
    """
    Calculate the mean absolute scaled error.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    mae_insample_naive = np.mean(np.abs(y_true[1:] - y_true[:-1]))
    return mae / mae_insample_naive
