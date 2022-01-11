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


def get_covid_data(
    country: str, start: str, end: str, freq: str = "MS"
) -> pd.DataFrame:
    """
    Data extracted from COVID-19 Data Repository by the Center for
    Systems Science and Engineering (CSSE) at Johns Hopkins University
    See https://github.com/CSSEGISandData/COVID-19

    Parameters
    ----------
    country: str
        The country to get the data for. (e.g. "Peru", "Colombia", "Ecuador")
    start: str
        The start date of the data. (e.g. "2020-01-01")
    end: str
        The end date of the data. (e.g. "2020-12-01")
    freq: str
        The frequency of the data. (e.g. "D", "W", "M", "MS")
    """
    df = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    )
    df = df[df["Country/Region"] == country]
    df = pd.melt(
        df,
        value_vars=df.columns[4:],
        var_name="date",
        value_name="new_cases",
    )
    df.date = pd.to_datetime(df.date, format="%m/%d/%y")
    df.set_index("date", inplace=True)

    if freq != "D":
        df = df.resample(freq).sum()

    start = pd.to_datetime(start)
    if start < df.index.min():
        periods = np.round(
            (df.index.min() - start) / np.timedelta64(1, freq[:1])
        ).astype(int)
        missing_dates = pd.DataFrame(
            {
                "date": pd.date_range(start, periods=periods, freq=freq),
                "new_cases": np.zeros(periods),
            }
        ).set_index("date")
        df = pd.concat([missing_dates, df])

    df = df[start:end]

    df_diff = df - df.shift(1)
    df_diff.new_cases = df_diff.new_cases.fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_diff)

    return df_diff


### Functions for plotting the data.
def tsplot(y: pd.Series, lags: int = None, figsize: tuple = (12, 6)) -> None:
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


def plot_predictions(y_true, y_pred):
    """
    Plot time series vs predictions.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_true, "o", color="black", label="y_true")
    ax.plot(y_pred, "-", label="preds")
    ax.legend()
    plt.show()


### Functions for calculating metrics.
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the root mean squared error.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the mean absolute percentage error.
    """
    return np.mean(np.abs((y_true - y_pred) / y_true))


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """
    Calculate the mean absolute scaled error.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    mae_insample_naive = np.mean(np.abs(y_train[1:] - y_train[:-1]))
    return mae / mae_insample_naive


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> None:
    """
    Print the metrics.
    """
    print("RMSE: {:.2f}".format(rmse(y_true, y_pred)))
    print("MAPE: {:.2f}%".format(mape(y_true, y_pred) * 100))
    if y_train is not None:
        print("MASE: {:.2f}%".format(mase(y_true, y_pred, y_train) * 100))
