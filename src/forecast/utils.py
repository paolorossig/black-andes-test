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


def df_to_X_y(data: np.ndarray, window_size: int = 5) -> tuple:
    """
    Convert a dataframe to X and y for training.
    """
    X = []
    y = []

    for i in range(len(data) - window_size):
        row = [[a] for a in data[i : i + window_size]]
        X.append(row)
        label = data[i + window_size]
        y.append(label)

    return np.array(X), np.array(y)


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


def plot_predictions(model, X, y, start=0, end=300):
    """
    Plot the predictions of the model.
    """
    predictions = model.predict(X).flatten()
    df = pd.DataFrame(data={"Predictions": predictions, "Actuals": y})

    plt.plot(df["Predictions"][start:end])
    plt.plot(df["Actuals"][start:end])

    return df, mean_squared_error(y, predictions)


### Functions for calculating metrics.
def rmse(y_true, y_pred):
    """
    Calculate the root mean squared error.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))