import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.random import set_seed
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

from src.forecast.utils import print_metrics

set_seed(42)


def scale_data(data: [np.ndarray], scaler_type: str = "standard") -> tuple:
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("scaler must be 'minmax' or 'standar'")

    data_scaled = []
    scaler = scaler.fit(data[0])
    for i in data:
        data_scaled.append(scaler.transform(i).flatten())

    return data_scaled, scaler


def time_series_to_X_y(series: [np.ndarray], window_size: int = 5) -> list:
    data = []
    for serie in series:
        X, y = [], []

        for i in range(len(serie) - window_size):
            row = [[a] for a in serie[i : i + window_size]]
            X.append(row)
            label = serie[i + window_size]
            y.append(label)

        data.append((np.array(X), np.array(y)))

    return data


def plot_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray
) -> None:
    print_metrics(y_true, y_pred, y_train)

    plt.plot(y_true, "o", color="black", label="y_true")
    plt.plot(y_pred, "-", label="preds")
    plt.legend()
    plt.show()


class DeepLearningModel:
    def __init__(self, architecture, filters, window_size):
        self.architecture = architecture
        self.filters = filters
        self.window_size = window_size

        model = Sequential()
        model.add(InputLayer((self.window_size, 1)))
        model.add(Conv1D(self.filters, kernel_size=2))
        model.add(Flatten())
        model.add(Dense(20, "relu"))
        model.add(Dense(1, "linear"))

        self.model = model

    def fit(self, train_set, val_set, epochs):
        X_train, y_train = train_set
        X_val, y_val = val_set

        self.model.compile(
            loss=MeanSquaredError(),
            optimizer=Adam(learning_rate=0.0001),
            metrics=[RootMeanSquaredError()],
        )

        history = self.model.fit(
            X_train, y_train, validation_data=(X_val, y_val), epochs=epochs
        )
        self.history = history.history

    def predict(self, X):
        return self.model.predict(X)

    def plot_loss_function(self):
        plt.plot(self.history["loss"])
        plt.plot(self.history["val_loss"])
        plt.title("Model Loss")
        plt.xlabel("Epochs")
        plt.legend(["train", "val"], loc="upper right")
        plt.show()
