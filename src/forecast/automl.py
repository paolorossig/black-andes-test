from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

import matplotlib.pyplot as plt


class AutoMLModel:
    def __init__(
        self,
        architecture,
        yearly_seasonality,
        weekly_seasonality,
        holidays,
    ):
        self.architecture = architecture

        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            seasonality_mode="additive",
            holidays=holidays,
        )

    def fit(self, data):
        self.model.fit(data)
        self.data = data

    def predict(self, periods, freq):
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        return self.model.predict(future)

    def plot_components(self):
        train_forecast = self.model.predict(self.data)
        return plot_components_plotly(self.model, train_forecast)

    def plot_model_forecast(self, forecast):
        return plot_plotly(self.model, forecast)


def plot_prophet_predictions(test_set, forecast):
    preds = forecast.tail(len(test_set))
    test_set = test_set.merge(preds, on=["ds"], how="left").set_index("ds")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_set.y, "o", color="black", label="y_true")
    ax.plot(test_set.yhat, "-", label="preds")
    ax.fill_between(
        test_set.index,
        test_set["yhat_lower"],
        test_set["yhat_upper"],
        color="k",
        alpha=0.1,
    )
    ax.legend()
    plt.show()
