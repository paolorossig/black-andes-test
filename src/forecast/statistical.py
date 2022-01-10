import pmdarima as pm

from statsmodels.api import add_constant
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.statespace.sarimax import SARIMAX

# from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


class StatisticalModel:
    def __init__(self, architecture):
        self.architecture = architecture

    def auto_arima(self, data):
        if self.architecture != "auto_arima":
            raise ValueError("This is not an auto_arima model")

        self.model = pm.auto_arima(
            data,
            max_p=12,
            max_q=12,
            max_P=12,
            max_Q=12,
            max_d=3,
            seasonal=True,
            seasonal_test="ch",
            trace=True,
            test="adf",
        )

    def fit_auto_arima(self, data):
        if self.architecture != "auto_arima":
            raise ValueError("This is not an auto_arima model")

        return self.model.fit(data)

    def predict(self, n_periods):
        return self.model.predict(n_periods=n_periods)

    def fit_SARIMAX(self, data, exog, order, seasonal_order=(0, 0, 0)):
        if self.architecture != "SARIMAX":
            raise ValueError("This is not a SARIMAX model")

        model = SARIMAX(
            data, exog=exog, order=order, seasonal_order=seasonal_order, freq="MS"
        )
        self.model = model
        return model.fit(disp=False)


def plot_decomposition(data, period):
    decomposition = seasonal_decompose(data, model="additive", period=period)
    decomposition.plot()


def print_adf_test(data):
    adf = adfuller(data, autolag="AIC", regression="c")
    print("--- ADF test ---")
    print("ADF Statistic: {:.3f} \np-value: {:.3f}".format(adf[0], adf[1]))


def prepare_exog(data):
    return add_constant(data)
