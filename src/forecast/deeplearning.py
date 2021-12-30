import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

from utils import *


df = get_data("../../data/producto_C.csv")
df_np = df.set_index("ds").to_numpy()

N_TEST = 30
y_train, y_test = df_np[:-N_TEST], df_np[-N_TEST:]

scaler = StandardScaler()
# scaler = MinMaxScaler()
scaler = scaler.fit(y_train)
y_train = scaler.transform(y_train).flatten()
y_test = scaler.transform(y_test).flatten()

window_size = 7
X, y = df_to_X_y(y_train, window_size)
# generator = TimeseriesGenerator(y_train, y_train, length=window_size, batch_size=1)
X.shape, y.shape

X_train, y_train = X[:-15], y[:-15]
X_val, y_val = X[-15:], y[-15:]
X_test, y_test = df_to_X_y(y_test, window_size)
X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape


# model = Sequential()
# model.add(InputLayer((window_size, 1)))
# model.add(LSTM(200))
# model.add(Dense(20, "relu"))
# model.add(Dense(1, "linear"))

# model.summary()

model = Sequential()
model.add(InputLayer((window_size, 1)))
model.add(Conv1D(200, kernel_size=2))
model.add(Flatten())
model.add(Dense(20, "relu"))
model.add(Dense(1, "linear"))

model.summary()

# model = Sequential()
# model.add(InputLayer((window_size, 1)))
# model.add(GRU(64))
# model.add(Dense(8, "relu"))
# model.add(Dense(1, "linear"))

# model.summary()

# model = Sequential()
# model.add(InputLayer((window_size, 1)))
# model.add(LSTM(32, return_sequences=True))
# model.add(LSTM(64))
# model.add(Dense(8, "relu"))
# model.add(Dense(1, "linear"))

# model.summary()

model.compile(
    loss=MeanSquaredError(),
    optimizer=Adam(learning_rate=0.0001),
    metrics=[RootMeanSquaredError()],
)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=False)
# model.fit_generator(generator, validation_data=(X_val, y_val), epochs=10, verbose=False)

preds = model.predict(X_test)

preds = scaler.inverse_transform(preds).flatten()
y_test = scaler.inverse_transform(y_test).flatten()

rmse(y_test, preds)
plt.plot(y_test, label="y_test")
plt.plot(preds, label="preds")
plt.show()
# 359.44 (64 LSTM - 8 relu)
# 331.98 (200 LSTM - 20 relu)
# 364.02 (64 CNN - 8 relu)
# 277.05 (200 CNN - 20 relu)
# 339.97 (64 GRU - 8 relu)
# 328.05 (200 GRU - 20 relu)

plot_predictions(model, X, y, end=600)
plot_predictions(model, X_test, y_test)
