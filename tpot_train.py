import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, TimeDistributed, RepeatVector, Bidirectional, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import os
import multiprocessing
import tensorflow as tf

# Путь к логам
log_file = "/home/host/tpotmodel/filtered_cowrie_all_time.json"

# Объединение всех данных в один DataFrame
df = pd.read_json(log_file, lines=True)
df.set_index('timestamp', inplace=True)
df = df.resample('1min').size().to_frame('log_count')

df = df[df['log_count'] > 1]
df.reset_index(drop=True, inplace=True)

# Используем RobustScaler вместо MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df['log_count'] = scaler.fit_transform(df['log_count'].values.reshape(-1, 1))

# Создание временного окна
def create_dataset(dataset, look_back):
    dataX = []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
    return np.array(dataX)

look_back = 7
dataset = df.values
dataX = create_dataset(dataset, look_back)
dataX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1], 1))

model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01)), input_shape=(look_back, 1)),
    Dropout(0.4),
    BatchNormalization(),
    Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=l2(0.01))),
    Dropout(0.4),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dense(look_back),
    Reshape((look_back, 1))
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='mse', optimizer=optimizer)
model.summary()

# Callback для сохранения лучшей модели и ранней остановки
checkpoint_callback = ModelCheckpoint(filepath='/home/host/tpotmodel/best_autoencoder_model_tpotBi7.keras',
                                      monitor='val_loss', save_best_only=True, mode='min', verbose=1)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# Обучение модели
history = model.fit(dataX, dataX, epochs=1000, batch_size=32, verbose=2,  # Changed batch size
                    validation_split=0.2, callbacks=[checkpoint_callback, early_stopping_callback])

# Загрузка лучшей модели
model.load_weights('/home/host/tpotmodel/best_autoencoder_model_tpotBi7.keras')

# Оценка качества модели
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('/home/host/tpotmodel/img/loss_plotBi7.png')

# Прогнозирование на обучающих данных
predicted_dataX = model.predict(dataX)

# Вычисление среднеквадратичной ошибки
mse = np.mean(np.power(dataX - predicted_dataX, 2), axis=1)

threshold = np.percentile(mse, 95)  # Установите порог как 95-й процентиль ошибок

# Построение графика ошибок
plt.figure(figsize=(12, 6))
plt.plot(mse, label='Reconstruction Error')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Error')
plt.xlabel('Samples')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.savefig('/home/host/tpotmodel/img/reconstruction_error_plotBi7.png')