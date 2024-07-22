import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, TimeDistributed, RepeatVector
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import multiprocessing
import tensorflow as tf

# Путь к файлу с логами
logs_file = "/home/host/surimodel/filtered_surilogs.json"

# Чтение логов из файла в DataFrame
df = pd.read_json(logs_file, lines=True)

# Преобразование временной метки и установка её в качестве индекса
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Агрегирование логов по 1 минуте
df = df.resample('1min').size().to_frame('log_count')

# df = df[(df['log_count'] > 1) & (df['log_count'] < 50)]  
df = df[df['log_count'] > 1]  

# Пересоздание индексов
df.reset_index(drop=True, inplace=True)

# Нормализация данных
scaler = MinMaxScaler(feature_range=(0, 1))
df['log_count'] = scaler.fit_transform(df['log_count'].values.reshape(-1, 1))
df.to_csv('/home/host/surimodel/clearlogsMinMax.csv')

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


# Построение и обучение LSTM автоэнкодера с Sequential API
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(look_back, 1)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(RepeatVector(look_back))
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(TimeDistributed(Dense(1)))

model.compile(loss='mae', optimizer='adam')
model.summary()


# Callback для сохранения лучшей модели и ранней остановки
checkpoint_callback = ModelCheckpoint(filepath='/home/host/surimodel/autoencoder_model_suricata.keras', 
                                      monitor='val_loss', save_best_only=True, mode='min', verbose=1)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Обучение модели
history = model.fit(dataX, dataX, epochs=100, batch_size=32, verbose=2,
                    validation_split=0.2, callbacks=[checkpoint_callback, early_stopping_callback])

# Загрузка лучшей модели
model.load_weights('/home/host/surimodel/autoencoder_model_suricata.keras')

# Оценка качества модели
# Построение графика потерь
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('/home/host/surimodel/img/loss_plot2.png')
plt.show()

# Прогнозирование на обучающих данных
predicted_dataX = model.predict(dataX)

# Вычисление среднеквадратичной ошибки
mse = np.mean(np.power(dataX - predicted_dataX, 2), axis=1)

threshold = np.percentile(mse, 95)

# Построение графика ошибок
plt.figure(figsize=(12, 6))
plt.plot(mse, label='Reconstruction Error')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Error')
plt.xlabel('Samples')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.savefig('/home/host/surimodel/img/reconstruction_error_plot2.png')
plt.show()