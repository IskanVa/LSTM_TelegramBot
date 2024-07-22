import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from glob import glob
import os
import gzip

# Пути к директориям и файлам
filtered_logs_file = "/home/host/tpotmodel/new_cowrie.json"
model_path = '/home/host/tpotmodel/best_autoencoder_model_tpotBi.keras'
ANOMALIES_FILE = "/home/host/surimodel/telegrambot/detected_anomalies_tpot.json"

async def copy_logs(context):
    log_files = glob('/nfs/tpot/cowrie/log/cowrie.json.2024-*')
    log_files.append('/nfs/tpot/cowrie/log/cowrie.json')
    print("copy logs tpot")
    all_logs = []

    for file_path in log_files:
        with open(file_path, "rb") as file:
            try:
                decompressed_data = gzip.decompress(file.read())
                lines = decompressed_data.decode("utf-8", errors="ignore").splitlines()
            except gzip.BadGzipFile:
                file.seek(0)
                lines = file.read().decode("utf-8", errors="ignore").splitlines()

            for line in lines:
                try:
                    log_entry = json.loads(line)
                    all_logs.append(log_entry)
                except json.JSONDecodeError:
                    print(f"Error decoding line: {line}")
                    continue

    filtered_logs_sorted = sorted(all_logs, key=lambda x: datetime.strptime(x['timestamp'], '%Y-%m-%dT%H:%M:%S.%f%z'))

    with open(filtered_logs_file, "w") as file:
        for entry in filtered_logs_sorted:
            json.dump(entry, file)
            file.write('\n')

def copy_and_return_logs(log_files, destination_file):
    all_logs = []

    for file_path in log_files:
        with open(file_path, "rb") as file:
            try:
                decompressed_data = gzip.decompress(file.read())
                lines = decompressed_data.decode("utf-8", errors="ignore").splitlines()
            except gzip.BadGzipFile:
                file.seek(0)
                lines = file.read().decode("utf-8", errors="ignore").splitlines()

            for line in lines:
                try:
                    log_entry = json.loads(line)
                    all_logs.append(log_entry)
                except json.JSONDecodeError:
                    print(f"Error decoding line: {line}")
                    continue

    filtered_logs_sorted = sorted(all_logs, key=lambda x: datetime.strptime(x['timestamp'], '%Y-%m-%dT%H:%M:%S.%f%z'))

    with open(destination_file, "w") as file:
        for entry in filtered_logs_sorted:
            json.dump(entry, file)
            file.write('\n')

    return destination_file

def prepare_data(filtered_logs_file):
    data = []

    with open(filtered_logs_file, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.resample('1min').size().to_frame('log_count')
    df = df[df['log_count'] > 1]

    timestamps = df.index
    original_log_counts = df['log_count'].values

    df.reset_index(drop=True, inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    df['log_count'] = scaler.fit_transform(df['log_count'].values.reshape(-1, 1))

    return df, timestamps, original_log_counts

def create_dataset(dataset, look_back):
    dataX = []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
    return np.array(dataX)

def detect_anomalies(df, timestamps, original_log_counts):
    look_back = 10
    dataset = df.values
    dataX = create_dataset(dataset, look_back)
    dataX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1], 1))

    model = load_model(model_path)
    predicted_dataX = model.predict(dataX)
    mse = np.mean(np.power(dataX - predicted_dataX, 2), axis=1)
    threshold = np.percentile(mse, 95)
    anomalies = mse > threshold
    anomalies = anomalies.flatten()
    anomaly_indices = np.arange(look_back, look_back + len(anomalies))

    filtered_anomaly_indices = anomaly_indices[anomalies]
    filtered_anomaly_indices = filtered_anomaly_indices[original_log_counts[filtered_anomaly_indices] > 50]

    filtered_anomalies = []
    for i in range(0, len(filtered_anomaly_indices), 20):
        group = filtered_anomaly_indices[i:i + 10]
        if len(group) > 0:
            max_idx = group[np.argmax(original_log_counts[group])]
            filtered_anomalies.append(max_idx)

    return filtered_anomalies

def plot_anomalies(timestamps, original_log_counts, filtered_anomalies, interval):
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, original_log_counts, label='Число логов')
    plt.scatter(timestamps[filtered_anomalies], original_log_counts[filtered_anomalies], color='red', label='Аномалии', zorder=2)
    plt.title(f'Логи с аномалиями T-Pot(Минтранс) ({interval})')
    plt.xlabel('Время')
    plt.ylabel('Кол-во логов')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plot_path = f'/home/host/tpotmodel/img/anomalies_log_count_plot_{interval}.png'
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def load_detected_anomalies():
    if os.path.exists(ANOMALIES_FILE):
        with open(ANOMALIES_FILE, 'r') as f:
            return json.load(f)
    return []

def save_detected_anomalies(anomalies):
    with open(ANOMALIES_FILE, 'w') as f:
        json.dump(anomalies, f)
    
def process_tpot_logs(interval='7d'):
    now = datetime.now()
    if interval == '1d':
        start_time = now - timedelta(days=1)
    elif interval == '3d':
        start_time = now - timedelta(days=3)
    elif interval == '7d':
        start_time = now - timedelta(days=7)
    elif interval == '21d':
        start_time = now - timedelta(days=21)
    elif interval == 'all_time':
        start_time = datetime.strptime('2024-01-01', '%Y-%m-%d')
    else:
        start_time = now - timedelta(days=1)

    log_files = []
    for single_date in (start_time + timedelta(n) for n in range((now - start_time).days + 1)):
        log_files.extend(glob(f'/nfs/tpot/cowrie/log/cowrie.json.{single_date.strftime("%Y-%m-%d")}'))

    if not log_files:
        # print("No log files found for the given interval.")
        return [], [], []
    # print("log_files", log_files)

    filtered_logs_file = copy_and_return_logs(log_files, f"/home/host/tpotmodel/filtered_cowrie_{interval}.json")
    # print("filtered_logs_file", filtered_logs_file)

    df, timestamps, original_log_counts = prepare_data(filtered_logs_file)  # Подготовка данных
    peak_anomalies = detect_anomalies(df, timestamps, original_log_counts)  # Обнаружение аномалий
    print("peak_anomalies_tpot", peak_anomalies)

    return peak_anomalies, timestamps, original_log_counts
