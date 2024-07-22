import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from collections import Counter
from ipaddress import ip_network, ip_address
import glob
import os
import shutil
import gzip

# Путь к директории с логами
source_dir = '/nfs/home/'
destination_dir = '/home/host/surimodel/fromsuri/'

# Файл для хранения обнаруженных аномалий
ANOMALIES_FILE = '/home/host/surimodel/telegrambot/detected_anomalies_suricata.json'

async def copy_logs(context):
    source_files = glob.glob(os.path.join(source_dir, '*.json.gz'))
    destination_files = set(os.listdir(destination_dir))
    print("copy logs suricata")

    for source_file in source_files:
        file_name = os.path.basename(source_file)

        if file_name not in destination_files:
            shutil.copy(source_file, destination_dir)
            gzip_file = os.path.join(destination_dir, file_name)
            with gzip.open(gzip_file, 'rb') as f_in:
                with open(gzip_file[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(gzip_file)

def read_ip_list(file_path):
    networks = []
    ips = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if '/' in line:
                try:
                    networks.append(ip_network(line))
                except ValueError:
                    print(f"Invalid network: {line}")
            else:
                try:
                    ips.append(ip_address(line))
                except ValueError:
                    print(f"Invalid IP address: {line}")
    return networks, ips

excluded_networks, excluded_ips = read_ip_list('whitelist.txt')

def filter_logs(log_files):
    filtered_logs = []
    for file_path in log_files:
        with open(file_path, 'r') as file:
            for line in file:
                log_entry = json.loads(line)
                src_ip = ip_address(log_entry.get('src_ip'))

                if any(src_ip in network for network in excluded_networks) or src_ip in excluded_ips:
                    continue

                filtered_logs.append(log_entry)

    filtered_logs_sorted = sorted(filtered_logs, key=lambda x: datetime.strptime(x['timestamp'], '%Y-%m-%dT%H:%M:%S.%f%z'))
    return filtered_logs_sorted

def save_filtered_logs(filtered_logs):
    filtered_logs_file = '/home/host/surimodel/filtered_surilogs.json'
    with open(filtered_logs_file, 'w') as outfile:
        for entry in filtered_logs:
            json.dump(entry, outfile)
            outfile.write('\n')
    return filtered_logs_file

def prepare_data(filtered_logs_file):
    data = []

    with open(filtered_logs_file, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = df['timestamp'] + pd.Timedelta(hours=3)
    df.set_index('timestamp', inplace=True)
    df = df.resample('1min').size().to_frame('log_count')
    df = df[df['log_count'] > 1]

    timestamps = df.index
    original_log_counts = df['log_count'].values

    df.reset_index(drop=True, inplace=True)
    forscatter = df['log_count'].values

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
    look_back = 7
    dataset = df.values
    dataX = create_dataset(dataset, look_back)
    dataX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1], 1))

    model = load_model('/home/host/surimodel/autoencoder_model_suricata.keras')
    predicted_dataX = model.predict(dataX)
    mse = np.mean(np.power(dataX - predicted_dataX, 2), axis=1)
    threshold = np.percentile(mse, 95)
    anomalies = mse > threshold
    anomalies = anomalies.flatten()
    anomaly_indices = np.arange(look_back, look_back + len(anomalies))

    peak_anomalies = []
    window_size = 10

    for start in range(0, len(anomaly_indices), window_size):
        end = start + window_size
        window_indices = anomaly_indices[start:end]
        window_anomalies = anomalies[start:end]
        if any(window_anomalies):
            max_index = window_indices[np.argmax(df['log_count'].iloc[window_indices])]
            if df['log_count'].iloc[max_index] >= 0.03 and original_log_counts[max_index] >= 20:
                peak_anomalies.append(max_index)

    return peak_anomalies

def plot_anomalies(timestamps, original_log_counts, peak_anomalies, interval):
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, original_log_counts, label='Число логов')
    plt.scatter(timestamps[peak_anomalies], original_log_counts[peak_anomalies], color='red', label='Аномалии', zorder=2)
    plt.title(f'Логи с аномалиями Suricata(Минтранс) ({interval})')
    plt.xlabel('Время')
    plt.ylabel('Кол-во логов')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plot_path = f'/home/host/surimodel/img/anomalies_log_count_plot_{interval}.png'
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

def process_suricata_logs(interval='7d'):
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
        start_time = datetime.strptime('2024-06-11', '%Y-%m-%d')
    else:
        start_time = now - timedelta(days=1)

    log_files = []
    for single_date in (start_time + timedelta(n) for n in range((now - start_time).days + 1)):
        log_files.extend(glob.glob(f'/home/host/surimodel/fromsuri/eve-{single_date.strftime("%Y-%m-%d")}-*.json'))

    filtered_logs = filter_logs(log_files)
    filtered_logs_file = save_filtered_logs(filtered_logs)
    df, timestamps, original_log_counts = prepare_data(filtered_logs_file)
    peak_anomalies = detect_anomalies(df, timestamps, original_log_counts)
    print("peak_anomalies_suri", peak_anomalies)

    return peak_anomalies, timestamps, original_log_counts
