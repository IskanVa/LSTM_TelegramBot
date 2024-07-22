import os
from datetime import datetime, timedelta
import glob
import numpy as np
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, JobQueue
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from suricata_handler import copy_logs as suricata_copy_logs, process_suricata_logs, plot_anomalies as suricata_plot_anomalies, load_detected_anomalies as suricata_load_detected_anomalies, save_detected_anomalies as suricata_save_detected_anomalies
from tpot_handler import copy_logs as tpot_copy_logs, process_tpot_logs, plot_anomalies as tpot_plot_anomalies, load_detected_anomalies as tpot_load_detected_anomalies, save_detected_anomalies as tpot_save_detected_anomalies

# Ограничение использования видеопамяти
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    except RuntimeError as e:
        print(e)

# Телеграм токен
TOKEN = ''
AUTHORIZED_USERS = ['']

async def send_plot(plot_path, chat_id):
    bot = Bot(token=TOKEN)
    async with bot:
        await bot.send_photo(chat_id=chat_id, photo=open(plot_path, 'rb'))

async def notify_users(new_anomalies, interval, timestamps, original_log_counts, system):
    if system == 'suricata':
        plot_path = suricata_plot_anomalies(timestamps, original_log_counts, new_anomalies, interval)
    else:
        plot_path = tpot_plot_anomalies(timestamps, original_log_counts, new_anomalies, interval)
        
    for user_id in AUTHORIZED_USERS:
        await send_plot(plot_path, user_id)
        bot = Bot(token=TOKEN)
        async with bot:
            await bot.send_message(chat_id=user_id, text=f"Обнаружена новая аномалия за {interval}")

async def notify_users_combined(interval, suricata_data, tpot_data):
    timestamps_s, original_log_counts_s, peak_anomalies_s = suricata_data
    timestamps_t, original_log_counts_t, peak_anomalies_t = tpot_data
    
    plot_path = plot_combined_anomalies(timestamps_s, original_log_counts_s, peak_anomalies_s, 
                                        timestamps_t, original_log_counts_t, peak_anomalies_t, interval)
    for user_id in AUTHORIZED_USERS:
        await send_plot(plot_path, user_id)
        bot = Bot(token=TOKEN)
        async with bot:
            await bot.send_message(chat_id=user_id, text=f"Обнаружены новые аномалии за {interval}")

async def check_anomalies(system, context=None):
    now = datetime.now()
    start_time = now - timedelta(days=7)
    end_time = now

    if system == 'suricata':
        peak_anomalies, timestamps, original_log_counts = process_suricata_logs('7d')
        print("check_anomalies suricata")
        detected_anomalies = suricata_load_detected_anomalies()
    else:
        peak_anomalies, timestamps, original_log_counts = process_tpot_logs('7d')
        print("check_anomalies_tpot", peak_anomalies)
        detected_anomalies = tpot_load_detected_anomalies()
        
    print("Detected anomalies:", detected_anomalies)
    peak_anomalies = [int(a) for a in peak_anomalies]
    detected_anomalies = [int(a) for a in detected_anomalies]

    new_anomalies = [a for a in peak_anomalies if a not in detected_anomalies]
    new_anomalies = [int(a) if isinstance(a, np.int64) else a for a in new_anomalies]
    print("New anomalies:", new_anomalies)

    if new_anomalies:
        await notify_users(new_anomalies, 'последняя неделя', timestamps, original_log_counts, system)
        detected_anomalies.extend(new_anomalies)
        
        if system == 'suricata':
            suricata_save_detected_anomalies(detected_anomalies)
        else:
            tpot_save_detected_anomalies(detected_anomalies)

async def check_combined_anomalies(context=None):
    suricata_data = process_suricata_logs('7d')
    tpot_data = process_tpot_logs('7d')

    await notify_users_combined('последняя неделя', suricata_data, tpot_data)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if str(update.message.chat_id) not in AUTHORIZED_USERS:
        await update.message.reply_text('А кто это решил заглянуть? Покажи пропуск или напиши @iskanVal')
        return

    keyboard = [
        [
            InlineKeyboardButton("🛡️ Suricata(Минтранс)", callback_data='system_suricata'),
            InlineKeyboardButton("🐝 T-Pot(Минтранс)", callback_data='system_tpot')
        ],
        [
            InlineKeyboardButton("📊 Суммарный вывод аномалий", callback_data='system_combined')
        ]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('Выберите систему для анализа аномалий:', reply_markup=reply_markup)

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    chat_id = query.message.chat_id
    await query.answer()

    if str(query.message.chat_id) not in AUTHORIZED_USERS:
        await query.message.reply_text('А кто это решил заглянуть? Покажи пропуск или напиши @iskanVal')
        return

    if query.data.startswith(('system_s', 'system_t')):
        system = query.data.split('_')[1]
        context.user_data['system'] = system

        keyboard = [
            [
                InlineKeyboardButton("⏳ Последние сутки", callback_data='1d'),
                InlineKeyboardButton("📅 Последние 3 дня", callback_data='3d'),
            ],
            [
                InlineKeyboardButton("📆 Последняя неделя", callback_data='7d'),
                InlineKeyboardButton("📅 Последние 3 недели", callback_data='21d'),
            ],
            [
                InlineKeyboardButton("🔄 Увидеть за всё время", callback_data='all_time'),
            ]
        ]

        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.reply_text('Выберите период для анализа:', reply_markup=reply_markup)
    else:
        system = query.data.split('_')[1]
        context.user_data['system'] = system
        await handle_time_period(update, context)

async def handle_time_period(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("handle_time_period")
    query = update.callback_query
    chat_id = query.message.chat_id
    await query.answer()

    if str(query.message.chat_id) not in AUTHORIZED_USERS:
        await query.message.reply_text('А кто это решил заглянуть? Покажи пропуск или напиши @iskanVal')
        return

    interval = query.data
    system = context.user_data['system']

    if system == 'suricata':
        peak_anomalies, timestamps, original_log_counts = process_suricata_logs(interval)
        plot_path = suricata_plot_anomalies(timestamps, original_log_counts, peak_anomalies, interval)
    elif system == 'tpot':
        peak_anomalies, timestamps, original_log_counts = process_tpot_logs(interval)
        plot_path = tpot_plot_anomalies(timestamps, original_log_counts, peak_anomalies, interval)
    elif system == 'combined':
        interval = "7d"
        print("interval", interval)
        suricata_data = process_suricata_logs(interval)
        tpot_data = process_tpot_logs(interval)
        plot_path = plot_combined_anomalies(suricata_data[1], suricata_data[2], suricata_data[0], 
                                            tpot_data[1], tpot_data[2], tpot_data[0], interval)

    await send_plot(plot_path, chat_id)
    await query.edit_message_text(text=f"График с аномалиями за {interval} отправлен!")

def plot_combined_anomalies(timestamps_s, log_counts_s, anomalies_s, timestamps_t, log_counts_t, anomalies_t, interval):
    plt.figure(figsize=(15, 7))
    
    plt.plot(timestamps_s, log_counts_s, label='Логи Suricata')
    plt.scatter(timestamps_s[anomalies_s], log_counts_s[anomalies_s], color='red', label='Аномалии Suricata', zorder=2)
    
    plt.plot(timestamps_t, log_counts_t, label='Логи T-Pot')
    plt.scatter(timestamps_t[anomalies_t], log_counts_t[anomalies_t], color='indigo', label='Аномалии T-Pot', zorder=2)
    
    plt.legend()
    plt.title(f'Объединенные аномалии обнаруженные за  {interval}')
    plt.xlabel('Время')
    plt.ylabel('Число логов')
    
    plot_path = f'/home/host/tpotmodel/img/combined_anomalies_log_count_plot_{interval}.png'
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

def main() -> None:
    application = Application.builder().token(TOKEN).build()

    job_queue = application.job_queue
    job_queue.run_repeating(suricata_copy_logs, interval=3600, first=5)
    job_queue.run_repeating(tpot_copy_logs, interval=3600, first=10)
    job_queue.run_repeating(lambda context: check_anomalies('suricata', context), interval=3600, first=35)
    job_queue.run_repeating(lambda context: check_anomalies('tpot', context), interval=3600, first=70)

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button))

    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
