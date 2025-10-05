import pandas as pd
import numpy as np
import torch
import joblib
import os
import argparse
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, set_seed
from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from sklearn.metrics import mean_absolute_error

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from src.patchtst_model import create_regression_model, create_probability_model

set_seed(2025)

def train_task(task_type: str):
    if task_type not in ['regression', 'probability']: raise ValueError("...")

    print(f"\n{'='*20}\n ЗАПУСК ОБУЧЕНИЯ: {task_type.upper()} \n{'='*20}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Устройство: {device.upper()} ---")

    print("1/4: Загрузка данных...")
    df = pd.read_parquet(FINAL_TRAIN_DF_PATH)

    # --- ФИНАЛЬНОЕ ИСПРАВЛЕНИЕ ---
    feature_columns = [col for col in df.columns if col not in ['ticker', 'begin'] and not col.startswith('target_')]
    
    if task_type == 'regression':
        future_columns = [f'target_return_{i}d' for i in range(1, PREDICTION_LENGTH + 1)]
        # Входные каналы - это признаки + будущие цели
        all_input_columns = feature_columns + future_columns
        num_input_channels = len(all_input_columns)
        model_path, preprocessor_path = REG_MODEL_PATH, REG_PREPROCESSOR_PATH
    else:
        future_columns = [f'target_grew_{i}d' for i in range(1, PREDICTION_LENGTH + 1)]
        all_input_columns = feature_columns + future_columns
        num_input_channels = len(all_input_columns)
        model_path, preprocessor_path = PROB_MODEL_PATH, PROB_PREPROCESSOR_PATH

    print(f"  ✓ Данные: {df.shape}. Входных каналов: {num_input_channels}. Целей: {len(future_columns)}.")

    print("\n2/4: Обучение финальной модели...")
    train_df, val_df = split_data_for_final_train(df)

    preprocessor = TimeSeriesPreprocessor(
        timestamp_column="begin",
        id_columns=["ticker"],
        target_columns=all_input_columns, # <- ВСЕ ВХОДНЫЕ ДАННЫЕ ЗДЕСЬ
        scaling=True
    )
    preprocessor.train(train_df)
    
    train_dataset = ForecastDFDataset(
        preprocessor.preprocess(train_df),
        id_columns=["ticker"], timestamp_column="begin",
        target_columns=all_input_columns, # <- И ЗДЕСЬ ТОЖЕ
        context_length=CONTEXT_LENGTH, prediction_length=PREDICTION_LENGTH
    )
    val_dataset = ForecastDFDataset(
        preprocessor.preprocess(val_df),
        id_columns=["ticker"], timestamp_column="begin",
        target_columns=all_input_columns, # <- И ЗДЕСЬ ТОЖЕ
        context_length=CONTEXT_LENGTH, prediction_length=PREDICTION_LENGTH
    )

    if task_type == 'regression': model = create_regression_model(num_input_channels)
    else: model = create_probability_model(num_input_channels)

    # В Trainer мы должны указать, какие из future_values являются настоящими целями
    training_args = create_training_args(model_path)
    training_args.label_names = ["future_values"]


    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, callbacks=[EarlyStoppingCallback(10)])
    
    print("\n3/4: Запуск обучения...")
    trainer.train()
    
    results = trainer.evaluate()
    print(f"  ✓ Финальные потери на валидации (MSE): {results['eval_loss']:.6f}")

    print("\n4/4: Сохранение артефактов...")
    trainer.save_model(model_path)
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
    joblib.dump(preprocessor, preprocessor_path)
    
    # Сохраняем только скейлеры для НАСТОЯЩИХ таргетов
    target_scalers = {k: v for k, v in preprocessor.scaler_dict.items() if k in future_columns}
    scaler_path = preprocessor_path.replace('.pkl', '_scalers.pkl')
    joblib.dump(target_scalers, scaler_path)
    
    print(f"\n✅ Пайплайн '{task_type}' завершен!")

# Вспомогательные функции
def create_training_args(output_dir, num_epochs=100): # ИСПРАВЛЕНО
    return TrainingArguments(
        output_dir=output_dir, overwrite_output_dir=True, num_train_epochs=num_epochs,
        do_eval=True, evaluation_strategy="epoch", per_device_train_batch_size=256,
        per_device_eval_batch_size=256, save_strategy="epoch", logging_strategy="epoch",
        save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, label_names=["future_values"], learning_rate=1e-4, report_to="none"
    )

def split_data_for_final_train(df):
    unique_dates = sorted(df['begin'].unique())
    split_date = unique_dates[int(len(unique_dates) * 0.9)]
    return df[df['begin'] < split_date], df[df['begin'] >= split_date]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Обучение моделей PatchTST.")
    parser.add_argument('task', choices=['regression', 'probability', 'all'], help="Задача для обучения.")
    args = parser.parse_args()

    if args.task == 'all':
        train_task('regression')
        train_task('probability')
    else:
        train_task(args.task)