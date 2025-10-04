import pandas as pd
import numpy as np
import logging
import sys
import os
import joblib
from sklearn.preprocessing import StandardScaler

# --- Импорты из Hugging Face и tsfm ---
from transformers import (
    PatchTSTConfig,
    PatchTSTForPrediction,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor

# --- Блок для корректного импорта ---
try:
    from config import *
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import *
# --- Конец блока ---

set_seed(42)

def run_model_training():
    """
    Основной пайплайн для обучения модели PatchTST.
    """
    logging.info("="*50)
    logging.info("🚀 Запуск пайплайна обучения модели PatchTST...")
    logging.info("="*50)

    # 1. Загрузка финального датасета
    logging.info("--- Шаг 1/6: Загрузка финального датасета ---")
    try:
        df = pd.read_parquet(FINAL_TRAIN_DF_PATH)
        df['date'] = pd.to_datetime(df['date'])
    except FileNotFoundError:
        logging.error(f"Файл {FINAL_TRAIN_DF_PATH} не найден.")
        return

    # 2. Определение колонок для модели
    logging.info("--- Шаг 2/6: Определение признаков и таргетов ---")
    target_columns = [f'target_return_{i}d' for i in range(1, PREDICTION_LENGTH + 1)]
    reserved_cols = ['date', 'ticker'] + target_columns
    context_columns = [col for col in df.columns if col not in reserved_cols]
    
    logging.info(f"Найдено {len(context_columns)} признаков (context_columns).")
    logging.info(f"Найдено {len(target_columns)} таргетов (target_columns).")

    # 3. Разбиение данных на train/validation
    logging.info("--- Шаг 3/6: Разбиение данных на обучающую и валидационную выборки ---")
    unique_dates = sorted(df['date'].unique())
    split_date = unique_dates[int(len(unique_dates) * 0.8)]
    train_df = df[df['date'] < split_date]
    val_df = df[df['date'] >= split_date]
    logging.info(f"Обучение: {len(train_df)} строк. Валидация: {len(val_df)} строк.")

    # 4. Препроцессинг и создание датасетов
    logging.info("--- Шаг 4/6: Препроцессинг и создание PyTorch датасетов ---")
    
    def scale_per_ticker(df, columns, fit=False, scalers=None):
        if scalers is None:
            scalers = {}
        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            if fit:
                scalers[ticker] = StandardScaler().fit(df.loc[mask, columns])
            df.loc[mask, columns] = scalers[ticker].transform(df.loc[mask, columns])
        return df, scalers

    train_df, context_scalers = scale_per_ticker(train_df, context_columns, fit=True)
    val_df, _ = scale_per_ticker(val_df, context_columns, scalers=context_scalers)

    # Препроцессор масштабирует ТОЛЬКО признаки (context_columns)
    preprocessor = TimeSeriesPreprocessor(
        timestamp_column="date",
        id_columns=["ticker"],
        input_columns=context_columns,
        target_columns=target_columns,
        scaling=True
    )
    preprocessor.train(train_df)
    
    # --- ИСПРАВЛЕННЫЙ БЛОК СОГЛАСНО HELP() ---
    train_dataset = ForecastDFDataset(
        preprocessor.preprocess(train_df),
        id_columns=["ticker"],
        timestamp_column="date",
        target_columns=target_columns,       # <--- Здесь ТОЛЬКО таргеты
        conditional_columns=context_columns, # <--- Здесь ТОЛЬКО признаки
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
    )
    val_dataset = ForecastDFDataset(
        preprocessor.preprocess(val_df),
        id_columns=["ticker"],
        timestamp_column="date",
        target_columns=target_columns,       # <--- Здесь ТОЛЬКО таргеты
        conditional_columns=context_columns, # <--- Здесь ТОЛЬКО признаки
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
    )
    # --- КОНЕЦ ИСПРАВЛЕНИЙ ---
    logging.info("Препроцессор и датасеты созданы.")

    # 5. Конфигурация модели и Trainer'а
    logging.info("--- Шаг 5/6: Конфигурация модели и Trainer ---")
    
    # Общее количество каналов, которые датасет подаст в модель
    num_total_channels = len(target_columns) + len(context_columns)
    
    config = PatchTSTConfig(
        # Модель ожидает общее количество каналов
        num_input_channels=num_total_channels, # <--- Корректное значение
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        # Количество каналов, которые являются таргетами. Модель будет предсказывать только их.
        num_output_channels=len(target_columns), # <--- Важный параметр для разделения
        patch_length=PATCH_LENGTH,
        patch_stride=PATCH_LENGTH,
        d_model=D_MODEL,
        num_attention_heads=N_HEADS,
        num_hidden_layers=ENCODER_LAYERS,
        ffn_dim=D_MODEL * 2,
        dropout=DROPOUT,
        head_dropout=DROPOUT,
        loss="mse",
    )
    model = PatchTSTForPrediction(config)

    training_args = TrainingArguments(
        output_dir=PATCHTST_MODEL_PATH.parent / "checkpoints",
        overwrite_output_dir=True,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        do_eval=True,
        evaluation_strategy="epoch", 
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        fp16=True,
        dataloader_num_workers=4,
        save_strategy="epoch",
        save_total_limit=1,
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        label_names=["future_values"],
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )
    logging.info("Конфигурация завершена. Начинаем обучение...")

    # 6. Обучение и сохранение
    trainer.train()
    
    logging.info("\n--- Шаг 6/6: Сохранение артефактов ---")
    results = trainer.evaluate()
    logging.info(f"Финальные потери на валидации (MSE): {results['eval_loss']:.6f}")

    trainer.save_model(PATCHTST_MODEL_PATH)
    artifacts_to_save = {
        'preprocessor': preprocessor,
        'context_scalers': context_scalers
    }
    joblib.dump(artifacts_to_save, PATCHTST_PREPROCESSOR_PATH)

    logging.info(f" Модель сохранена в: {PATCHTST_MODEL_PATH}")
    logging.info(f" Препроцессор сохранен в: {PATCHTST_PREPROCESSOR_PATH}")
    logging.info("\nПайплайн обучения успешно завершен!")


if __name__ == '__main__':
    run_model_training()