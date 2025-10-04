import pandas as pd
import numpy as np
import logging
import sys
import os
import joblib
import torch
from transformers import PatchTSTForPrediction, Trainer, TrainingArguments
from tsfm_public.toolkit.dataset import ForecastDFDataset

# --- Блок для корректного импорта ---
try:
    from config import *
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import *
# --- Конец блока ---

def run_prediction():
    """
    Основной пайплайн для получения предсказаний на основе обученной модели.
    """
    logging.info("="*50)
    logging.info("🚀 Запуск пайплайна предсказания...")
    logging.info("="*50)

    # 1. Загрузка модели и артефактов
    logging.info("--- Шаг 1/4: Загрузка модели и артефактов ---")
    try:
        model = PatchTSTForPrediction.from_pretrained(PATCHTST_MODEL_PATH).to(DEVICE)
        model.eval()
        
        artifacts = joblib.load(PATCHTST_PREPROCESSOR_PATH)
        preprocessor = artifacts['preprocessor']
        context_scalers = artifacts['context_scalers']
        
        logging.info("Модель, препроцессор и скейлеры успешно загружены.")
    except Exception as e:
        logging.error(f"Ошибка при загрузке артефактов: {e}.")
        return

    # 2. Загрузка и подготовка данных
    logging.info("--- Шаг 2/4: Загрузка и подготовка данных ---")
    try:
        df_train = pd.read_parquet(FINAL_TRAIN_DF_PATH)
        df_inference_tail = pd.read_parquet(INFERENCE_DF_PATH)
        df_full = pd.concat([df_train, df_inference_tail]).sort_values(['ticker', 'date'])
    except FileNotFoundError as e:
        logging.error(f"Не найден файл с данными: {e}. Запустите prepare_final_dataset.py.")
        return
        
    inference_chunks = [df_full[df_full['ticker'] == ticker].tail(CONTEXT_LENGTH) for ticker in df_full['ticker'].unique()]
    df_inference = pd.concat(inference_chunks)

    target_columns = [f'target_return_{i}d' for i in range(1, PREDICTION_LENGTH + 1)]
    reserved_cols = ['date', 'ticker'] + target_columns
    context_columns = [col for col in df_full.columns if col not in reserved_cols]
    df_inference[target_columns] = df_inference[target_columns].fillna(0.0)

    # 3. Препроцессинг и предсказание
    logging.info("--- Шаг 3/4: Препроцессинг и выполнение предсказания ---")
    for ticker in df_inference['ticker'].unique():
        if ticker in context_scalers:
            mask = df_inference['ticker'] == ticker
            df_inference.loc[mask, context_columns] = context_scalers[ticker].transform(df_inference.loc[mask, context_columns])
            
    processed_inference_df = preprocessor.preprocess(df_inference)
    inference_dataset = ForecastDFDataset(processed_inference_df, id_columns=["ticker"], timestamp_column="date",
                                          target_columns=target_columns, conditional_columns=context_columns,
                                          context_length=CONTEXT_LENGTH, prediction_length=PREDICTION_LENGTH)

    trainer = Trainer(model=model, args=TrainingArguments(output_dir="./temp_output", per_device_eval_batch_size=BATCH_SIZE))
    predictions_output = trainer.predict(inference_dataset)
    
    # 4. Форматирование и сохранение результата
    logging.info("--- Шаг 4/4: Форматирование и сохранение результатов ---")
    
    predictions_for_all_channels = predictions_output.predictions[1]
    predictions_raw = predictions_for_all_channels[:, :, :PREDICTION_LENGTH]

    # --- НОВАЯ ЛОГИКА ДЛЯ СОЗДАНИЯ SUBMISSION С УЧЕТОМ ВЫХОДНЫХ ---
    
    last_known_date = df_full['date'].max()
    tickers_in_order = sorted(df_inference['ticker'].unique())

    # 1. Создаем два списка дат: 20 КАЛЕНДАРНЫХ дней и 20 ТОРГОВЫХ дней
    calendar_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=PREDICTION_LENGTH, freq='D')
    trading_dates = pd.bdate_range(start=last_known_date + pd.Timedelta(days=1), periods=PREDICTION_LENGTH)
    
    results_list = []
    for i, ticker in enumerate(tickers_in_order):
        prediction_matrix_for_ticker = predictions_raw[i]
        diagonal_predictions = np.diag(prediction_matrix_for_ticker) # Вектор из 20 предсказаний для ТОРГОВЫХ дней

        # 2. Создаем словарь: {торговая_дата: предсказание}
        trading_predictions_map = dict(zip(trading_dates, diagonal_predictions))
        
        # 3. Итерируемся по КАЛЕНДАРНЫМ дням
        ticker_calendar_preds = {'ticker': ticker}
        for day_num, calendar_date in enumerate(calendar_dates):
            p_col = f'p{day_num + 1}'
            
            # 4. Проверяем, является ли календарный день торговым.
            # Если да - берем предсказание из словаря. Если нет (выходной) - ставим 0.
            prediction_value = trading_predictions_map.get(calendar_date, 0.0)
            ticker_calendar_preds[p_col] = prediction_value
        
        results_list.append(ticker_calendar_preds)

    # 5. Создаем финальный DataFrame
    df_submission = pd.DataFrame(results_list)
    
    # Гарантируем правильный порядок колонок
    final_columns = ['ticker'] + [f'p{i}' for i in range(1, PREDICTION_LENGTH + 1)]
    df_submission = df_submission[final_columns]
    
    # --- КОНЕЦ НОВОЙ ЛОГИКИ ---

    # Сохраняем итоговый файл
    df_submission.to_csv(SUBMISSION_PATH, index=False)
    
    logging.info(f"\n✅ Финальный submission-файл успешно сохранен в: {SUBMISSION_PATH}")
    logging.info(f"Пример submission-файла (с нулями на выходных):\n{df_submission.head().to_string()}")


if __name__ == '__main__':
    run_prediction()