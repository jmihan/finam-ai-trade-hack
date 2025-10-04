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
    logging.info(" Запуск пайплайна предсказания...")
    logging.info("="*50)

    # 1. Загрузка модели и артефактов
    logging.info("--- Шаг 1/4: Загрузка модели и артефактов ---")
    try:
        model = PatchTSTForPrediction.from_pretrained(PATCHTST_MODEL_PATH).to(DEVICE)
        model.eval()
        
        # Убедитесь, что вы сохраняли артефакты в виде словаря!
        artifacts = joblib.load(PATCHTST_PREPROCESSOR_PATH)
        preprocessor = artifacts['preprocessor']
        context_scalers = artifacts['context_scalers']
        
        logging.info("Модель, препроцессор и скейлеры успешно загружены.")
    except Exception as e:
        logging.error(f"Ошибка при загрузке артефактов: {e}. Убедитесь, что train_model.py отработал и сохранил артефакты в виде словаря.")
        return

    # 2. Загрузка и подготовка данных
    logging.info("--- Шаг 2/4: Загрузка и подготовка данных ---")
    try:
        df_train = pd.read_parquet(FINAL_TRAIN_DF_PATH)
        df_inference_tail = pd.read_parquet(INFERENCE_DF_PATH)
        # Объединяем, чтобы для каждого тикера можно было взять последние CONTEXT_LENGTH точек
        df_full = pd.concat([df_train, df_inference_tail]).sort_values(['ticker', 'date'])
    except FileNotFoundError as e:
        logging.error(f"Не найден файл с данными: {e}. Запустите prepare_final_dataset.py.")
        return
        
    inference_chunks = []
    for ticker in df_full['ticker'].unique():
        # Берем последние CONTEXT_LENGTH записей для каждого тикера из полного датафрейма
        inference_chunk = df_full[df_full['ticker'] == ticker].tail(CONTEXT_LENGTH)
        
        if len(inference_chunk) < CONTEXT_LENGTH:
            logging.warning(f"Для тикера {ticker} найдено только {len(inference_chunk)} точек (нужно {CONTEXT_LENGTH}).")
        
        inference_chunks.append(inference_chunk)
        
    df_inference = pd.concat(inference_chunks)

    # Определяем колонки, как в обучении
    target_columns = [f'target_return_{i}d' for i in range(1, PREDICTION_LENGTH + 1)]
    reserved_cols = ['date', 'ticker'] + target_columns
    context_columns = [col for col in df_full.columns if col not in reserved_cols]

    # Заполняем NaN в таргет-колонках нулями, т.к. препроцессор их ожидает
    df_inference[target_columns] = df_inference[target_columns].fillna(0.0)

    # 3. Препроцессинг и предсказание
    logging.info("--- Шаг 3/4: Препроцессинг и выполнение предсказания ---")

    # Масштабируем признаки, используя ЗАГРУЖЕННЫЕ скейлеры
    for ticker in df_inference['ticker'].unique():
        if ticker in context_scalers:
            mask = df_inference['ticker'] == ticker
            df_inference.loc[mask, context_columns] = context_scalers[ticker].transform(df_inference.loc[mask, context_columns])
        else:
            logging.warning(f"Скейлер для тикера {ticker} не найден. Масштабирование пропущено.")
            
    processed_inference_df = preprocessor.preprocess(df_inference)

    inference_dataset = ForecastDFDataset(
        processed_inference_df,
        id_columns=["ticker"],
        timestamp_column="date",
        target_columns=target_columns,
        conditional_columns=context_columns,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
    )

    trainer = Trainer(model=model, args=TrainingArguments(output_dir="./temp_output", per_device_eval_batch_size=BATCH_SIZE))
    predictions_output = trainer.predict(inference_dataset)
    predictions_raw = predictions_output.predictions

    # 4. Форматирование и сохранение результата
    logging.info("--- Шаг 4/4: Форматирование и сохранение результатов ---")
    
    # Последняя дата в наших данных - 8 сентября. Предсказания начинаются со следующего торгового дня.
    last_known_date = df_full['date'].max()
    prediction_dates = pd.bdate_range(start=last_known_date + pd.Timedelta(days=1), periods=PREDICTION_LENGTH)
    
    results_list = []
    tickers_in_order = sorted(df_inference['ticker'].unique())
    
    for i, ticker in enumerate(tickers_in_order):
        # ВАЖНО: trainer.predict() возвращает предсказания в отсортированном по id_columns порядке
        ticker_predictions = predictions_raw[i] 
        for j in range(PREDICTION_LENGTH):
            results_list.append({
                'ticker': ticker,
                'date': prediction_dates[j],
                'predicted_return': ticker_predictions[j]
            })
            
    df_results = pd.DataFrame(results_list)
    
    # Фильтруем до 28 сентября включительно
    final_end_date = pd.to_datetime('2025-09-28') # Укажите корректный год из ваших данных
    df_results = df_results[df_results['date'] <= final_end_date]
    
    output_path = ARTIFACTS_DIR / 'predictions.csv'
    df_results.to_csv(output_path, index=False)
    
    logging.info(f"\n Предсказания успешно сохранены в: {output_path}")
    logging.info(f"Пример предсказаний:\n{df_results.head().to_string()}")


if __name__ == '__main__':
    run_prediction()