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
    
    logging.info("--- Шаг 4/4: ГЛУБОКАЯ ОТЛАДКА РЕЗУЛЬТАТОВ ---")
    
    predictions_tuple = predictions_output.predictions

    # # --- БЛОК ДЛЯ ГЛУБОКОЙ ОТЛАДКИ ---
    # print("\n" + "="*20 + " НАЧАЛО ОТЛАДКИ " + "="*20)
    
    # print(f"Тип predictions_output.predictions: {type(predictions_tuple)}")
    
    # # Проверяем, действительно ли это кортеж
    # if isinstance(predictions_tuple, tuple):
    #     print(f"Длина кортежа (количество элементов): {len(predictions_tuple)}")
        
    #     print("\n--- Анализ каждого элемента в кортеже ---")
    #     for idx, element in enumerate(predictions_tuple):
    #         print(f"\n--- Элемент #{idx} ---")
    #         print(f"Тип элемента: {type(element)}")
            
    #         # Если элемент - это массив NumPy, выводим его форму
    #         if isinstance(element, np.ndarray):
    #             print(f"Форма элемента: {element.shape}")
    #         else:
    #             # Если это не массив, просто выводим его
    #             print(f"Содержимое элемента: {element}")
    # else:
    #     # Если это не кортеж, выводим информацию о том, что это
    #     print("Объект predictions_output.predictions не является кортежем.")
    #     if isinstance(predictions_tuple, np.ndarray):
    #         print(f"Это массив NumPy с формой: {predictions_tuple.shape}")

    # print("\n" + "="*20 + " КОНЕЦ ОТЛАДКИ " + "="*20)
    
    # # Прерываем выполнение, чтобы изучить вывод
    # logging.info("Отладка завершена. Прерываю выполнение.")
    # sys.exit()


    # 4. Форматирование и сохранение результата
    logging.info("--- Шаг 4/4: Форматирование и сохранение результатов ---")
    
    # --- ГЛАВНОЕ ИСПРАВЛЕНИЕ НА ОСНОВЕ ОТЛАДКИ ---
    # 1. Берем ЭЛЕМЕНТ #1 из кортежа - это тензор (19, 20, 366)
    predictions_for_all_channels = predictions_output.predictions[1]
    
    # 2. Вырезаем из него предсказания только для наших 20 целевых переменных (первые 20 каналов)
    # Получаем тензор (19, 20, 20)
    predictions_raw = predictions_for_all_channels[:, :, :PREDICTION_LENGTH]

    last_known_date = df_full['date'].max()
    prediction_dates = pd.bdate_range(start=last_known_date + pd.Timedelta(days=1), periods=PREDICTION_LENGTH)
    
    results_list = []
    tickers_in_order = sorted(df_inference['ticker'].unique())
    
    for i, ticker in enumerate(tickers_in_order):
        # predictions_raw[i] - это матрица (20, 20) для одного тикера
        prediction_matrix_for_ticker = predictions_raw[i]
        
        # Берем главную диагональ этой матрицы
        diagonal_predictions = np.diag(prediction_matrix_for_ticker)
        
        for j in range(PREDICTION_LENGTH):
            results_list.append({
                'ticker': ticker,
                'date': prediction_dates[j],
                'predicted_return': diagonal_predictions[j]
            })
            
    df_long = pd.DataFrame(results_list)
    
    final_end_date = pd.to_datetime('2025-09-28') # Укажите корректный год из ваших данных
    df_long = df_long[df_long['date'] <= final_end_date]
    
    # --- НОВЫЙ БЛОК: ТРАНСФОРМАЦИЯ В ФОРМАТ SUBMISSION ---
    logging.info("--- Преобразование предсказаний в формат submission ---")
    
    # 1. Создаем колонку с номером предсказания (p1, p2, ...)
    # groupby().cumcount() создает счетчик для каждого тикера (0, 1, 2...)
    df_long['p_col_num'] = df_long.groupby('ticker').cumcount() + 1
    df_long['p_col'] = 'p' + df_long['p_col_num'].astype(str)
    
    # 2. Используем pivot для преобразования из "длинного" формата в "широкий"
    df_submission = df_long.pivot(
        index='ticker', 
        columns='p_col', 
        values='predicted_return'
    )
    
    # 3. Приводим в порядок колонки
    # Убедимся, что колонки идут в правильном порядке (p1, p2, ..., p20)
    p_columns_ordered = [f'p{i}' for i in range(1, PREDICTION_LENGTH + 1)]
    # Оставляем только те колонки, которые есть в нашем датафрейме
    p_columns_to_use = [col for col in p_columns_ordered if col in df_submission.columns]
    
    df_submission = df_submission[p_columns_to_use]
    
    # Сбрасываем индекс, чтобы 'ticker' стал обычной колонкой
    df_submission.reset_index(inplace=True)
    
    df_submission.to_csv(SUBMISSION_PATH, index=False)
    
    logging.info(f"\n Финальный submission-файл успешно сохранен в: {SUBMISSION_PATH}")
    logging.info(f"Пример submission-файла:\n{df_submission.head().to_string()}")


if __name__ == '__main__':
    run_prediction()