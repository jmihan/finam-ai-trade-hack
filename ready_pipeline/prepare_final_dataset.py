import pandas as pd
import numpy as np
import logging
import sys
import os

# --- Блок для корректного импорта ---
try:
    from config import *
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import *
# --- Конец блока ---


def create_future_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Рассчитывает будущие таргеты (доходность) для каждого тикера
    на `horizon` дней вперед.
    """
    logging.info(f"  > Расчет таргетов на {horizon} дней вперед...")
    df_out = df.copy()
    
    # Группируем по тикеру для корректных сдвигов
    grouped = df_out.groupby('ticker')
    
    for i in range(1, horizon + 1):
        # Формула доходности: (Цена_закрытия_будущая / Цена_закрытия_сегодня) - 1
        df_out[f'target_return_{i}d'] = grouped['close'].shift(-i) / df_out['close'] - 1
        
    return df_out


def run_final_dataset_preparation():
    """
    Основной пайплайн подготовки финального датасета для обучения.
    Объединяет TS и NLP признаки, создает таргеты и сохраняет результат.
    """
    logging.info("="*50)
    logging.info(" Запуск подготовки финального датасета...")
    logging.info("="*50)

    # 1. Загрузка обработанных признаков
    logging.info("--- Шаг 1/4: Загрузка TS и NLP признаков ---")
    try:
        df_ts = pd.read_parquet(PROCESSED_TS_FEATURES_PATH)
        df_nlp = pd.read_parquet(PROCESSED_NEWS_FEATURES_PATH)
        df_ts.rename(columns={'begin': 'date'}, inplace=True)
        logging.info(f"TS-признаки загружены. Форма: {df_ts.shape}")
        logging.info(f"NLP-признаки загружены. Форма: {df_nlp.shape}")
    except FileNotFoundError as e:
        logging.error(f"ОШИБКА: Не удалось загрузить файл с признаками: {e}. "
                      "Убедитесь, что скрипты generate_... отработали корректно.")
        return

    # 2. Создание карты торговых дней С УЧЕТОМ ВЫХОДНЫХ
    logging.info("--- Шаг 2/4: Создание карты торговых дней для корректного сдвига ---")
    df_ts = df_ts.sort_values(by=['ticker', 'date'])
    df_ts['trade_day_rank'] = df_ts.groupby('ticker').cumcount()

    # --- НОВЫЙ БЛОК: Создание полного календаря и forward fill ---
    all_tickers = df_ts['ticker'].unique()
    full_calendar = pd.date_range(start=df_ts['date'].min(), end=df_ts['date'].max(), freq='D')
    
    # Создаем "шаблон" со всеми днями для всех тикеров
    calendar_template = pd.MultiIndex.from_product([all_tickers, full_calendar], names=['ticker', 'date'])
    df_full_calendar = pd.DataFrame(index=calendar_template).reset_index()

    # Накладываем наши данные о рангах на полный календарь
    df_ts_ranks = df_ts[['ticker', 'date', 'trade_day_rank']]
    df_mapper = pd.merge(df_full_calendar, df_ts_ranks, on=['ticker', 'date'], how='left')

    # КЛЮЧЕВОЙ МОМЕНТ: Заполняем пропуски в выходные последним известным рангом
    df_mapper['trade_day_rank'] = df_mapper.groupby('ticker')['trade_day_rank'].ffill()
    df_mapper.dropna(subset=['trade_day_rank'], inplace=True) # Удаляем дни до начала торгов тикера
    # --- КОНЕЦ НОВОГО БЛОКА ---

    # Присваиваем ранг каждой новости, используя новый полный маппер
    # Используем `merge_asof` для ближайшей предыдущей даты, если точной нет
    df_nlp = df_nlp.sort_values('date')
    df_mapper = df_mapper.sort_values('date')
    df_nlp = pd.merge_asof(df_nlp, df_mapper, on='date', by='ticker', direction='backward')
    df_nlp.dropna(subset=['trade_day_rank'], inplace=True)
    df_nlp['trade_day_rank'] = df_nlp['trade_day_rank'].astype(int)

    # Сдвигаем ранг на +1. Теперь новости с пт, сб, вс указывают на пн.
    df_nlp['trade_day_rank'] += 1
    
    # --- НОВЫЙ БЛОК: Агрегация новостей, относящихся к одному дню ---
    logging.info("---  Агрегация новостей (за выходные и т.д.) ---")
    
    # Определяем, как агрегировать каждую колонку
    nlp_feature_cols = [col for col in df_nlp.columns if col not in ['date', 'ticker', 'trade_day_rank']]
    agg_funcs = {}
    # Эмбеддинги и сентимент - усредняем
    agg_funcs.update({col: 'mean' for col in nlp_feature_cols if 'emb_' in col or 'sentiment_' in col})
    # Количественные признаки, которые были средними - тоже усредняем
    agg_funcs.update({col: 'mean' for col in nlp_feature_cols if 'mean' in col})
    # Количественные признаки, которые были суммами/счетчиками - складываем
    agg_funcs.update({col: 'sum' for col in nlp_feature_cols if 'sum' in col or 'count' in col or 'num_news' in col})
    
    df_nlp_agg = df_nlp.groupby(['ticker', 'trade_day_rank']).agg(agg_funcs).reset_index()
    # --- КОНЕЦ НОВОГО БЛОКА ---

    # Добавляем префикс к колонкам NLP, чтобы избежать конфликтов имен
    nlp_cols = [col for col in df_nlp_agg.columns if col not in ['date', 'ticker', 'trade_day_rank']]
    df_nlp_agg.rename(columns={col: f'nlp_{col}' for col in nlp_cols}, inplace=True)
    
    # 3. Объединение признаков по торговому рангу
    logging.info("--- Шаг 3/4: Объединение TS и NLP признаков по рангу торгового дня ---")
    # Используем `left join` по `ticker` и `trade_day_rank`
    df_final = pd.merge(df_ts, df_nlp_agg, on=['ticker', 'trade_day_rank'], how='left')
    
    # Заполняем пропуски в NLP-признаках нулями
    nlp_feature_columns = [col for col in df_final.columns if col.startswith('nlp_')]
    df_final[nlp_feature_columns] = df_final[nlp_feature_columns].fillna(0)
    
    # Удаляем временные и ненужные колонки
    df_final.drop(columns=['trade_day_rank', 'nlp_date'], inplace=True, errors='ignore')
    
    logging.info(f"Признаки объединены. Форма после merge: {df_final.shape}")

    # 4. Создание таргетов
    logging.info("--- Шаг 4/4: Создание целевых переменных (таргетов) ---")
    # Используем PREDICTION_LENGTH из конфига
    df_final = create_future_targets(df_final, horizon=PREDICTION_LENGTH)
    
    # # Важно: удаляем строки, где мы не смогли рассчитать таргеты (хвост данных)
    # df_final.dropna(subset=[f'target_return_{PREDICTION_LENGTH}d'], inplace=True)
    # df_final.reset_index(drop=True, inplace=True)

    # # Сохранение
    # df_final.to_parquet(FINAL_TRAIN_DF_PATH, index=False)

    logging.info("--- Разделение на обучающие и инференс-данные ---")

    # Данные для инференса - это те строки, где таргеты посчитать не удалось (хвост)
    is_inference_data = df_final[f'target_return_{PREDICTION_LENGTH}d'].isnull()
    df_inference = df_final[is_inference_data].copy()

    # Обучающие данные - все остальные
    df_train = df_final[~is_inference_data].copy()

    # Сохраняем данные для инференса
    df_inference.to_parquet(INFERENCE_DF_PATH, index=False)
    logging.info(f" Данные для предсказания сохранены в {INFERENCE_DF_PATH} ({len(df_inference)} строк)")

    # Сбрасываем индекс у обучающих данных перед сохранением
    df_train.reset_index(drop=True, inplace=True)

    # Сохранение обучающих данных
    df_train.to_parquet(FINAL_TRAIN_DF_PATH, index=False)
    
    logging.info(f"\n Готово! Финальный датасет сохранен в {FINAL_TRAIN_DF_PATH}")
    logging.info(f"   Финальная форма: {df_train.shape}")
    
    # Проверка
    target_cols_exist = all(f'target_return_{i}d' in df_train.columns for i in [1, 5, 20])
    logging.info(f"   Проверка наличия таргетных колонок: {'Успешно' if target_cols_exist else 'Ошибка'}")
    logging.info(f"   Количество NaN в датасете: {df_train.isnull().sum().sum()}")


if __name__ == '__main__':
    run_final_dataset_preparation()