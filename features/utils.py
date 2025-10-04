import pandas as pd
import os
import logging
from config import * # Импортируем все из config

def load_cache(filename):
    if os.path.exists(filename):
        logging.info(f"Загружаю существующий кэш из '{filename}'...")
        try:
            return pd.read_parquet(filename)
        except Exception as e:
            logging.error(f"Ошибка при загрузке кэша '{filename}': {e}. Удаляю поврежденный кэш и создаю новый.")
            os.remove(filename)
    return pd.DataFrame()

def save_cache(df, filename):
    df.to_parquet(filename, index=False)
    logging.info(f"\nКэш сохранен в '{filename}' ({len(df)} записей).")
