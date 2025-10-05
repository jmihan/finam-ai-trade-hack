import pandas as pd
import os
import logging

def load_cache(filename: str) -> pd.DataFrame:
    """Загружает кэш из parquet файла, если он существует."""
    if os.path.exists(filename):
        logging.info(f"Загрузка кэша из '{filename}'...")
        try:
            return pd.read_parquet(filename)
        except Exception as e:
            logging.error(f"Ошибка при загрузке кэша '{filename}': {e}. Поврежденный файл будет перезаписан.")
            os.remove(filename)
    return pd.DataFrame()

def save_cache(df: pd.DataFrame, filename: str):
    """Сохраняет DataFrame в parquet файл."""
    try:
        df.to_parquet(filename, index=False)
        logging.info(f"Кэш сохранен в '{filename}' ({len(df)} записей).")
    except Exception as e:
        logging.error(f"Не удалось сохранить кэш в '{filename}': {e}")