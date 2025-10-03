import pandas as pd
import numpy as np
import os

# --- Блок для исправления импорта ---
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
# --- Конец блока ---

def calculate_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Расчет RSI вручную."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Расчет MACD Histogram вручную."""
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return histogram

def run_preparation():
    """
    Основной пайплайн подготовки данных:
    1. Загрузка
    2. Приведение к стационарности (% изменение)
    3. Расчет признаков (RSI, MACD, волатильность, календарь)
    4. Создание таргетов
    5. Сохранение
    """
    print("🚀 Запуск ПОЛНОГО пайплайна подготовки данных (без pandas_ta)...")
    
    # 1. Загрузка данных
    print("1/5: Загрузка сырых свечей...")
    try:
        df = pd.read_csv(RAW_TRAIN_CANDLES_PATH, parse_dates=['begin'])
    except FileNotFoundError:
        print(f"ОШИБКА: Файл {RAW_TRAIN_CANDLES_PATH} не найден.")
        return
        
    df = df.sort_values(by=['ticker', 'begin']).reset_index(drop=True)
    
    # Сохраняем оригинальные цены для расчета таргетов
    original_prices = df[['ticker', 'begin', 'close']].copy()

    # 2. Приведение к стационарности
    print("2/5: Приведение рядов к стационарности (расчет доходностей)...")
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    
    # Используем pct_change() для расчета процентного изменения
    # groupby('ticker') гарантирует, что расчеты не "перетекают" между акциями
    df[ohlcv_cols] = df.groupby('ticker')[ohlcv_cols].pct_change()
    
    # Заменяем возможные inf значения на NaN, чтобы потом их обработать
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Удаляем первую строку для каждого тикера, где pct_change() дает NaN
    df = df.dropna(subset=ohlcv_cols).reset_index(drop=True)

    # 3. Генерация признаков на СТАЦИОНАРНЫХ данных
    print("3/5: Генерация TA-признаков на стационарных данных...")
    
    # Создаем DataFrame для сбора результатов, чтобы избежать фрагментации
    features_list = []
    for ticker, group in df.groupby('ticker'):
        group = group.copy()
        
        # Расчет RSI на основе дневных доходностей ('close')
        group['rsi'] = calculate_rsi(group['close'], length=14)
        
        # Расчет MACD Histogram на основе дневных доходностей ('close')
        group['macd_hist'] = calculate_macd(group['close'])
        
        # Волатильность (стандартное отклонение доходностей за 20 дней)
        group['volatility_20d'] = group['close'].rolling(20).std()
        
        features_list.append(group)
    
    df = pd.concat(features_list)
    
    # Календарные признаки
    df['day_of_week'] = df['begin'].dt.dayofweek
    df['month'] = df['begin'].dt.month
    
    # Заполняем NaN, которые появились в начале каждого ряда после расчета индикаторов
    df.fillna(method='bfill', inplace=True) # bfill, чтобы заполнить начало данными из будущего
    df.fillna(0, inplace=True) # Если весь тикер - NaN, заполняем нулями
    
    print("  ✓ TA и календарные признаки созданы.")

    # 4. Создание таргетов
    print("4/5: Создание будущих таргетов...")
    # Мерджим обратно оригинальные цены для расчета таргетов
    df = pd.merge(df, original_prices, on=['ticker', 'begin'], how='left', suffixes=('', '_orig'))
    
    grouped_orig = df.groupby('ticker')
    for i in range(1, PREDICTION_LENGTH + 1):
        df[f'target_return_{i}d'] = grouped_orig['close_orig'].shift(-i) / df['close_orig'] - 1
        df[f'target_grew_{i}d'] = (df[f'target_return_{i}d'] > 0).astype(float)
        
    df = df.drop(columns=['close_orig'])
    df.dropna(subset=[f'target_return_{PREDICTION_LENGTH}d'], inplace=True)
    df = df.reset_index(drop=True)
    
    print("  ✓ Таргеты созданы.")

    # 5. Сохранение
    print("5/5: Сохранение финального датасета...")
    os.makedirs(os.path.dirname(FINAL_TRAIN_DF_PATH), exist_ok=True)
    df.to_parquet(FINAL_TRAIN_DF_PATH, index=False)
    
    print(f"\n✅ Готово! Финальный датасет сохранен в {FINAL_TRAIN_DF_PATH}")
    print(f"   Форма: {df.shape}")
    print("\n   Проверка на NaN:")
    print(df.isnull().sum().sum()) # Должен быть 0

if __name__ == '__main__':
    run_preparation()