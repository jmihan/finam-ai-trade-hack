import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys

# --- Блок для корректного импорта конфига ---
# Добавляем корневую директорию проекта в sys.path
try:
    from config import *
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import *
# --- Конец блока ---

def generate_features_for_group(group: pd.DataFrame) -> pd.DataFrame:
    """
    Генерирует признаки для одной группы данных (одного тикера), используя только pandas и numpy.
    """
    df = group.copy()
    
    # --- 1. Признаки доходности и волатильности ---
    df['log_return_1d'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_20d'] = df['log_return_1d'].rolling(window=20).std()
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close_prev = np.abs(df['high'] - df['close'].shift(1))
    low_close_prev = np.abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df['atr_14'] = true_range.ewm(alpha=1/14, adjust=False).mean()

    # --- 2. Индикаторы моментума ---
    # RSI (Relative Strength Index)
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10) # + epsilon для избежания деления на ноль
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema_12 - ema_26
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']

    # --- 3. Скользящие средние и осцилляторы ---
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['dist_from_sma_50'] = (df['close'] - df['sma_50']) / df['sma_50']

    # Bollinger Bands
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    df['bb_upper'] = sma_20 + (std_20 * 2)
    df['bb_lower'] = sma_20 - (std_20 * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_20 # Нормализованная ширина

    # --- 4. Признаки на основе объема ---
    df['volume_change_1d'] = df['volume'].pct_change()
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_vs_sma20'] = df['volume'] / df['volume_sma_20']

    # --- 5. Лаговые признаки ---
    for lag in [1, 2, 3, 5, 10]:
        df[f'log_return_lag_{lag}'] = df['log_return_1d'].shift(lag)

    # --- 6. Очистка промежуточных колонок ---
    df.drop(columns=['macd_line', 'macd_signal'], inplace=True, errors='ignore')

    return df


def run_ts_feature_generation():
    """
    Основной пайплайн генерации признаков для временных рядов.
    Загружает все данные, генерирует признаки для каждого тикера и сохраняет результат.
    """
    print(" Запуск генерации признаков временных рядов...")

    # 1. Загрузка данных
    print("1/4: Загрузка данных о свечах...")
    try:
        df = pd.read_csv(RAW_CANDLES_PATH, parse_dates=['begin'])
    except FileNotFoundError as e:
        print(f"ОШИБКА: Не удалось загрузить файл данных: {e}. Проверьте путь в config.py.")
        return

    df = df.sort_values(by=['ticker', 'begin']).reset_index(drop=True)
    print(f" Данные загружены. Форма: {df.shape}")

    # 2. Генерация календарных признаков
    print("2/4: Генерация календарных признаков...")
    df['day_of_week'] = df['begin'].dt.dayofweek
    df['week_of_year'] = df['begin'].dt.isocalendar().week.astype(int)
    df['month'] = df['begin'].dt.month
    df['day_of_year'] = df['begin'].dt.dayofyear
    print(" Календарные признаки созданы.")

    # 3. Генерация признаков TA по группам
    print("3/4: Генерация признаков технического анализа для каждого тикера...")
    
    initial_columns = set(df.columns)

    processed_groups = []
    # Используем `tqdm` для отслеживания прогресса
    for ticker, group in tqdm(df.groupby('ticker'), desc="Обработка тикеров"):
        processed_groups.append(generate_features_for_group(group))

    df = pd.concat(processed_groups)
    
    new_ta_cols = list(set(df.columns) - initial_columns)
    print(f" Сгенерировано {len(new_ta_cols)} новых признаков.")
    
    # 4. Обработка пропусков и сохранение
    print("4/4: Обработка пропусков и сохранение результата...")
    
    # После расчетов в начале каждого ряда появляются NaN. Заполняем их внутри групп
    all_feature_cols = new_ta_cols + ['day_of_week', 'week_of_year', 'month', 'day_of_year']
    
    # Важно: применяем transform внутри groupby, чтобы избежать утечек данных между тикерами
    df[all_feature_cols] = df.groupby('ticker', group_keys=False)[all_feature_cols].apply(lambda x: x.ffill().bfill())
    
    # Заменяем оставшиеся NaN и бесконечности, которые могли возникнуть при делении
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    os.makedirs(os.path.dirname(PROCESSED_TS_FEATURES_PATH), exist_ok=True)
    df.to_parquet(PROCESSED_TS_FEATURES_PATH, index=False)
    
    print(f"\n Готово! Признаки сохранены в: {PROCESSED_TS_FEATURES_PATH}")
    print(f"   Финальная форма данных с признаками: {df.shape}")
    print(f"   Количество NaN в финальном датасете: {df.isnull().sum().sum()}")


if __name__ == '__main__':
    run_ts_feature_generation()