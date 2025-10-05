import pandas as pd
import pandas_ta as ta
import numpy as np
from tqdm import tqdm

# --- Блок для исправления импорта при локальном запуске ---
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# --- Конец блока ---

# Обновляем пути в config.py, если нужно
from src.config import * 

def generate_features():
    """
    Загружает train, public_test, private_test свечи, объединяет их,
    генерирует признаки технического анализа для каждого тикера
    и сохраняет результат в parquet-файл.
    """
    print("🚀 Запуск генерации признаков временных рядов...")

    # 1. Загрузка ВСЕХ данных о свечах
    print("1/5: Загрузка всех данных свечей (train, public, private)...")
    
    # Определим пути. Убедитесь, что они верны или добавьте их в config.py
    # Пример путей, если данные лежат в data/raw/
    base_path = '../data/raw/' if os.path.exists('../data/raw/') else 'data/raw/'
    train_candles_path = os.path.join(base_path, 'train_candles.csv')
    public_test_candles_path = os.path.join(base_path, 'public_test_candles.csv')
    private_test_candles_path = os.path.join(base_path, 'private_test_candles.csv')

    try:
        train_df = pd.read_csv(train_candles_path, parse_dates=['begin'])
        public_df = pd.read_csv(public_test_candles_path, parse_dates=['begin'])
        private_df = pd.read_csv(private_test_candles_path, parse_dates=['begin'])
    except FileNotFoundError as e:
        print(f"ОШИБКА: Не удалось загрузить файл данных: {e}. Проверьте пути.")
        return

    # Объединяем все данные для сквозного расчета признаков
    df = pd.concat([train_df, public_df, private_df], ignore_index=True)
    
    # Сортировка обязательна для корректного расчета индикаторов
    df = df.sort_values(by=['ticker', 'begin']).reset_index(drop=True)
    print(f"  ✓ Все данные загружены и объединены. Финальная форма: {df.shape}")

    # 2. Генерация календарных признаков
    print("2/5: Генерация календарных признаков...")
    df['day_of_week'] = df['begin'].dt.dayofweek
    df['week_of_year'] = df['begin'].dt.isocalendar().week.astype(int)
    df['month'] = df['begin'].dt.month
    print("  ✓ Календарные признаки созданы.")

    # 3. Генерация кастомных признаков на основе OHLCV
    print("3/5: Генерация кастомных OHLCV признаков...")
    df['day_range_norm'] = (df['high'] - df['low']) / df['close']
    df['body_size_norm'] = (df['close'] - df['open']) / df['open']
    print("  ✓ Кастомные признаки созданы.")

    # 4. Генерация признаков Технического Анализа (TA)
    print("4/5: Генерация TA признаков (RSI, MACD, BBands, Stochastic)...")
        
    # Сохраняем изначальные колонки, чтобы потом найти новые
    initial_columns = set(df.columns)

    # Создаем пустой список для сбора обработанных групп
    processed_groups = []
        
    # tqdm для отслеживания прогресса по тикерам
    for ticker, group in tqdm(df.groupby('ticker'), desc="Обработка тикеров"):
        # Копируем, чтобы избежать SettingWithCopyWarning
        ticker_df = group.copy()
            
        # Расчет индикаторов для текущего тикера
        # `append=True` модифицирует ticker_df на месте
        ticker_df.ta.rsi(length=14, append=True)
        ticker_df.ta.macd(append=True)
        ticker_df.ta.bbands(length=20, append=True)
        ticker_df.ta.stoch(append=True)
            
        processed_groups.append(ticker_df)

    # Объединяем результаты всех тикеров обратно в один DataFrame
    df = pd.concat(processed_groups)

    # Динамически находим все новые колонки, которые были добавлены pandas_ta
    final_columns = set(df.columns)
    ta_cols = list(final_columns - initial_columns)

    print(f"  ✓ TA признаки созданы. Новые колонки: {ta_cols}")
        
    # Удаляем лишние колонки MACD, оставляя только гистограмму
    cols_to_drop = [col for col in ta_cols if col.startswith('MACD_') or col.startswith('MACDs_')]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    # Обновляем список ta_cols после удаления
    ta_cols = [col for col in ta_cols if col not in cols_to_drop]

    # Применяем forward fill и backward fill ВНУТРИ каждой группы тикеров
    print("  > Обработка NaN значений в TA признаках...")
    df[ta_cols] = df.groupby('ticker')[ta_cols].transform(lambda x: x.ffill().bfill())

    print("  ✓ TA признаки обработаны.")

    # 5. Сохранение результата
    print("5/5: Сохранение обработанных признаков...")
    
    # Выбираем только ключевые колонки и новые признаки
    feature_columns = [
        'ticker', 'begin',
        'open', 'close', 'high', 'low', 'volume', # Оставляем исходные OHLCV
        'day_of_week', 'week_of_year', 'month',
        'day_range_norm', 'body_size_norm'
    ] + [col for col in ta_cols if col in df.columns]

    final_df = df[feature_columns].copy()
    
    # Создаем директорию, если она не существует
    os.makedirs(os.path.dirname(PROCESSED_TS_FEATURES_PATH), exist_ok=True)

    final_df.to_parquet(PROCESSED_TS_FEATURES_PATH, index=False)
    
    print(f"\n✅ Готово! Признаки сохранены в: {PROCESSED_TS_FEATURES_PATH}")
    print(f"   Финальная форма данных с признаками: {final_df.shape}")
    print("\n   Проверка на NaN после обработки:")
    print(final_df.isnull().sum())
    print("\n   Пример сгенерированных признаков:")
    print(final_df.tail().to_string())


if __name__ == '__main__':
    generate_features()