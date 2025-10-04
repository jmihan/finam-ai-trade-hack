import os
import torch
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- НАСТРОЙКИ ФАЙЛОВ ДАННЫХ ---
TRAIN_CANDLES_PATH = 'train_candles.csv'
TEST_NEWS_PATH = 'test_news.csv'
FINAL_FEATURES_FILENAME = 'features_news.parquet'
TARGET_COLUMN = 'target_return_1d' # Добавь эту строку - название твоей целевой переменной

# --- НАСТРОЙКИ КЭШИРОВАНИЯ ---
TICKER_MATCH_CACHE_FILENAME = 'ticker_matches_cache.parquet'
QUANT_FEATURES_CACHE_FILENAME = 'quant_features_cache.parquet'
TINYBERT_EMBEDDINGS_CACHE_FILENAME = 'tinybert_embeddings_cache.parquet'
EMOBERT_FEATURES_CACHE_FILENAME = 'emobert_features_cache.parquet'

SAVE_INTERVAL = 500 # Сохранять кэш каждые N батчей для моделей

# --- НАСТРОЙКИ МОДЕЛЕЙ (ЛОКАЛЬНЫЕ ПУТИ) ---
MODELS_DIR = "C:/Users/nikit/Desktop/models" # Базовая директория для моделей
TINYBERT_MODEL_NAME = os.path.join(MODELS_DIR, "bert-tiny")
EMOBERT_MODEL_NAME = os.path.join(MODELS_DIR, "emobert")

# --- НАСТРОЙКИ БУСТИНГ-МОДЕЛЕЙ --- # НОВЫЙ БЛОК
BOOSTING_MODELS_DIR = "C:/Users/nikit/Desktop/models/boosting_models"
CATBOOST_MODEL_PATH = os.path.join(BOOSTING_MODELS_DIR, "catboost_model.cbm")
XGBOOST_MODEL_PATH = os.path.join(BOOSTING_MODELS_DIR, "xgboost_model.json") # XGBoost сохраняется в JSON
LIGHTGBM_MODEL_PATH = os.path.join(BOOSTING_MODELS_DIR, "lightgbm_model.txt") # LightGBM сохраняется в TXT

# --- НАСТРОЙКИ УСТРОЙСТВА И БАТЧА ---
BATCH_SIZE = 4 # ОЧЕНЬ ВАЖНО: Подберите это значение .
              # Начните с 4, если получаете CUDA OOM, уменьшите до 2 или 1.
              # Если все равно не работает, переключите модель на CPU.

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Используется устройство: {device}")
    if device.type == 'cuda':
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram:.2f} GB")
        # Изменено: если VRAM меньше 3GB, используем CPU. CatBoost/LightGBM/XGBoost также могут использовать GPU,
        # но при недостатке VRAM лучше явно указать CPU
        if vram < 3: 
            logging.warning("Обнаружен GPU, но VRAM < 3GB. Производительность может быть нестабильной. Модели будут использовать CPU.")
            device = torch.device("cpu") # Принудительно переключаем на CPU
            logging.info(f"Переключено на CPU из-за недостатка VRAM. Используется устройство: {device}")
    return device

DEVICE = get_device()
