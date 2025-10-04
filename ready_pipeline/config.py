import os
import torch
import logging
from pathlib import Path

# --- 1. Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 2. Базовые пути проекта ---
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / 'data'
MODELS_DIR = ROOT_DIR / 'models'
ARTIFACTS_DIR = ROOT_DIR / 'artifacts'

# Создаем директории, если они не существуют
MODELS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)
(DATA_DIR / 'raw').mkdir(exist_ok=True)
(DATA_DIR / 'processed').mkdir(exist_ok=True)

# --- 3. Пути к исходным данным ---
RAW_CANDLES_PATH = DATA_DIR / 'raw' / 'candles.csv'
RAW_NEWS_PATH = DATA_DIR / 'raw' / 'news.csv'

# --- 4. Пути к обработанным признакам (промежуточные файлы) ---
PROCESSED_TS_FEATURES_PATH = DATA_DIR / 'processed' / 'features_ts.parquet'
PROCESSED_NEWS_FEATURES_PATH = DATA_DIR / 'processed' / 'features_news.parquet'

# --- 5. Путь к финальному объединенному датасету ---
FINAL_TRAIN_DF_PATH = DATA_DIR / 'processed' / 'final_train_data.parquet'
INFERENCE_DF_PATH = DATA_DIR / 'processed' / 'inference_data.parquet' 

# --- 6. Пути для NLP-кэшей ---
TICKER_MATCH_CACHE_PATH = ARTIFACTS_DIR / 'ticker_matches_cache.parquet'
QUANT_FEATURES_CACHE_PATH = ARTIFACTS_DIR / 'quant_features_cache.parquet'
RUBERT_FEATURES_CACHE_PATH = ARTIFACTS_DIR / 'rubert_features_cache.parquet'

# --- 7. Конфигурация NLP и Torch ---
# ВАЖНО: Модель должна лежать в папке models/rubert-tiny-sentiment-balanced
SENTIMENT_MODEL_PATH = MODELS_DIR / "rubert-tiny-sentiment-balanced"

BATCH_SIZE = 48
SAVE_INTERVAL_NLP = 100

# --- Остальная часть файла без изменений ---
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Используется устройство: {device}")
    if device.type == 'cuda':
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram:.2f} GB")
    return device

DEVICE = get_device()

# --- 8. Гиперпараметры модели PatchTST ---
# Общие
CONTEXT_LENGTH = 128
PREDICTION_LENGTH = 20
PATCH_LENGTH = 16
# Архитектура
D_MODEL = 256
N_HEADS = 16
ENCODER_LAYERS = 4
DROPOUT = 0.3
# Обучение
LEARNING_RATE = 0.00005
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5

# --- 9. Пути для сохранения финальной модели и артефактов ---
PATCHTST_MODEL_PATH = MODELS_DIR / 'patchtst_final_model'
PATCHTST_PREPROCESSOR_PATH = ARTIFACTS_DIR / 'patchtst_preprocessor.pkl'