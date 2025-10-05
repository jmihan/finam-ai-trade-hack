import os

# --- Пути к данным и моделям ---
# Корень проекта
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Папка с данными
DATA_DIR = os.path.join(ROOT_DIR, 'data')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'features.parquet')

# Папка для сохранения моделей
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'catboost_model.cbm')


# --- Параметры модели CatBoost ---
# loss_function='RMSE' - стандартная метрика для регрессии
# eval_metric='MAE' - может быть более интерпретируемой метрикой для оценки
# verbose=100 - выводить информацию об обучении каждые 100 итераций
# early_stopping_rounds=50 - если качество на валидации не улучшается 50 итераций, остановить обучение
CATBOOST_PARAMS = {
    'iterations': 2000,
    'learning_rate': 0.05,
    'depth': 8,
    'loss_function': 'RMSE',
    'eval_metric': 'MAE',
    'random_seed': 42,
    'verbose': 100,
    'task_type': 'GPU',
    'devices': '0',
    'early_stopping_rounds': 50
}

# --- Параметры валидации ---
# Название колонки с датой/временем
# TODO: Заменить на реальное название колонки с датой
TIME_COLUMN = 'event_date' 
# Количество фолдов для временной кросс-валидации
N_SPLITS = 5

# --- Параметры модели LSTM ---
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_model.pth')

# Длина последовательности (сколько прошлых дней используем для предсказания)
SEQUENCE_LENGTH = 30 
# Количество эпох обучения
EPOCHS = 50
# Размер батча
BATCH_SIZE = 32

# Параметры архитектуры
LSTM_UNITS = 64
DENSE_UNITS = 32

# Количество фолдов для временной кросс-валидации
N_SPLITS_LSTM = 5 

# --- Параметры модели Transformer ---
TRANSFORMER_MODEL_PATH = os.path.join(MODEL_DIR, 'transformer_model.pth')

# d_model: Размерность эмбеддингов внутри трансформера. Должна быть делимой на n_heads.
D_MODEL = 128
# n_heads: Количество "голов" в механизме multi-head attention.
N_HEADS = 8
# num_encoder_layers: Количество слоев кодировщика.
NUM_ENCODER_LAYERS = 3
# dropout: Вероятность dropout.
DROPOUT = 0.1

# --- Параметры признаков ---
TARGET_COLUMN = 'target_price' # TODO: Заменить на реальное название целевой переменной
# Сюда можно будет добавить список категориальных признаков, если они появятся
CATEGORICAL_FEATURES = []
# Признаки, которые не нужно использовать в обучении
DROP_FEATURES = ['event_date'] # TODO: Заменить на реальные признаки (например, дата)

# --- Пути к данным ---
# ...
RAW_CANDLES_PATH = os.path.join(DATA_DIR, 'raw', 'train_candles.csv')
RAW_NEWS_PATH = os.path.join(DATA_DIR, 'raw', 'train_news.csv')

# Пути к обработанным признакам
PROCESSED_TS_FEATURES_PATH = os.path.join(DATA_DIR, 'processed', 'features_ts.parquet')
PROCESSED_NEWS_FEATURES_PATH = os.path.join(DATA_DIR, 'processed', 'features_news.parquet')

# --- Пути к обработанным данным ---
FINAL_TRAIN_DF_PATH = os.path.join(DATA_DIR, 'processed', 'final_train_data.parquet')

# --- Параметры Моделей PatchTST ---
# Общие
CONTEXT_LENGTH = 128         # Сколько прошлых дней смотрим
PREDICTION_LENGTH = 20       # На сколько дней вперед предсказываем (траектория)
PATCH_LENGTH = 16            # Размер одного "патча"
N_SPLITS_CV = 5              # Количество фолдов для CV

# Для регрессии
REG_D_MODEL = 128
REG_N_HEADS = 8
REG_ENCODER_LAYERS = 3
REG_DROPOUT = 0.1

# Для вероятности (модель может быть поменьше)
PROB_D_MODEL = 64
PROB_N_HEADS = 4
PROB_ENCODER_LAYERS = 2
PROB_DROPOUT = 0.2


ARTIFACTS_DIR = os.path.join(ROOT_DIR, 'artifacts') # Для сохранения скейлеров и препроцессоров

# --- Пути к сырым данным ---
RAW_TRAIN_CANDLES_PATH = os.path.join(DATA_DIR, 'raw', 'train_candles.csv')
RAW_PUBLIC_TEST_CANDLES_PATH = os.path.join(DATA_DIR, 'raw', 'public_test_candles.csv')
RAW_PRIVATE_TEST_CANDLES_PATH = os.path.join(DATA_DIR, 'raw', 'private_test_candles.csv')
RAW_NEWS_PATH = os.path.join(DATA_DIR, 'raw', 'train_news.csv') # Пока train, потом объединим

# --- Пути к артефактам моделей ---
# Регрессионная модель
REG_MODEL_PATH = os.path.join(MODEL_DIR, 'patchtst_regression')
REG_PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, 'preprocessor_reg.pkl')

# Вероятностная модель
PROB_MODEL_PATH = os.path.join(MODEL_DIR, 'patchtst_probability')
PROB_PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, 'preprocessor_prob.pkl')