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