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


# --- Параметры признаков ---
TARGET_COLUMN = 'target_price' # TODO: Заменить на реальное название целевой переменной
# Сюда можно будет добавить список категориальных признаков, если они появятся
CATEGORICAL_FEATURES = []
# Признаки, которые не нужно использовать в обучении
DROP_FEATURES = ['event_date'] # TODO: Заменить на реальные признаки (например, дата)