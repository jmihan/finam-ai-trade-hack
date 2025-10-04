import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
import os
import torch
import catboost as cb # Для использования CatBoostModel

# Импортируем из config
from config import (
    TRAIN_CANDLES_PATH, FINAL_FEATURES_FILENAME, TARGET_COLUMN,
    BOOSTING_MODELS_DIR, CATBOOST_MODEL_PATH, XGBOOST_MODEL_PATH, LIGHTGBM_MODEL_PATH
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Базовый класс для CatBoost, который предоставил капитан
class CatBoostModel:
    """
    Класс-обертка для модели CatBoostRegressor.
    Предоставляет методы для обучения, предсказания, сохранения и загрузки модели.
    """
    def __init__(self, params: dict):
        """
        Инициализация модели с заданными параметрами.
        """
        self.params = params
        self.model = cb.CatBoostRegressor(**self.params)

    def fit(self, 
            X_train: pd.DataFrame, 
            y_train: pd.Series, 
            X_val: pd.DataFrame, 
            y_val: pd.Series,
            cat_features: list = None):
        """
        Обучение модели.
        Использует валидационную выборку для ранней остановки.
        """
        logging.info("Начинаем обучение модели CatBoost...")
        
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_features,
            use_best_model=True, # Важно: сохраняем лучшую итерацию модели
            plot=False,
            verbose=100 # Вывод каждые 100 итераций
        )
        
        logging.info("Обучение завершено.")
        logging.info(f"Лучший результат на валидации (MAE): {self.model.get_best_score()['validation']['MAE']:.4f}")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Получение предсказаний для новых данных.
        """
        return pd.Series(self.model.predict(X), index=X.index)

    def save(self, path: str):
        """
        Сохранение обученной модели в файл.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        logging.info(f"Модель CatBoost успешно сохранена в: {path}")

    def load(self, path: str):
        """
        Загрузка модели из файла.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл модели CatBoost не найден по пути: {path}")
            
        self.model.load_model(path)
        logging.info(f"Модель CatBoost успешно загружена из: {path}")


def prepare_boosting_data():
    """
    Загружает и объединяет свечные данные с агрегированными новостными признаками.
    """
    logging.info("Подготовка данных для бустинговых моделей...")

    try:
        df_candles = pd.read_csv(TRAIN_CANDLES_PATH)
        df_candles['begin'] = pd.to_datetime(df_candles['begin']).dt.normalize()
        df_candles = df_candles.rename(columns={'begin': 'date'})
        logging.info(f"Загружено {len(df_candles)} строк свечных данных.")
    except FileNotFoundError:
        logging.error(f"Ошибка: {TRAIN_CANDLES_PATH} не найден. Убедитесь, что файл находится в той же директории.")
        return None, None

    try:
        df_news_features = pd.read_parquet(FINAL_FEATURES_FILENAME)
        df_news_features['date'] = pd.to_datetime(df_news_features['date']).dt.normalize()
        logging.info(f"Загружено {len(df_news_features)} строк агрегированных новостных признаков.")
    except FileNotFoundError:
        logging.error(f"Ошибка: {FINAL_FEATURES_FILENAME} не найден. Сначала запустите основной скрипт 'main.py'.")
        return None, None
    except Exception as e:
        logging.error(f"Ошибка при загрузке '{FINAL_FEATURES_FILENAME}': {e}")
        return None, None

    # Объединяем данные
    df_merged = pd.merge(df_candles, df_news_features, on=['date', 'ticker'], how='left')
    logging.info(f"Данные объединены. Размерность: {df_merged.shape}")
    
    # Заполняем пропуски в новостных признаках нулями, если новостей не было
    news_feature_cols = [col for col in df_merged.columns if col not in df_candles.columns and col not in ['date', 'ticker', TARGET_COLUMN]]
    if news_feature_cols:
        df_merged[news_feature_cols] = df_merged[news_feature_cols].fillna(0)
        logging.info(f"Пропущенные значения в новостных признаках заполнены нулями.")
    else:
        logging.warning("Не найдено новостных признаков для заполнения пропусков.")

    # Удаляем строки, где нет целевой переменной (target_return_1d)
    df_merged = df_merged.dropna(subset=[TARGET_COLUMN])
    logging.info(f"Удалены строки с пропущенной целевой переменной. Итоговая размерность: {df_merged.shape}")
    logging.info("\nОбъединенный датасет (первые 5 строк):")
    logging.info(df_merged.head())
    logging.info("-" * 50)
    return df_merged


def train_boosting_models(df_data: pd.DataFrame):
    """
    Обучает и тестирует бустинговые модели (XGBoost, LightGBM, CatBoost).
    """
    if df_data is None or df_data.empty:
        logging.error("Нет данных для обучения бустинговых моделей.")
        return

    # Подготовка данных для обучения
    features = [col for col in df_data.columns if col not in ['date', 'ticker', TARGET_COLUMN]]
    X = df_data[features]
    y = df_data[TARGET_COLUMN]

    # Разделение на тренировочную и валидационную выборки (80/20)
    # Используем StratifiedSplit, если таргет категориальный, здесь регрессия - обычный split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False) # shuffle=False для временных рядов

    logging.info(f"Размер тренировочной выборки: {X_train.shape}")
    logging.info(f"Размер валидационной выборки: {X_val.shape}")
    logging.info("-" * 50)

    # --- CatBoost ---
    logging.info("Обучение CatBoost модели...")
    cat_params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'loss_function': 'MAE', # Mean Absolute Error
        'eval_metric': 'MAE',
        'random_seed': 42,
        'verbose': 0,
        'early_stopping_rounds': 50,
        'task_type': "GPU" if torch.cuda.is_available() else "CPU" # Используем GPU, если доступно
    }
    cat_model = CatBoostModel(cat_params)
    cat_model.fit(X_train, y_train, X_val, y_val)
    cat_preds = cat_model.predict(X_val)
    cat_mae = mean_absolute_error(y_val, cat_preds)
    logging.info(f"CatBoost MAE на валидации: {cat_mae:.4f}")
    cat_model.save(CATBOOST_MODEL_PATH)
    logging.info("-" * 50)

    # --- LightGBM ---
    logging.info("Обучение LightGBM модели...")
    lgbm_params = {
        'objective': 'mae',
        'metric': 'mae',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'num_leaves': 31,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        'device': 'gpu' if torch.cuda.is_available() else 'cpu' # Используем GPU, если доступно
    }
    lgbm_model = lgb.LGBMRegressor(**lgbm_params)
    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgbm_preds = lgbm_model.predict(X_val)
    lgbm_mae = mean_absolute_error(y_val, lgbm_preds)
    logging.info(f"LightGBM MAE на валидации: {lgbm_mae:.4f}")
    os.makedirs(os.path.dirname(LIGHTGBM_MODEL_PATH), exist_ok=True)
    lgbm_model.booster_.save_model(LIGHTGBM_MODEL_PATH)
    logging.info(f"Модель LightGBM успешно сохранена в: {LIGHTGBM_MODEL_PATH}")
    logging.info("-" * 50)

    # --- XGBoost ---
    logging.info("Обучение XGBoost модели...")
    xgb_params = {
        'objective': 'reg:absoluteerror', # Mean Absolute Error
        'eval_metric': 'mae',
        'n_estimators': 1000, # Увеличим число estimators, так как используем раннюю остановку
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist',
        'enable_categorical': True, # Для совместимости, если есть категориальные признаки
    }
    xgb_model = xgb.XGBRegressor(**xgb_params)

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50, # Ранняя остановка по 50 итерациям без улучшения
        verbose=False # Отключаем подробный вывод во время обучения
    )

    # best_iteration будет установлен автоматически после ранней остановки
    if xgb_model.best_iteration is not None:
        logging.info(f"XGBoost: Лучшая итерация: {xgb_model.best_iteration}")
        # Предсказание с использованием лучшей модели
        # predict() в XGBoost 2.x с early_stopping_rounds автоматически использует best_iteration.
        # Явное указание iteration_range оставлено для надежности, но часто необязательно.
        xgb_preds = xgb_model.predict(X_val, iteration_range=(0, xgb_model.best_iteration + 1))
    else:
        logging.warning("XGBoost: Лучшая итерация не определена, предсказание со всей моделью.")
        xgb_preds = xgb_model.predict(X_val)

    xgb_mae = mean_absolute_error(y_val, xgb_preds)
    logging.info(f"XGBoost MAE на валидации: {xgb_mae:.4f}")
    os.makedirs(os.path.dirname(XGBOOST_MODEL_PATH), exist_ok=True)
    # Сохраняем модель. best_iteration будет использоваться по умолчанию при сохранении,
    # если ранняя остановка была активирована.
    xgb_model.save_model(XGBOOST_MODEL_PATH) 
    logging.info(f"Модель XGBoost успешно сохранена в: {XGBOOST_MODEL_PATH}")
    logging.info("-" * 50) 
    
    logging.info("Все бустинговые модели обучены и сохранены.")
