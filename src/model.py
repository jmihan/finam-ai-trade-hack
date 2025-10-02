import os
import catboost as cb
import pandas as pd

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
        print("Начинаем обучение модели CatBoost...")
        
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_features,
            use_best_model=True, # Важно: сохраняем лучшую итерацию модели
            plot=False
        )
        
        print("Обучение завершено.")
        print(f"Лучший результат на валидации (MAE): {self.model.get_best_score()['validation']['MAE']:.4f}")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Получение предсказаний для новых данных.
        """
        return self.model.predict(X)

    def save(self, path: str):
        """
        Сохранение обученной модели в файл.
        """
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        print(f"Модель успешно сохранена в: {path}")

    def load(self, path: str):
        """
        Загрузка модели из файла.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл модели не найден по пути: {path}")
            
        self.model.load_model(path)
        print(f"Модель успешно загружена из: {path}")