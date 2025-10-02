import pandas as pd
from sklearn.model_selection import train_test_split # Временно, для каркаса

# Импортируем наши собственные модули
from src.model import CatBoostModel
from src.config import (
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    CATBOOST_PARAMS,
    TARGET_COLUMN,
    DROP_FEATURES,
    CATEGORICAL_FEATURES
)

def run_training():
    """
    Основная функция для запуска пайплайна обучения.
    """
    print("Запуск пайплайна обучения...")
    
    # 1. Загрузка данных
    # -------------------
    # TODO: Заменить на реальный код загрузки, когда Инженеры А и Б подготовят данные
    # Пока создадим фейковый датафрейм для проверки работы каркаса
    print(f"Загрузка данных из {PROCESSED_DATA_PATH}...")
    try:
        # data = pd.read_parquet(PROCESSED_DATA_PATH)
        # --- ЗАГЛУШКА ---
        data = pd.DataFrame({
            'feature_1': range(100),
            'feature_2': [x*0.5 for x in range(100)],
            'event_date': pd.to_datetime(pd.date_range('2024-01-01', periods=100)),
            TARGET_COLUMN: [x*2 + 3 for x in range(100)]
        })
        # --- КОНЕЦ ЗАГЛУШКИ ---
    except FileNotFoundError:
        print("Ошибка: Файл с обработанными данными не найден. Завершение работы.")
        return

    print("Данные успешно загружены.")

    # 2. Подготовка данных для модели
    # --------------------------------
    # ВАЖНО: для временных рядов нельзя использовать случайное разбиение!
    # Здесь должна быть логика разделения по дате (Walk-Forward Validation).
    # Для каркаса мы временно используем простое разделение.
    
    # TODO: Реализовать правильное разделение по дате
    train_data, val_data = train_test_split(data, test_size=0.2, shuffle=False) # shuffle=False - важно!

    features = [col for col in train_data.columns if col not in [TARGET_COLUMN] + DROP_FEATURES]
    
    X_train = train_data[features]
    y_train = train_data[TARGET_COLUMN]
    
    X_val = val_data[features]
    y_val = val_data[TARGET_COLUMN]

    print(f"Данные разделены. Размер обучающей выборки: {len(X_train)}, валидационной: {len(X_val)}")
    print(f"Используемые признаки ({len(features)}): {features}")

    # 3. Обучение модели
    # --------------------
    model = CatBoostModel(params=CATBOOST_PARAMS)
    
    model.fit(
        X_train, y_train,
        X_val, y_val,
        cat_features=CATEGORICAL_FEATURES
    )

    # 4. Сохранение модели
    # ---------------------
    model.save(MODEL_PATH)


if __name__ == '__main__':
    run_training()