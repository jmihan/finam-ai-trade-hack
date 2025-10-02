import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# Импортируем наши собственные модули
from src.model import CatBoostModel
from src.config import (
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    CATBOOST_PARAMS,
    TARGET_COLUMN,
    DROP_FEATURES,
    CATEGORICAL_FEATURES,
    TIME_COLUMN,
    N_SPLITS
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

    # Убедимся, что данные отсортированы по времени перед сплитом
    data = data.sort_values(by=TIME_COLUMN).reset_index(drop=True)
    
    features = [col for col in data.columns if col not in [TARGET_COLUMN, TIME_COLUMN] + DROP_FEATURES]
    X = data[features]
    y = data[TARGET_COLUMN]

    print(f"Данные загружены. {len(data)} строк. Используемые признаки ({len(features)}): {features}")

    # 2. Кросс-валидация
    # --------------------
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    oof_preds = []  # Out-of-fold predictions (предсказания на валидационных фолдах)
    oof_true = []   # Истинные значения для тех же фолдов
    fold_scores = []

    print(f"\nНачинаем кросс-валидацию на {N_SPLITS} фолдах...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print("-" * 40)
        print(f"Фолд {fold + 1}/{N_SPLITS}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        print(f"  Обучение на {len(X_train)} сэмплах, валидация на {len(X_val)} сэмплах.")
        
        # ВАЖНО: создаем новый экземпляр модели для каждого фолда!
        model = CatBoostModel(params=CATBOOST_PARAMS)
        
        # CatBoost требует, чтобы eval_set был отдельным, поэтому мы разделим наш val_set
        # для ранней остановки. Мы используем тот же принцип, что и раньше.
        # Возьмем последние 15% из обучающей выборки фолда как eval_set.
        es_split_idx = int(len(X_train) * 0.85)
        X_train_fold, X_es_fold = X_train.iloc[:es_split_idx], X_train.iloc[es_split_idx:]
        y_train_fold, y_es_fold = y_train.iloc[:es_split_idx], y_train.iloc[es_split_idx:]
        
        model.fit(
            X_train_fold, y_train_fold,
            X_es_fold, y_es_fold, # Используем хвост обучающей выборки для early stopping
            cat_features=CATEGORICAL_FEATURES
        )
        
        # Делаем предсказания на валидационной части фолда
        val_preds = model.predict(X_val)
        
        oof_preds.extend(val_preds)
        oof_true.extend(y_val.values)
        
        fold_mae = mean_absolute_error(y_val, val_preds)
        fold_scores.append(fold_mae)
        print(f"  MAE на фолде {fold + 1}: {fold_mae:.4f}")

    # 3. Итоговая оценка
    # --------------------
    overall_mae = mean_absolute_error(oof_true, oof_preds)
    print("\n" + "=" * 40)
    print("Кросс-валидация завершена.")
    print(f"Средний MAE по фолдам: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"Общий OOF MAE: {overall_mae:.4f}")
    print("=" * 40)

    # 4. Обучение финальной модели на всех данных
    # -------------------------------------------
    print("\nОбучение финальной модели на всех доступных данных...")
    
    # Для финальной модели также используем early stopping на последних данных
    final_es_split_idx = int(len(X) * 0.85)
    X_train_final, X_es_final = X.iloc[:final_es_split_idx], X.iloc[final_es_split_idx:]
    y_train_final, y_es_final = y.iloc[:final_es_split_idx], y.iloc[final_es_split_idx:]

    final_model = CatBoostModel(params=CATBOOST_PARAMS)
    final_model.fit(
        X_train_final, y_train_final,
        X_es_final, y_es_final,
        cat_features=CATEGORICAL_FEATURES
    )
    
    # Сохраняем модель, обученную на всех данных
    final_model.save(MODEL_PATH)


if __name__ == '__main__':
    run_training()