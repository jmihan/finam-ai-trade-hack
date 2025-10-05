import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# --- Блок для исправления импорта при локальном запуске ---
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем наши модули
from src.transformer_model import TimeSeriesTransformer
from src.trainer import train_model
from src.data_processing import create_sequences
from src.config import *

def run_transformer_training_cv():
    """
    Основной пайплайн для обучения и оценки модели Трансформера 
    с использованием временной кросс-валидации.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Используемое устройство: {device.upper()} ---")

    # 1. Загрузка данных (используем ту же заглушку)
    print("Загрузка данных...")
    num_rows = 1000
    data = pd.DataFrame({
        'feature_1': range(num_rows), 'feature_2': [x * 0.5 for x in range(num_rows)],
        TIME_COLUMN: pd.to_datetime(pd.date_range('2023-01-01', periods=num_rows)),
        TARGET_COLUMN: [x * 2 + 3 + np.random.randn() * 5 for x in range(num_rows)]
    })
    data = data.sort_values(by=TIME_COLUMN).reset_index(drop=True)

    # 2. Приведение рядов к стационарности (ключевой шаг)
    print("Приведение рядов к стационарности...")
    numeric_cols = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]
    
    # Сохраняем исходные цены ДО дифференцирования для восстановления прогнозов
    original_prices = data[TARGET_COLUMN].copy()
    data[numeric_cols] = data[numeric_cols].diff()
    
    data = data.dropna().reset_index(drop=True)
    original_prices = original_prices.iloc[1:].reset_index(drop=True)

    features = [col for col in numeric_cols if col != TARGET_COLUMN]
    X_df = data[features]
    y_s = data[TARGET_COLUMN] # Теперь это target_diff

    # 3. Кросс-валидация
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_LSTM)
    oof_preds, oof_true, fold_scores = [], [], []
    last_fold_vars = {}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_df)):
        print("-" * 40)
        print(f"Фолд {fold + 1}/{N_SPLITS_LSTM}")
        
        X_train_df, X_val_df = X_df.iloc[train_idx], X_df.iloc[val_idx]
        y_train_s, y_val_s = y_s.iloc[train_idx], y_s.iloc[val_idx]

        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train_df)
        y_train_scaled = target_scaler.fit_transform(y_train_s.values.reshape(-1, 1))
        
        train_data_for_seq_df = pd.DataFrame(
            np.hstack((X_train_scaled, y_train_scaled)),
            columns=features + [TARGET_COLUMN]
        )
        X_train_seq, y_train_seq = create_sequences(train_data_for_seq_df, SEQUENCE_LENGTH, TARGET_COLUMN)
        
        if len(X_train_seq) < BATCH_SIZE:
            print("  Пропуск фолда: недостаточно данных для создания даже одного батча.")
            continue

        es_split_idx = int(len(X_train_seq) * 0.85)
        X_train, y_train = X_train_seq[:es_split_idx], y_train_seq[:es_split_idx]
        X_es, y_es = X_train_seq[es_split_idx:], y_train_seq[es_split_idx:]

        train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        val_dataset = TensorDataset(torch.from_numpy(X_es).float(), torch.from_numpy(y_es).float())
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = TimeSeriesTransformer(
            num_features=X_train.shape[2],
            d_model=D_MODEL,
            n_head=N_HEADS,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            dropout=DROPOUT
        ).to(device)
        
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model_fold_path = TRANSFORMER_MODEL_PATH.replace('.pth', f'_fold_{fold}.pth')
        
        trained_model = train_model(model, train_loader, val_loader, loss_function, optimizer, device, EPOCHS, model_fold_path)

        # Оценка на валидационном фолде
        print(f"  Оценка на валидационном фолде...")
        X_val_scaled = feature_scaler.transform(X_val_df)
        tail_data_scaled = np.hstack((
            feature_scaler.transform(X_train_df.tail(SEQUENCE_LENGTH)),
            target_scaler.transform(y_train_s.tail(SEQUENCE_LENGTH).values.reshape(-1, 1))
        ))
        combined_val_data_scaled = np.vstack((tail_data_scaled, np.hstack((X_val_scaled, np.zeros((len(X_val_scaled), 1))))))
        combined_val_df = pd.DataFrame(combined_val_data_scaled, columns=features + [TARGET_COLUMN])
        X_val_seq, _ = create_sequences(combined_val_df, SEQUENCE_LENGTH, TARGET_COLUMN)
        
        if len(X_val_seq) > 0:
            X_val_tensor = torch.from_numpy(X_val_seq).float().to(device)
            trained_model.eval()
            with torch.no_grad():
                val_preds_scaled = trained_model(X_val_tensor)
            
            predicted_diffs = target_scaler.inverse_transform(val_preds_scaled.cpu().numpy())
            last_known_prices = original_prices.iloc[val_idx - 1].values
            
            restored_preds = last_known_prices[0] + np.cumsum(predicted_diffs.flatten())
            true_values = original_prices.iloc[val_idx].values

            oof_preds.extend(restored_preds)
            oof_true.extend(true_values)

            fold_mae = mean_absolute_error(true_values, restored_preds)
            fold_scores.append(fold_mae)
            print(f"  MAE на фолде {fold + 1}: {fold_mae:.4f}")

            if fold == N_SPLITS_LSTM - 1:
                last_fold_vars = {
                    'X_val_tensor': X_val_tensor,
                    'target_scaler': target_scaler,
                    'val_idx': val_idx
                }

    # 4. Итоговая оценка по кросс-валидации
    if oof_true:
        overall_mae = mean_absolute_error(oof_true, oof_preds)
        print("\n" + "=" * 40)
        print("Кросс-валидация Transformer завершена.")
        print(f"Средний MAE по фолдам: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
        print(f"Общий OOF MAE: {overall_mae:.4f}")
        print("=" * 40)

    # 5. Обучение и оценка финальной модели
    print("\nОбучение финальной модели Transformer на всех доступных данных...")
    final_feature_scaler = MinMaxScaler()
    final_target_scaler = MinMaxScaler()
    X_scaled = final_feature_scaler.fit_transform(X_df)
    y_scaled = final_target_scaler.fit_transform(y_s.values.reshape(-1, 1))
    
    final_data_for_seq_df = pd.DataFrame(np.hstack((X_scaled, y_scaled)), columns=features + [TARGET_COLUMN])
    X_final_seq, y_final_seq = create_sequences(final_data_for_seq_df, SEQUENCE_LENGTH, TARGET_COLUMN)

    if len(X_final_seq) > 0:
        final_es_split_idx = int(len(X_final_seq) * 0.85)
        X_train, y_train = X_final_seq[:final_es_split_idx], y_final_seq[:final_es_split_idx]
        X_val, y_val = X_final_seq[final_es_split_idx:], y_final_seq[final_es_split_idx:]
        
        train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        final_model = TimeSeriesTransformer(
            num_features=X_final_seq.shape[2], d_model=D_MODEL, n_head=N_HEADS,
            num_encoder_layers=NUM_ENCODER_LAYERS, dropout=DROPOUT
        ).to(device)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001)
        
        print("Начинаем обучение финальной модели...")
        trained_final_model = train_model(
            final_model, train_loader, val_loader, loss_function, optimizer, device, EPOCHS, TRANSFORMER_MODEL_PATH
        )
        print(f"Финальная модель Transformer обучена и сохранена в {TRANSFORMER_MODEL_PATH}")

        # Финальная оценка
        print("\nФинальная оценка лучшей модели на самых свежих данных...")
        if last_fold_vars:
            trained_final_model.eval()
            with torch.no_grad():
                preds_scaled = trained_final_model(last_fold_vars['X_val_tensor'])
            
            predicted_diffs = last_fold_vars['target_scaler'].inverse_transform(preds_scaled.cpu().numpy())
            last_known_price = original_prices.iloc[last_fold_vars['val_idx'][0] - 1]
            restored_preds = last_known_price + np.cumsum(predicted_diffs.flatten())
            true_values = original_prices.iloc[last_fold_vars['val_idx']].values
            
            final_mae_on_test = mean_absolute_error(true_values, restored_preds)
            
            print("-" * 30)
            print(f"ФИНАЛЬНАЯ MAE НА ТЕСТЕ (PyTorch Transformer): {final_mae_on_test:.4f}")
            print("-" * 30)

if __name__ == '__main__':
    run_transformer_training_cv()