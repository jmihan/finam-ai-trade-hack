import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# --- Блок для исправления импорта ---
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# --- Конец блока ---

from src.lstm_model import PyTorchLSTM
from src.trainer import train_model
from src.data_processing import create_sequences
from src.config import *

def run_pytorch_lstm_training_cv():
    # 1. Проверка GPU и установка device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Используемое устройство: {device.upper()} ---")

    # 2. Загрузка данных
    print("Загрузка данных...")
    # ... (код загрузки данных точно такой же, как раньше) ...
    num_rows = 1000
    data = pd.DataFrame({
        'feature_1': range(num_rows), 'feature_2': [x*0.5 for x in range(num_rows)],
        TIME_COLUMN: pd.to_datetime(pd.date_range('2023-01-01', periods=num_rows)),
        TARGET_COLUMN: [x*2 + 3 + np.random.randn()*5 for x in range(num_rows)]
    })
    # data = data.sort_values(by=TIME_COLUMN).reset_index(drop=True)
    # features = [col for col in data.columns if col not in [TARGET_COLUMN, TIME_COLUMN]]
    # X_df, y_s = data[features], data[TARGET_COLUMN]

    data = data.sort_values(by=TIME_COLUMN).reset_index(drop=True)

    # 1. Вычисляем разницу (изменение) цены
    DIFFERENCED_TARGET = f'{TARGET_COLUMN}_diff'
    data[DIFFERENCED_TARGET] = data[TARGET_COLUMN].diff()

    # 2. Удаляем первую строку, где diff равен NaN
    data = data.dropna().reset_index(drop=True)

    # 3. Теперь наша цель - это не абсолютная цена, а ее изменение
    features = [col for col in data.columns if col not in [TARGET_COLUMN, DIFFERENCED_TARGET, TIME_COLUMN]]
    X_df = data[features]
    y_s = data[DIFFERENCED_TARGET]

    # 3. Кросс-валидация
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_LSTM)
    oof_preds, oof_true, fold_scores = [], [], []

    # Инициализируем словарь здесь
    last_fold_vars = {} 

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_df)):
        print("-" * 40)
        print(f"Фолд {fold + 1}/{N_SPLITS_LSTM}")
        X_train_df, X_val_df = X_df.iloc[train_idx], X_df.iloc[val_idx]
        y_train_s, y_val_s = y_s.iloc[train_idx], y_s.iloc[val_idx]

        # Масштабирование
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train_df)
        y_train_scaled = target_scaler.fit_transform(y_train_s.values.reshape(-1, 1))
        
        # Создание последовательностей (используем старую функцию, она работает с numpy)
        train_data_for_seq = np.hstack((X_train_scaled, y_train_scaled))
        train_data_for_seq_df = pd.DataFrame(train_data_for_seq, columns=features + [TARGET_COLUMN])
        X_train_seq, y_train_seq = create_sequences(train_data_for_seq_df, SEQUENCE_LENGTH, TARGET_COLUMN)
        
        if len(X_train_seq) < 1: continue

        # Разделение на train/validation для early stopping
        es_split_idx = int(len(X_train_seq) * 0.85)
        X_train, y_train = X_train_seq[:es_split_idx], y_train_seq[:es_split_idx]
        X_es, y_es = X_train_seq[es_split_idx:], y_train_seq[es_split_idx:]

        # Создание PyTorch DataLoader'ов
        train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        val_dataset = TensorDataset(torch.from_numpy(X_es).float(), torch.from_numpy(y_es).float())
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Обучение
        num_features = X_train.shape[2]
        model = PyTorchLSTM(num_features=num_features, hidden_units=LSTM_UNITS).to(device)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model_fold_path = LSTM_MODEL_PATH.replace('.pth', f'_fold_{fold}.pth')
        
        trained_model = train_model(model, train_loader, val_loader, loss_function, optimizer, device, EPOCHS, model_fold_path)

        # --- БЛОК ОЦЕНКИ МОДЕЛИ ---
        # -------------------------
        print(f"  Оценка на валидационном фолде...")
        
        # Готовим валидационные данные для предсказания
        X_val_scaled = feature_scaler.transform(X_val_df)
        
        # Для создания последовательностей нам нужен "хвост" от обучающих данных
        tail_data_scaled = np.hstack((
            feature_scaler.transform(X_train_df.tail(SEQUENCE_LENGTH)),
            target_scaler.transform(y_train_s.tail(SEQUENCE_LENGTH).values.reshape(-1, 1))
        ))

        # Соединяем хвост и валидационные данные
        combined_val_data_scaled = np.vstack((
            tail_data_scaled,
            np.hstack((X_val_scaled, np.zeros((len(X_val_scaled), 1)))) # y здесь не важен
        ))
        
        combined_val_df = pd.DataFrame(combined_val_data_scaled, columns=features + [TARGET_COLUMN])
        X_val_seq, _ = create_sequences(combined_val_df, SEQUENCE_LENGTH, TARGET_COLUMN)
        
        if len(X_val_seq) > 0:
            X_val_tensor = torch.from_numpy(X_val_seq).float().to(device)

            # Делаем предсказания
            trained_model.eval()
            with torch.no_grad():
                val_preds_scaled = trained_model(X_val_tensor)
            
            # Обратное масштабирование
            # 1. "Разжимаем" предсказанные изменения
            predicted_diffs = target_scaler.inverse_transform(val_preds_scaled.cpu().numpy())

            # 2. Получаем последние известные цены перед началом валидационного периода
            # data[TARGET_COLUMN].iloc[val_idx.start - 1] - это цена в последний день train-фолда
            last_known_prices = data[TARGET_COLUMN].iloc[val_idx - 1].values

            # 3. Восстанавливаем прогноз цены: Цена(t) = Цена(t-1) + Предсказанный_Diff(t)
            # Мы используем cumsum() для последовательного добавления изменений
            restored_preds = last_known_prices[0] + np.cumsum(predicted_diffs.flatten())

            # 4. Получаем настоящие цены для сравнения
            true_values = data[TARGET_COLUMN].iloc[val_idx].values

            oof_preds.extend(restored_preds)
            oof_true.extend(true_values)

            fold_mae = mean_absolute_error(true_values, restored_preds)
            fold_scores.append(fold_mae)
            print(f"  MAE на фолде {fold + 1}: {fold_mae:.4f}")

            # Сохраняем данные последнего фолда для финальной оценки
            if fold == N_SPLITS_LSTM - 1:
                last_fold_vars = {
                    'X_val_tensor': X_val_tensor,
                    'y_val_s': y_val_s,
                    'target_scaler': target_scaler
                }


    # --- ОБУЧЕНИЕ И ОЦЕНКА ФИНАЛЬНОЙ МОДЕЛИ ---
    # --------------------------------------------
    print("\nОбучение финальной модели PyTorch LSTM на всех доступных данных...")

    # Масштабируем все данные
    final_feature_scaler = MinMaxScaler()
    final_target_scaler = MinMaxScaler()
    X_scaled = final_feature_scaler.fit_transform(X_df)
    y_scaled = final_target_scaler.fit_transform(y_s.values.reshape(-1, 1))

    # Создаем последовательности
    final_data_for_seq = np.hstack((X_scaled, y_scaled))
    final_data_for_seq_df = pd.DataFrame(final_data_for_seq, columns=features + [TARGET_COLUMN])
    X_final_seq, y_final_seq = create_sequences(final_data_for_seq_df, SEQUENCE_LENGTH, TARGET_COLUMN)

    if len(X_final_seq) > 0:
        # Разделяем на train/val для early stopping
        final_es_split_idx = int(len(X_final_seq) * 0.85)
        X_train, y_train = X_final_seq[:final_es_split_idx], y_final_seq[:final_es_split_idx]
        X_val, y_val = X_final_seq[final_es_split_idx:], y_final_seq[final_es_split_idx:]
        
        # Создаем DataLoader'ы
        train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Обучаем финальную модель
        num_features = X_final_seq.shape[2]
        final_model = PyTorchLSTM(num_features=num_features, hidden_units=LSTM_UNITS).to(device)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001)
        
        print("Начинаем обучение финальной модели...")
        trained_final_model = train_model(
            final_model, train_loader, val_loader, loss_function, optimizer, device, EPOCHS, LSTM_MODEL_PATH
        )
        print(f"Финальная модель PyTorch обучена и сохранена в {LSTM_MODEL_PATH}")

        # Финальная оценка на данных последнего фолда
        print("\nФинальная оценка лучшей модели на самых свежих данных...")

        # Используем `val_idx` из последнего фолда, чтобы точно знать индексы тестовых данных
        # Переменная `val_idx` доступна после цикла, так как она была определена в его последней итерации
        if 'last_fold_vars' in locals() and last_fold_vars:
            trained_final_model.eval()
            with torch.no_grad():
                # 1. Получаем предсказания (это предсказанные масштабированные diffs)
                preds_scaled = trained_final_model(last_fold_vars['X_val_tensor'])

            # 2. "Разжимаем" предсказанные изменения
            predicted_diffs = last_fold_vars['target_scaler'].inverse_transform(preds_scaled.cpu().numpy())

            # 3. Находим последнюю известную АБСОЛЮТНУЮ цену перед тестовым периодом
            last_known_price = data[TARGET_COLUMN].iloc[val_idx[0] - 1]

            # 4. Восстанавливаем предсказания абсолютных цен
            # np.cumsum() последовательно прибавляет изменения к начальной точке
            restored_preds = last_known_price + np.cumsum(predicted_diffs.flatten())

            # 5. Получаем настоящие АБСОЛЮТНЫЕ цены для сравнения
            true_values = data[TARGET_COLUMN].iloc[val_idx].values

            # 6. Считаем MAE между восстановленными и настоящими ценами
            final_mae_on_test = mean_absolute_error(true_values, restored_preds)

            print("-" * 30)
            print(f"ФИНАЛЬНАЯ MAE НА ТЕСТЕ (PyTorch LSTM): {final_mae_on_test:.4f}")
            print("-" * 30)
        else:
            print("Не удалось провести финальную оценку, не найдены данные последнего фолда.")

if __name__ == '__main__':
    run_pytorch_lstm_training_cv()