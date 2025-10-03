# scripts/predict.py

import pandas as pd
import numpy as np
import torch
import joblib
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import PatchTSTForPrediction

# --- Импорты ---
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
# --- Конец блока ---

def run_prediction():
    print("🚀 Запуск пайплайна предсказаний...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Используемое устройство: {device.upper()} ---")

    # 1. Загрузка артефактов
    print("1/5: Загрузка обученных моделей, препроцессоров и скейлеров...")
    try:
        model_reg = PatchTSTForPrediction.from_pretrained(REG_MODEL_PATH).to(device)
        model_prob = PatchTSTForPrediction.from_pretrained(PROB_MODEL_PATH).to(device)
        preprocessor_reg = joblib.load(REG_PREPROCESSOR_PATH)
        preprocessor_prob = joblib.load(PROB_PREPROCESSOR_PATH)
        # ЗАГРУЖАЕМ СКЕЙЛЕРЫ
        scalers_reg = joblib.load(REG_PREPROCESSOR_PATH.replace('.pkl', '_scalers.pkl'))
        scalers_prob = joblib.load(PROB_PREPROCESSOR_PATH.replace('.pkl', '_scalers.pkl'))
    except Exception as e:
        print(f"ОШИБКА: Загрузка артефактов не удалась. Убедитесь, что train.py отработал. {e}")
        return
    model_reg.eval(); model_prob.eval()

    # 2. Загрузка данных
    print("2/5: Загрузка и подготовка данных для предсказания...")
    train_df = pd.read_csv(RAW_TRAIN_CANDLES_PATH, parse_dates=['begin'])
    test_df = pd.concat([
        pd.read_csv(RAW_PUBLIC_TEST_CANDLES_PATH, parse_dates=['begin']),
        pd.read_csv(RAW_PRIVATE_TEST_CANDLES_PATH, parse_dates=['begin'])
    ], ignore_index=True)
    
    full_candles_df = pd.concat([train_df, test_df], ignore_index=True).sort_values(by=['ticker', 'begin'])
    features_df = pd.read_parquet(PROCESSED_TS_FEATURES_PATH).set_index(['ticker', 'begin'])
    
    data_to_predict = pd.merge(
        features_df,
        full_candles_df[['ticker', 'begin', 'close']].rename(columns={'close': 'original_close'}), # Переименовываем, чтобы не было конфликта
        on=['ticker', 'begin'],
        how='left' # left join, чтобы сохранить все строки из features_df
    )
    data_to_predict = data_to_predict.sort_values(by=['ticker', 'begin']).reset_index(drop=True)

    # 3. Генерация предсказаний
    print("3/5: Генерация предсказаний для каждой тестовой даты...")
    all_predictions = []
    
    num_context_features_reg = len(preprocessor_reg.control_columns)
    num_context_features_prob = len(preprocessor_prob.control_columns)

    for ticker in tqdm(test_df['ticker'].unique(), desc="Предсказание по тикерам"):
        # Получаем скейлеры для текущего тикера
        scaler_key = (ticker,)
        scaler_reg = scalers_reg.get(scaler_key)
        scaler_prob = scalers_prob.get(scaler_key)
        if not scaler_reg or not scaler_prob:
            print(f"  ! Внимание: Скейлеры для тикера {ticker} не найдены. Пропуск.")
            continue

        for date in test_df[test_df['ticker'] == ticker]['begin']:
            history = data_to_predict[(data_to_predict['ticker'] == ticker) & (data_to_predict['begin'] < date)].copy()
            if len(history) < CONTEXT_LENGTH: continue

            for i in range(1, PREDICTION_LENGTH + 1):
                history[f'target_return_{i}d'], history[f'target_grew_{i}d'] = 0.0, 0.0

            proc_reg = preprocessor_reg.preprocess(history)
            proc_prob = preprocessor_prob.preprocess(history)
            
            ctx_reg_vals = proc_reg.drop(columns=['ticker', 'begin']).iloc[-CONTEXT_LENGTH:, :num_context_features_reg].values
            ctx_prob_vals = proc_prob.drop(columns=['ticker', 'begin']).iloc[-CONTEXT_LENGTH:, :num_context_features_prob].values
            
            ctx_reg = torch.tensor(ctx_reg_vals, dtype=torch.float32).unsqueeze(0).to(device)
            ctx_prob = torch.tensor(ctx_prob_vals, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                preds_reg_scaled = model_reg(past_values=ctx_reg).prediction_outputs
                preds_prob_scaled = model_prob(past_values=ctx_prob).prediction_outputs

            # --- ФИНАЛЬНОЕ ИСПРАВЛЕНИЕ ---
            # Выход модели имеет форму (batch_size, prediction_length, num_targets)
            # В нашем случае это (1, 20, 20). Нам нужна диагональ.
            # Но скорее всего, num_targets=1, и форма (1, 20, 1). Убираем последнюю размерность.
            
            # Преобразуем в numpy
            preds_reg_np = preds_reg_scaled.cpu().numpy()
            preds_prob_np = preds_prob_scaled.cpu().numpy()

            # Убираем batch-размерность (первую)
            preds_reg_np = preds_reg_np.squeeze(0)
            preds_prob_np = preds_prob_np.squeeze(0)
            
            # Если последняя размерность = 1, убираем и ее. Форма станет (20,)
            if preds_reg_np.shape[-1] == 1:
                preds_reg_np = preds_reg_np.squeeze(-1)
            
            if preds_prob_np.shape[-1] == 1:
                preds_prob_np = preds_prob_np.squeeze(-1)

            # Теперь форма должна быть (20,) или (20, 20). Скейлер ожидает (n_samples, n_features)
            # В нашем случае это (1, 20) для одного предсказания.
            if preds_reg_np.ndim == 1:
                preds_reg_np = preds_reg_np.reshape(1, -1) # (20,) -> (1, 20)
            if preds_prob_np.ndim == 1:
                preds_prob_np = preds_prob_np.reshape(1, -1) # (20,) -> (1, 20)
            
            preds_reg_unscaled = scaler_reg.inverse_transform(preds_reg_np)
            preds_prob_unscaled = scaler_prob.inverse_transform(preds_prob_np)
            
            record = {'ticker': ticker, 'begin': date}
            record['pred_return_1d'] = preds_reg_unscaled[0, 0]
            record['pred_return_20d'] = np.sum(preds_reg_unscaled[0, :])
            record['pred_prob_up_1d'] = np.clip(preds_prob_unscaled[0, 0], 0, 1)
            record['pred_prob_up_20d'] = np.clip(preds_prob_unscaled[0, 19], 0, 1)
            all_predictions.append(record)

    submission_df = pd.DataFrame(all_predictions)
    
    # 4. Сохранение
    print("4/5: Сохранение submission.csv...")
    final_cols = ['ticker', 'begin', 'pred_return_1d', 'pred_return_20d', 'pred_prob_up_1d', 'pred_prob_up_20d']
    submission_df.reindex(columns=final_cols).fillna(0.5).to_csv("submission.csv", index=False)
    print("  ✓ Submission сохранен.")
    if not submission_df.empty: print(submission_df.head().to_string())

    # 5. Визуализация
    print("5/5: Создание визуализаций...")
    if not submission_df.empty:
        plot_predictions(submission_df, train_df, test_df, submission_df['ticker'].iloc[0])

def plot_predictions(submission_df, train_df, test_df, ticker):
    ticker_preds = submission_df[submission_df['ticker'] == ticker].sort_values(by='begin')
    ticker_train = train_df[train_df['ticker'] == ticker]
    ticker_test = test_df[test_df['ticker'] == ticker]
    if ticker_preds.empty or ticker_train.empty or ticker_test.empty: return
    full_history = pd.concat([ticker_train, ticker_test]).sort_values(by='begin')
    full_history['close_prev'] = full_history['close'].shift(1)
    plot_df = pd.merge(ticker_preds, full_history[['begin', 'close_prev']], on='begin', how='left').ffill()
    plot_df['pred_price'] = plot_df['close_prev'] * (1 + plot_df['pred_return_1d'])
    plt.figure(figsize=(15, 7))
    history_plot = ticker_train.tail(90)
    plt.plot(history_plot['begin'], history_plot['close'], label=f'История цены {ticker}')
    plt.plot(ticker_test['begin'], ticker_test['close'], label=f'Реальная цена (тест)', color='gray', linestyle='--')
    plt.plot(plot_df['begin'], plot_df['pred_price'], label='Предсказанная цена', linestyle='-', marker='o', color='red', markersize=4)
    plt.title(f'Прогноз движения цены для {ticker}'); plt.xlabel('Дата'); plt.ylabel('Цена')
    plt.legend(); plt.grid(True); plt.savefig(f'forecast_price_{ticker}.png'); plt.close()

if __name__ == '__main__':
    run_prediction()