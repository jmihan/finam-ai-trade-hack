import pandas as pd
import numpy as np
import re
from rapidfuzz import fuzz
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm
import os
import sys

# --- Блок для корректного импорта ---
# Позволяет запускать скрипт из любой директории
try:
    from config import *
    from utils import load_cache, save_cache
except ImportError:
    # Добавляем корневую директорию проекта в sys.path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import *
    from utils import load_cache, save_cache
# --- Конец блока ---


def run_nlp_feature_generation():
    """
    Основной пайплайн для генерации признаков из новостей.
    Выполняет все шаги: загрузка, сопоставление, извлечение признаков, агрегация и сохранение.
    """
    logging.info("="*50)
    logging.info(" Запуск полного пайплайна генерации NLP признаков...")
    logging.info("="*50)

    # Шаг 1: Загрузка данных
    df_news, unique_tickers = _load_data()

    # Шаг 2: Сопоставление новостей и тикеров
    df_ticker_matches = _match_tickers_to_news(df_news, unique_tickers)

    # Шаг 3: Извлечение количественных признаков
    df_quant_features = _extract_quantitative_features(df_news)

    # Шаг 4: Извлечение признаков из RuBERT
    df_rubert_features = _extract_rubert_features(df_news)

    # Шаг 5: Агрегация и сохранение
    _aggregate_and_save_features(df_news, df_ticker_matches, df_quant_features, df_rubert_features)

    logging.info("\n✅ Пайплайн генерации NLP признаков успешно завершен!")
    logging.info("="*50)


def _load_data():
    """Загружает свечи для получения списка тикеров и сами новости."""
    logging.info("--- Шаг 1/5: Загрузка исходных данных ---")
    try:
        df_candles = pd.read_csv(RAW_CANDLES_PATH)
        unique_tickers = df_candles['ticker'].unique().tolist()
        logging.info(f"Найдено {len(unique_tickers)} уникальных тикеров.")
    except FileNotFoundError:
        logging.error(f"Файл не найден: {RAW_CANDLES_PATH}. Прерывание.")
        sys.exit(1)

    try:
        df_news = pd.read_csv(RAW_NEWS_PATH, index_col=0, parse_dates=['publish_date'])
        df_news.index.name = 'original_news_id'
        df_news.reset_index(inplace=True)
        
        df_news['full_text'] = df_news['title'].fillna('') + ". " + df_news['publication'].fillna('')
        df_news = df_news.rename(columns={'publish_date': 'date'})
        df_news['date'] = df_news['date'].dt.normalize()
        logging.info(f"Загружено {len(df_news)} новостей.")
    except FileNotFoundError:
        logging.error(f"Файл не найден: {RAW_NEWS_PATH}. Прерывание.")
        sys.exit(1)
        
    return df_news, unique_tickers


def _match_tickers_to_news(df_news, unique_tickers, threshold=80):
    """Находит упоминания тикеров в текстах новостей."""

    TICKER_KEYWORDS_MAP = {
        'T': ['тинькофф', 'т-банк', 'tcs group', 'tinkoff', 'т-технологии',
               'ткс', 'ткс холдинг', 'т банк', 'т технологии']}

    TRANSLIT_MAP = str.maketrans({
        # Латиница → Кириллица (визуально или фонетически похожие)
        'a': 'а',  # A → А
        'b': 'в',  # B → В
        'c': 'с',  # C → С
        'd': 'д',  # D → Д
        'e': 'е',  # E → Е
        'f': 'ф',  # F → Ф
        'g': 'г',  # G → Г
        'h': 'н',  # H → Н
        'i': 'и',  # I → И
        'j': 'й',  # J → Й
        'k': 'к',  # K → К
        'l': 'л',  # L → Л
        'm': 'м',  # M → М
        'n': 'н',  # N → Н
        'o': 'о',  # O → О
        'p': 'р',  # P → Р
        'q': 'к',  # Q → К
        'r': 'р',  # R → Р
        's': 'с',  # S → С
        't': 'т',  # T → Т
        'u': 'у',  # U → У
        'v': 'в',  # V → В
        'w': 'в',  # W → В
        'x': 'х',  # X → Х
        'y': 'у',  # Y → У
        'z': 'з',  # Z → З
    })
    
    logging.info("\n--- Шаг 2/5: Сопоставление новостей и тикеров ---")
    df_cache = load_cache(TICKER_MATCH_CACHE_PATH)
    
    processed_ids = set(df_cache['original_news_id']) if not df_cache.empty else set()
    news_to_process = df_news[~df_news['original_news_id'].isin(processed_ids)]

    if news_to_process.empty:
        logging.info("Все сопоставления уже находятся в кэше.")
        return df_cache

    matches_list = df_cache.to_dict('records')

    for _, row in tqdm(news_to_process.iterrows(), total=len(news_to_process), desc="Поиск тикеров"):
        text_lower = row['full_text'].lower()
        found_tickers = []

        for ticker in unique_tickers:

            if ticker in TICKER_KEYWORDS_MAP:
                # Используем специальную логику для проблемных тикеров
                keywords = TICKER_KEYWORDS_MAP[ticker]
                if any(keyword in text_lower for keyword in keywords):
                    found_tickers.append(ticker)
            else:
                ticker_lower = ticker.lower()
                ticker_cyr = ticker_lower.translate(TRANSLIT_MAP)

                # --- 1. Прямое точное совпадение 
                if re.search(r'\b' + re.escape(ticker_lower) + r'\b', text_lower):
                    found_tickers.append(ticker)
                    continue

                # --- 2. Совпадение "по-русски" (транслитерация)
                if re.search(ticker_cyr, text_lower):
                    found_tickers.append(ticker)
                    continue

                # --- 3. Приблизительное совпадение (если текст похож, но не точный)
                if fuzz.partial_ratio(ticker_cyr, text_lower) >= threshold:
                    found_tickers.append(ticker)
                    continue

        matches_list.append({
            'original_news_id': row['original_news_id'],
            'identified_tickers': list(set(found_tickers))
        })

    df_ticker_matches = pd.DataFrame(matches_list)
    save_cache(df_ticker_matches, TICKER_MATCH_CACHE_PATH)
    return df_ticker_matches


def _extract_quantitative_features(df_news):
    """Извлекает простые текстовые метрики."""
    logging.info("\n--- Шаг 3/5: Извлечение количественных признаков ---")
    df_cache = load_cache(QUANT_FEATURES_CACHE_PATH)
    
    processed_ids = set(df_cache['original_news_id']) if not df_cache.empty else set()
    news_to_process = df_news[~df_news['original_news_id'].isin(processed_ids)]

    if news_to_process.empty:
        logging.info("Все количественные признаки уже находятся в кэше.")
        return df_cache

    quant_list = df_cache.to_dict('records')
    for _, row in tqdm(news_to_process.iterrows(), total=len(news_to_process), desc="Расчет метрик"):
        text = str(row['full_text'])
        words = text.split()
        word_count = len(words)
        quant_list.append({
            'original_news_id': row['original_news_id'],
            'char_count': len(text),
            'word_count': word_count,
            'avg_word_len': np.mean([len(w) for w in words]) if word_count > 0 else 0,
            'caps_count': sum(1 for w in words if w.isupper() and len(w) > 1),
        })

    df_quant_features = pd.DataFrame(quant_list)
    save_cache(df_quant_features, QUANT_FEATURES_CACHE_PATH)
    return df_quant_features


def _extract_rubert_features(df_news):
    """Извлекает сентимент и эмбеддинги из одной модели."""
    logging.info("\n--- Шаг 4/5: Извлечение признаков из RuBERT Sentiment ---")
    
    # ВАЖНО: Модель должна быть скачана и лежать в `MODELS_DIR`
    model_checkpoint = 'rubert-tiny-sentiment-balanced'
    
    df_cache = load_cache(RUBERT_FEATURES_CACHE_PATH) # Используем старый кэш для эмбеддингов
    
    processed_ids = set(df_cache['original_news_id']) if not df_cache.empty else set()
    news_to_process = df_news[~df_news['original_news_id'].isin(processed_ids)]

    if news_to_process.empty:
        logging.info("Все RuBERT признаки уже находятся в кэше.")
        return df_cache
    
    logging.info(f"Загрузка модели '{model_checkpoint}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_PATH, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH, local_files_only=True).to(DEVICE)
        model.eval()
    except OSError:
        logging.error(f"Модель не найдена. Пожалуйста, скачайте '{model_checkpoint}' и поместите ее в директорию, указанную в config.py.")
        sys.exit(1)

    rubert_list = df_cache.to_dict('records')
    texts_to_process = news_to_process['full_text'].tolist()
    ids_to_process = news_to_process['original_news_id'].tolist()

    for i in tqdm(range(0, len(texts_to_process), BATCH_SIZE), desc="Обработка новостей"):
        batch_texts = texts_to_process[i:i+BATCH_SIZE]
        batch_ids = ids_to_process[i:i+BATCH_SIZE]

        inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=512).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
            # 1. Получаем вероятности сентимента
            probabilities = torch.sigmoid(outputs.logits).cpu().numpy()
            
            # 2. Получаем эмбеддинги (mean pooling последнего слоя)
            last_hidden_states = outputs.hidden_states[-1]
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
            sum_embeddings = torch.sum(last_hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_pooled = (sum_embeddings / sum_mask).cpu().numpy()

        for j, news_id in enumerate(batch_ids):
            rec = {'original_news_id': news_id, 'embedding': mean_pooled[j]}
            # Добавляем сентимент признаки с понятными именами
            for k, label in model.config.id2label.items():
                rec[f'sentiment_{label}'] = probabilities[j][k]
            rubert_list.append(rec)
        
        if (i // BATCH_SIZE + 1) % SAVE_INTERVAL_NLP == 0:
            save_cache(pd.DataFrame(rubert_list), RUBERT_FEATURES_CACHE_PATH)

    df_rubert_features = pd.DataFrame(rubert_list)
    save_cache(df_rubert_features, RUBERT_FEATURES_CACHE_PATH)
    return df_rubert_features


def _aggregate_and_save_features(df_news, df_tickers, df_quant, df_rubert):
    """Объединяет все признаки, агрегирует и сохраняет."""
    logging.info("\n--- Шаг 5/5: Финальная агрегация и сохранение ---")
    
    # 1. Объединяем df_news с количественными признаками
    df_merged = df_news[['original_news_id', 'date']].merge(df_quant, on='original_news_id', how='left')
    
    # 2. Разворачиваем эмбеддинги в отдельные колонки
    embedding_dim = 0
    if 'embedding' in df_rubert.columns and not df_rubert.empty:
        first_valid_embedding = df_rubert['embedding'].dropna().iloc[0]
        if isinstance(first_valid_embedding, (np.ndarray, list)):
             embedding_dim = len(first_valid_embedding)

    if embedding_dim > 0:
        emb_cols = [f'emb_{i}' for i in range(embedding_dim)]
        emb_df = pd.DataFrame(df_rubert['embedding'].tolist(), index=df_rubert.index, columns=emb_cols)
        df_rubert_expanded = pd.concat([df_rubert.drop('embedding', axis=1), emb_df], axis=1)
        # 3. Объединяем с признаками из RuBERT
        df_merged = df_merged.merge(df_rubert_expanded, on='original_news_id', how='left')

    # 4. Объединяем с найденными тикерами
    df_merged = df_merged.merge(df_tickers, on='original_news_id', how='left')

    # --- ТЕПЕРЬ, В САМОМ КОНЦЕ, ДЕЛАЕМ EXPLODE ---
    df_merged = df_merged.explode('identified_tickers').rename(columns={'identified_tickers': 'ticker'}).dropna(subset=['ticker'])
    
    # Теперь df_merged готов к агрегации, и он был создан с минимальным потреблением памяти.

    # Словарь для агрегации
    agg_dict = {
        'original_news_id': 'count', # Посчитаем как 'num_news'
        'char_count': 'mean',
        'word_count': 'mean',
        'avg_word_len': 'mean',
        'caps_count': 'sum'
    }
    # Добавляем сентимент и эмбеддинги (все усредняем)
    for col in df_merged.columns:
        if col.startswith('sentiment_') or col.startswith('emb_'):
            agg_dict[col] = 'mean'
            
    # Агрегируем
    df_agg = df_merged.groupby(['date', 'ticker']).agg(agg_dict).reset_index()
    df_agg.rename(columns={'original_news_id': 'num_news'}, inplace=True)

    save_cache(df_agg, PROCESSED_NEWS_FEATURES_PATH)
    logging.info(f"Финальные NLP признаки сохранены в {PROCESSED_NEWS_FEATURES_PATH}")
    logging.info(f"Итоговая форма: {df_agg.shape}")
    logging.info("Пример финальных данных:\n" + df_agg.head().to_string())


if __name__ == '__main__':
    run_nlp_feature_generation()