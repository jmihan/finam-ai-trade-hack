import pandas as pd
import numpy as np
import logging
from config import FINAL_FEATURES_FILENAME

def combine_and_explode_features(df_news, df_ticker_matches, df_quant_features, df_tinybert_embeddings, df_emobert_features):
    logging.info("6. Объединение всех признаков")

    df_final_features = df_news.merge(df_ticker_matches, on='original_news_id', how='left')
    df_final_features = df_final_features.explode('identified_tickers')
    df_final_features = df_final_features.rename(columns={'identified_tickers': 'ticker'})
    df_final_features = df_final_features.dropna(subset=['ticker'])

    df_final_features = df_final_features.merge(df_quant_features, on='original_news_id', how='left')

    embedding_dim = 0
    if not df_tinybert_embeddings.empty and 'tinybert_embedding' in df_tinybert_embeddings.columns:
        if not df_tinybert_embeddings['tinybert_embedding'].isnull().all():
            first_embedding = df_tinybert_embeddings['tinybert_embedding'].dropna().iloc[0]
            if isinstance(first_embedding, np.ndarray) and first_embedding.size > 0:
                embedding_dim = first_embedding.shape[0]
            elif isinstance(first_embedding, list) and len(first_embedding) > 0:
                embedding_dim = len(first_embedding)

    if embedding_dim > 0:
        embedding_cols = [f'tinybert_{i}' for i in range(embedding_dim)]
        tinybert_embeddings_expanded = df_tinybert_embeddings['tinybert_embedding'].apply(pd.Series)
        tinybert_embeddings_expanded.columns = embedding_cols
        
        tinybert_df_processed = pd.concat([df_tinybert_embeddings.drop(columns=['tinybert_embedding']), tinybert_embeddings_expanded], axis=1)
        df_final_features = df_final_features.merge(tinybert_df_processed, on='original_news_id', how='left')
    else:
        logging.warning("Предупреждение: TinyBERT эмбеддинги не были извлечены или имеют нулевую размерность. Они не будут добавлены.")

    df_final_features = df_final_features.merge(df_emobert_features, on='original_news_id', how='left')
    df_final_features = df_final_features.drop(columns=['full_text', 'title'], errors='ignore')

    logging.info("\nФинальный датасет признаков (первые 5 строк):")
    logging.info(df_final_features.head())
    logging.info(f"Размерность финального датасета: {df_final_features.shape}")
    logging.info("-" * 50)
    return df_final_features, embedding_dim


def aggregate_daily_ticker_features(df_final_features, embedding_dim):
    logging.info("7. Агрегация признаков по дням и тикерам")

    if df_final_features.empty:
        logging.warning("Не осталось новостей для агрегации после обработки и связывания с тикерами.")
        all_possible_emobert_cols = [f'emobert_emotion_{i}' for i in range(6)]
        all_possible_tinybert_cols = [f'tinybert_{i}' for i in range(embedding_dim)] if embedding_dim > 0 else []

        daily_ticker_features = pd.DataFrame(columns=[
            'date', 'ticker', 'num_news', 'char_count_mean', 'word_count_mean',
            'avg_word_len_mean', 'caps_count_sum', 'repeated_words_count_sum',
            'special_char_count_sum'
        ] + all_possible_tinybert_cols + all_possible_emobert_cols)
        
        for col in ['char_count', 'word_count', 'avg_word_len']:
            daily_ticker_features[f'{col}_mean'] = pd.Series(dtype=float)
        for col in ['caps_count', 'repeated_words_count', 'special_char_count']:
            daily_ticker_features[f'{col}_sum'] = pd.Series(dtype=float)
        
    else:
        agg_funcs = {
            'char_count': 'mean',
            'word_count': 'mean',
            'avg_word_len': 'mean',
            'caps_count': 'sum',
            'repeated_words_count': 'sum',
            'special_char_count': 'sum',
        }

        emobert_cols = [col for col in df_final_features.columns if col.startswith('emobert_')]
        for col in emobert_cols:
            agg_funcs[col] = 'mean'
        
        tinybert_cols = [col for col in df_final_features.columns if col.startswith('tinybert_')]
        for col in tinybert_cols:
            agg_funcs[col] = 'mean'

        if not agg_funcs and 'original_news_id' not in df_final_features.columns:
            logging.warning("Нет колонок для агрегации, кроме 'date' и 'ticker'. Будет агрегировано только количество новостей.")
            daily_ticker_features = df_final_features.groupby(['date', 'ticker']).size().reset_index(name='num_news')
        else:
            final_agg_funcs = {}
            for col, func in agg_funcs.items():
                if col in df_final_features.columns:
                    final_agg_funcs[f'{col}_{func}'] = pd.NamedAgg(column=col, aggfunc=func)
            
            final_agg_funcs['num_news'] = pd.NamedAgg(column='original_news_id', aggfunc='count')

            daily_ticker_features = df_final_features.groupby(['date', 'ticker']).agg(**final_agg_funcs).reset_index()

    logging.info("\nАгрегированные признаки по дням и тикерам (первые 5 строк):")
    logging.info(daily_ticker_features.head())
    logging.info(f"Размерность агрегированного датасета: {daily_ticker_features.shape}")
    logging.info("-" * 50)
    return daily_ticker_features

def save_final_features(daily_ticker_features):
    logging.info("8. Сохранение результатов")
    daily_ticker_features.to_parquet(FINAL_FEATURES_FILENAME, index=False)
    logging.info(f"Финальные агрегированные признаки сохранены в '{FINAL_FEATURES_FILENAME}'")
