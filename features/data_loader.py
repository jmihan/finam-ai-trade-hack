import pandas as pd
import logging
import re
from config import TRAIN_CANDLES_PATH, TEST_NEWS_PATH

def load_and_prepare_data():
    logging.info("1. Загрузка и подготовка данных")

    try:
        df_train_candles = pd.read_csv(TRAIN_CANDLES_PATH)
        unique_tickers = df_train_candles['ticker'].unique().tolist()
        logging.info(f"Загружено {len(df_train_candles)} строк из {TRAIN_CANDLES_PATH}.")
        logging.info(f"Обнаружены уникальные тикеры: {unique_tickers[:5]}...")
    except FileNotFoundError:
        logging.error(f"Ошибка: {TRAIN_CANDLES_PATH} не найден. Убедитесь, что файл находится в той же директории.")
        exit()

    try:
        df_news = pd.read_csv(TEST_NEWS_PATH)
        
        if 'Unnamed: 0' in df_news.columns:
            df_news = df_news.rename(columns={'Unnamed: 0': 'original_news_id'})
        else:
            logging.warning("Колонка 'Unnamed: 0' не найдена в test_news.csv. Будет использован индекс DataFrame как 'original_news_id'.")
            df_news['original_news_id'] = df_news.index 
        
        if 'title' in df_news.columns and 'publication' in df_news.columns:
            df_news['full_text'] = df_news['title'] + ". " + df_news['publication']
            df_news = df_news.drop(columns=['publication'])
        elif 'text' not in df_news.columns:
            df_news['full_text'] = df_news['title']
        
        df_news = df_news.rename(columns={'publish_date': 'date'})
        df_news['date'] = pd.to_datetime(df_news['date']).dt.normalize()
        
        logging.info(f"Загружено {len(df_news)} строк из {TEST_NEWS_PATH}.")
        logging.info(df_news.head())
    except FileNotFoundError:
        logging.error(f"Ошибка: {TEST_NEWS_PATH} не найден. Убедитесь, что файл находится в той же директории.")
        exit()

    logging.info("-" * 50)
    return df_news, unique_tickers

def match_tickers_to_news(df_news, unique_tickers):
    logging.info("2. Сопоставление тикеров с новостями (поиск в тексте)")
    from utils import load_cache, save_cache
    from config import TICKER_MATCH_CACHE_FILENAME

    df_ticker_matches = load_cache(TICKER_MATCH_CACHE_FILENAME)
    processed_news_ids_tickers = set(df_ticker_matches['original_news_id'].unique()) if not df_ticker_matches.empty else set()
    ticker_matches_list = df_ticker_matches.to_dict('records')

    news_to_process_tickers = df_news[~df_news['original_news_id'].isin(processed_news_ids_tickers)].copy()

    if not news_to_process_tickers.empty:
        for _, row in tqdm(news_to_process_tickers.iterrows(), total=len(news_to_process_tickers), desc="Сопоставление тикеров"):
            news_id = row['original_news_id']
            news_text = row['full_text'].lower()
            
            identified_tickers = []
            for ticker in unique_tickers:
                if re.search(r'\b' + re.escape(ticker.lower()) + r'\b', news_text):
                    identified_tickers.append(ticker)
            
            ticker_matches_list.append({
                'original_news_id': news_id,
                'identified_tickers': identified_tickers
            })

        df_ticker_matches = pd.DataFrame(ticker_matches_list)
        save_cache(df_ticker_matches, TICKER_MATCH_CACHE_FILENAME)
    else:
        logging.info("Все новости уже сопоставлены с тикерами в кэше.")

    logging.info("Первые 5 строк с сопоставленными тикерами:")
    logging.info(df_ticker_matches.head())
    logging.info("-" * 50)
    return df_ticker_matches
