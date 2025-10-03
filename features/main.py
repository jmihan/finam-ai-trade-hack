import logging
from config import FINAL_FEATURES_FILENAME
from data_loader import load_and_prepare_data, match_tickers_to_news
from feature_extractor import extract_quantitative_features, extract_tinybert_embeddings, extract_emobert_features
from aggregator import combine_and_explode_features, aggregate_daily_ticker_features, save_final_features

def main():
    logging.info("Запуск основного скрипта обработки новостей.")

    # 1. Загрузка и подготовка данных
    df_news, unique_tickers = load_and_prepare_data()

    # 2. Сопоставление тикеров с новостями
    df_ticker_matches = match_tickers_to_news(df_news, unique_tickers)

    # 3. Извлечение количественных признаков
    df_quant_features = extract_quantitative_features(df_news)

    # 4. TinyBERT эмбеддинги
    df_tinybert_embeddings = extract_tinybert_embeddings(df_news)

    # 5. EmoBERT признаки
    df_emobert_features = extract_emobert_features(df_news)

    # 6. Объединение и разворачивание данных
    df_final_features, embedding_dim = combine_and_explode_features(
        df_news, df_ticker_matches, df_quant_features, df_tinybert_embeddings, df_emobert_features
    )

    # 7. Агрегация признаков по датам и тикерам
    daily_ticker_features = aggregate_daily_ticker_features(df_final_features, embedding_dim)

    # 8. Сохранение результатов
    save_final_features(daily_ticker_features)

    logging.info("\nСкрипт завершен.")

if __name__ == "__main__":
    main()
