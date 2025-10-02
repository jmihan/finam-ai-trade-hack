import pandas as pd
import re
import time
from llm_api_mock import fake_call_llm # Импортируем заглушку

# Загружаем датафрейм новостей, созданный в первом скрипте
try:
    df_news = pd.read_pickle('df_news.pkl')
    print("Датафрейм новостей 'df_news.pkl' успешно загружен.")
except FileNotFoundError:
    print("Ошибка: 'df_news.pkl' не найден. Убедитесь, что '1_prepare_fake_news_dataset.py' был запущен первым.")
    exit()

print("3. Генерация признаков с помощью fake LLM")

def process_news_with_mock_llm(row):
    news_text = row['text']
    features = {}

    # 1. тональность (Sentiment)
    sentiment_prompt = "Оцени тональность этой новости для финансового рынка по шкале от -1 (очень негативно) до 1 (очень позитивно). Ответ дай только числом."
    sentiment_response = fake_call_llm(sentiment_prompt, news_text)
    try:
        features['sentiment'] = float(re.search(r'[-+]?\d*\.?\d+', sentiment_response).group()) if sentiment_response else None
    except (ValueError, TypeError, AttributeError):
        features['sentiment'] = None

    time.sleep(0.05) # имитация задержки api

    # 2. оценка влияния (Impact Score)
    impact_prompt = "Оцени потенциальное влияние этой новости на цену финансового инструмента по шкале от 0 (нет влияния) до 10 (очень сильное влияние). Ответ дай только числом."
    impact_response = fake_call_llm(impact_prompt, news_text)
    try:
        features['impact'] = int(re.search(r'\d+', impact_response).group()) if impact_response else None
    except (ValueError, TypeError, AttributeError):
        features['impact'] = None

    time.sleep(0.05) # имитация задержки api

    # 3. извлечение терминов
    entities_prompt = "Назови ключевые экономические термины в этой новости (например, инфляция, процентная ставка, ВВП). Перечисли через запятую."
    entities_response = fake_call_llm(entities_prompt, news_text)
    features['entities'] = [e.strip() for e in entities_response.split(',')] if entities_response and entities_response != "Нет данных" else []

    return features

try:
    from tqdm.auto import tqdm
    tqdm.pandas()
    df_llm_features = df_news.progress_apply(process_news_with_mock_llm, axis=1, result_type='expand')
except ImportError:
    print("tqdm не установлен, обработка без прогресс-бара. Для прогресс-бара: pip install tqdm")
    df_llm_features = df_news.apply(process_news_with_mock_llm, axis=1, result_type='expand')


# объединяем новые признаки с исходным DataFrame
df_processed_news = pd.concat([df_news, df_llm_features], axis=1)

print("\nДатасет новостей с сгенерированными признаками (первые 5 строк):")
print(df_processed_news.head())
print("-" * 50)

# Сохраняем обработанный датафрейм для следующего скрипта
df_processed_news.to_pickle('df_processed_news.pkl')
print("Обработанный датафрейм новостей сохранен как 'df_processed_news.pkl'")
