import pandas as pd

# Загружаем датафрейм с признаками LLM, созданный в третьем скрипте
try:
    df_processed_news = pd.read_pickle('df_processed_news.pkl')
    print("Датафрейм обработанных новостей 'df_processed_news.pkl' успешно загружен.")
except FileNotFoundError:
    print("Ошибка: 'df_processed_news.pkl' не найден. Убедитесь, что '3_generate_features_with_mock_llm.py' был запущен первым.")
    exit()

print("4. Агрегация признаков по дням")

# проверка колонки 'date'
# df_processed_news['date'] уже должен быть datetime из 1_prepare_fake_news_dataset.py
# но на всякий случай можно повторить или убедиться:
if not pd.api.types.is_datetime64_any_dtype(df_processed_news['date']):
    df_processed_news['date'] = pd.to_datetime(df_processed_news['date'])


# агрегация по дням
daily_features = df_processed_news.groupby('date').agg(
    avg_sentiment=('sentiment', 'mean'),
    max_impact=('impact', 'max'),
    num_positive_news=('sentiment', lambda x: (x > 0).sum()), # количество позитивных
    num_negative_news=('sentiment', lambda x: (x < 0).sum()), # количество негативных
    all_entities=('entities', lambda x: list(set(sum(x, [])))) # объединяем списки и берем уникальные
).reset_index()

print("\nАгрегированные признаки по дням (первые 5 строк):")
print(daily_features.head())
print("-" * 50)

# Сохраняем агрегированный датафрейм
daily_features.to_pickle('daily_features.pkl')
print("Агрегированные дневные признаки сохранены как 'daily_features.pkl'")

print("\nДемо скрипт завершен.")
print("df_processed_news содержит признаки для каждой новости.")
print("daily_features содержит агрегированные признаки по дням (готов к передаче в Поток 3).")
