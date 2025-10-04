import pandas as pd
import numpy as np
import re
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from tqdm.auto import tqdm
from config import DEVICE, BATCH_SIZE, TINYBERT_MODEL_NAME, EMOBERT_MODEL_NAME, SAVE_INTERVAL
from utils import load_cache, save_cache, TICKER_MATCH_CACHE_FILENAME, QUANT_FEATURES_CACHE_FILENAME, TINYBERT_EMBEDDINGS_CACHE_FILENAME, EMOBERT_FEATURES_CACHE_FILENAME


def extract_quantitative_features(df_news):
    logging.info("3. Извлечение количественных признаков")
    
    df_quant_features = load_cache(QUANT_FEATURES_CACHE_FILENAME)
    processed_news_ids_quant = set(df_quant_features['original_news_id'].unique()) if not df_quant_features.empty else set()
    quant_features_list = df_quant_features.to_dict('records')

    news_to_process_quant = df_news[~df_news['original_news_id'].isin(processed_news_ids_quant)].copy()

    if not news_to_process_quant.empty:
        for _, row in tqdm(news_to_process_quant.iterrows(), total=len(news_to_process_quant), desc="Извлечение количественных признаков"):
            news_id = row['original_news_id']
            full_text = str(row['full_text'])

            char_count = len(full_text)
            words = full_text.split()
            word_count = len(words)
            
            avg_word_len = np.mean([len(word) for word in words if word]) if words else 0
            caps_count = sum(1 for word in words if word.isupper() and len(word) > 1)
            
            word_counts = pd.Series(words).value_counts()
            repeated_words_count = (word_counts > 1).sum()

            special_chars = "!@#$%^&*()_+{}[]|:;'<>,.?/~`"
            special_char_count = sum(1 for char in full_text if char in special_chars)

            quant_features_list.append({
                'original_news_id': news_id,
                'char_count': char_count,
                'word_count': word_count,
                'avg_word_len': avg_word_len,
                'caps_count': caps_count,
                'repeated_words_count': repeated_words_count,
                'special_char_count': special_char_count
            })

        df_quant_features = pd.DataFrame(quant_features_list)
        save_cache(df_quant_features, QUANT_FEATURES_CACHE_FILENAME)
    else:
        logging.info("Все количественные признаки уже извлечены в кэше.")

    logging.info("Первые 5 строк с количественными признаками:")
    logging.info(df_quant_features.head())
    logging.info("-" * 50)
    return df_quant_features

def extract_tinybert_embeddings(df_news):
    logging.info(f"4. Извлечение TinyBERT эмбеддингов (Модель: {TINYBERT_MODEL_NAME}, Устройство: {DEVICE})")

    try:
        tinybert_tokenizer = AutoTokenizer.from_pretrained(TINYBERT_MODEL_NAME, local_files_only=True)
        tinybert_model = AutoModel.from_pretrained(TINYBERT_MODEL_NAME, local_files_only=True).to(DEVICE)
        logging.info("TinyBERT успешно загружен из локального пути.")
    except Exception as e:
        logging.error(f"Ошибка загрузки TinyBERT модели: {e}. Убедитесь, что модель находится по указанному пути и файлы корректны.")
        exit()

    tinybert_model.eval()

    df_tinybert_embeddings = load_cache(TINYBERT_EMBEDDINGS_CACHE_FILENAME)
    processed_news_ids_tinybert = set(df_tinybert_embeddings['original_news_id'].unique()) if not df_tinybert_embeddings.empty else set()
    tinybert_embeddings_list = df_tinybert_embeddings.to_dict('records')

    news_to_process_tinybert = df_news[~df_news['original_news_id'].isin(processed_news_ids_tinybert)].copy()

    if not news_to_process_tinybert.empty:
        texts = news_to_process_tinybert['full_text'].tolist()
        news_ids = news_to_process_tinybert['original_news_id'].tolist()

        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Извлечение TinyBERT эмбеддингов"):
            batch_texts = texts[i:i + BATCH_SIZE]
            batch_news_ids = news_ids[i:i + BATCH_SIZE]

            with torch.no_grad():
                inputs = tinybert_tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                
                outputs = tinybert_model(**inputs)
                
                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
                sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                mean_pooled_embeddings = sum_embeddings / sum_mask
                
                embeddings_np = mean_pooled_embeddings.cpu().numpy()

            for j, news_id in enumerate(batch_news_ids):
                tinybert_embeddings_list.append({
                    'original_news_id': news_id,
                    'tinybert_embedding': embeddings_np[j]
                })
            
            if (i // BATCH_SIZE + 1) % SAVE_INTERVAL == 0: 
                df_temp = pd.DataFrame(tinybert_embeddings_list)
                save_cache(df_temp, TINYBERT_EMBEDDINGS_CACHE_FILENAME)

        df_tinybert_embeddings = pd.DataFrame(tinybert_embeddings_list)
        save_cache(df_tinybert_embeddings, TINYBERT_EMBEDDINGS_CACHE_FILENAME)
    else:
        logging.info("Все TinyBERT эмбеддинги уже извлечены в кэше.")

    logging.info("Первые 5 строк с TinyBERT эмбеддингами (только ID и форма эмбеддинга):")
    if not df_tinybert_embeddings.empty:
        if 'tinybert_embedding' in df_tinybert_embeddings.columns and \
           not df_tinybert_embeddings['tinybert_embedding'].isnull().all() and \
           len(df_tinybert_embeddings['tinybert_embedding'].iloc[0].shape) > 0:
            logging.info(df_tinybert_embeddings[['original_news_id', 'tinybert_embedding']].head().assign(
                tinybert_embedding_shape=df_tinybert_embeddings['tinybert_embedding'].apply(lambda x: x.shape)
            ))
        else:
            logging.warning("TinyBERT эмбеддинги отсутствуют или имеют некорректный формат в кэше.")
    logging.info("-" * 50)
    return df_tinybert_embeddings


def extract_emobert_features(df_news):
    logging.info(f"5. Извлечение EmoBERT признаков (Модель: {EMOBERT_MODEL_NAME}, Устройство: {DEVICE})")

    try:
        emobert_tokenizer = AutoTokenizer.from_pretrained(EMOBERT_MODEL_NAME, local_files_only=True)
        emobert_model = AutoModelForSequenceClassification.from_pretrained(EMOBERT_MODEL_NAME, local_files_only=True).to(DEVICE)
        logging.info("EmoBERT успешно загружен из локального пути.")
    except Exception as e:
        logging.error(f"Ошибка загрузки EmoBERT модели: {e}. Убедитесь, что модель находится по указанному пути и файлы корректны.")
        exit()

    emobert_model.eval()

    df_emobert_features = load_cache(EMOBERT_FEATURES_CACHE_FILENAME)
    processed_news_ids_emobert = set(df_emobert_features['original_news_id'].unique()) if not df_emobert_features.empty else set()
    emobert_features_list = df_emobert_features.to_dict('records')

    news_to_process_emobert = df_news[~df_news['original_news_id'].isin(processed_news_ids_emobert)].copy()

    if not news_to_process_emobert.empty:
        texts = news_to_process_emobert['full_text'].tolist()
        news_ids = news_to_process_emobert['original_news_id'].tolist()

        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Извлечение EmoBERT признаков"):
            batch_texts = texts[i:i + BATCH_SIZE]
            batch_news_ids = news_ids[i:i + BATCH_SIZE]

            with torch.no_grad():
                inputs = emobert_tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                
                outputs = emobert_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()

            for j, news_id in enumerate(batch_news_ids):
                feature_dict = {'original_news_id': news_id}
                labels = emobert_model.config.id2label if hasattr(emobert_model.config, 'id2label') else {i: f'emotion_{i}' for i in range(probabilities.shape[1])}
                
                for k, prob in enumerate(probabilities[j]):
                    feature_dict[f'emobert_{labels[k]}'] = prob
                
                emobert_features_list.append(feature_dict)

            if (i // BATCH_SIZE + 1) % SAVE_INTERVAL == 0: 
                df_temp = pd.DataFrame(emobert_features_list)
                save_cache(df_temp, EMOBERT_FEATURES_CACHE_FILENAME)

        df_emobert_features = pd.DataFrame(emobert_features_list)
        save_cache(df_emobert_features, EMOBERT_FEATURES_CACHE_FILENAME)
    else:
        logging.info("Все EmoBERT признаки уже извлечены в кэше.")

    logging.info("Первые 5 строк с EmoBERT признаками:")
    logging.info(df_emobert_features.head())
    logging.info("-" * 50)
    return df_emobert_features
