import logging
import time
import sys
import os

# --- Настройка логирования для главного скрипта ---
# Это гарантирует, что сообщения будут выводиться с самого начала
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [MAIN] - %(message)s'
)

# --- Блок для корректного импорта модулей проекта ---
# Позволяет запускать main.py из любой директории
try:
    from generate_ts_features import run_ts_feature_generation
    from generate_nlp_features import run_nlp_feature_generation
    from prepare_final_dataset import run_final_dataset_preparation
    from train_model import run_model_training
    from predict import run_prediction
except ImportError as e:
    logging.error(f"Ошибка импорта: {e}")
    logging.error("Убедитесь, что все скрипты (generate_*, prepare_*, train_model.py, predict.py) находятся в той же директории, что и main.py")
    sys.exit(1)


def main():
    """
    Основной пайплайн, запускающий все этапы проекта последовательно.
    """
    start_time = time.time()
    logging.info(">>> ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА РЕШЕНИЯ <<<")

    try:
        # --- Шаг 1: Генерация признаков из временных рядов ---
        logging.info("\n" + "="*50)
        logging.info("--- [ШАГ 1/5] Запуск генерации TS признаков ---")
        run_ts_feature_generation()
        logging.info("--- [ШАГ 1/5] Генерация TS признаков УСПЕШНО ЗАВЕРШЕНА ---")
        logging.info("="*50)

        # --- Шаг 2: Генерация признаков из новостей ---
        logging.info("\n" + "="*50)
        logging.info("--- [ШАГ 2/5] Запуск генерации NLP признаков ---")
        run_nlp_feature_generation()
        logging.info("--- [ШАГ 2/5] Генерация NLP признаков УСПЕШНО ЗАВЕРШЕНА ---")
        logging.info("="*50)

        # --- Шаг 3: Подготовка финальных датасетов ---
        logging.info("\n" + "="*50)
        logging.info("--- [ШАГ 3/5] Запуск подготовки финальных датасетов (train/inference) ---")
        run_final_dataset_preparation()
        logging.info("--- [ШАГ 3/5] Подготовка датасетов УСПЕШНО ЗАВЕРШЕНА ---")
        logging.info("="*50)

        # --- Шаг 4: Обучение модели ---
        logging.info("\n" + "="*50)
        logging.info("--- [ШАГ 4/5] Запуск обучения модели PatchTST ---")
        run_model_training()
        logging.info("--- [ШАГ 4/5] Обучение модели УСПЕШНО ЗАВЕРШЕНО ---")
        logging.info("="*50)

        # --- Шаг 5: Генерация предсказаний ---
        logging.info("\n" + "="*50)
        logging.info("--- [ШАГ 5/5] Запуск генерации предсказаний ---")
        run_prediction()
        logging.info("--- [ШАГ 5/5] Генерация предсказаний УСПЕШНО ЗАВЕРШЕНА ---")
        logging.info("="*50)

    except Exception as e:
        logging.error("!!! КРИТИЧЕСКАЯ ОШИБКА ВО ВРЕМЯ ВЫПОЛНЕНИЯ ПАЙПЛАЙНА !!!")
        # Выводим полную информацию об ошибке
        logging.error(e, exc_info=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        logging.info(f"Пайплайн завершился С ОШИБКОЙ. Общее время выполнения: {total_time / 60:.2f} минут.")
        
        return # Выходим из функции

    end_time = time.time()
    total_time = end_time - start_time
    logging.info("\n>>> ПОЛНЫЙ ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН! <<<")
    logging.info(f"Итоговый файл с предсказаниями сохранен в 'artifacts/predictions.csv'")
    logging.info(f"Общее время выполнения: {total_time / 60:.2f} минут.")


if __name__ == '__main__':
    main()