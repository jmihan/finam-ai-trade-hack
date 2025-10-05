
---

# FINAM HACK — Решение команды

## Структура проекта

Решение находится в ветке **`ensemble`**.
Из этой ветки необходима только папка **`ready_pipeline`**.

После клонирования репозитория создайте следующую структуру директорий внутри `ready_pipeline`:

```
ready_pipeline/
├── data/
│   ├── raw/
│   │   ├── candles.csv
│   │   └── news.csv
│   └── processed/
├── models/
│   └── rubert-tiny-sentiment-balanced/
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── ... (остальные файлы модели)
├── main.py
├── train_model.py
├── requirements.txt
└── ...
```

---

## Данные

В папку `ready_pipeline/data/raw/` необходимо поместить файлы:

* `candles.csv`
* `news.csv`

Файлы должны содержать данные **по 8 сентября включительно** (в соответствии с текущим заданием).

---

## Предобученная модель

В директории `ready_pipeline/models/` создайте папку:

```
rubert-tiny-sentiment-balanced/
```

Скачайте все исходные файлы предобученной модели по ссылке:

 [cointegrated/rubert-tiny-sentiment-balanced — Hugging Face](https://huggingface.co/cointegrated/rubert-tiny-sentiment-balanced/tree/main)

и поместите их в эту папку.

---

## Установка и запуск

### 1. Установка зависимостей

Убедитесь, что установлена версия **Python 3.11.0**.
Далее выполните установку зависимостей:

```bash
pip install -r requirements.txt
```

Отдельно установите библиотеку **PatchTST** (модель от IBM):

```bash
pip install git+https://github.com/IBM/tsfm.git
```

---

### 2. Запуск пайплайна

Запуск возможен:

* либо из папки `ready_pipeline`,
* либо из корневой директории проекта.

Команда запуска:

```bash
python main.py
```

После успешного выполнения скрипта в папке `ready_pipeline` появится файл:

```
submission.csv
```

— это финальный результат работы пайплайна.

---

## Дополнительная информация

* Версия Python: **3.11.0**
* Все зависимости указаны в `requirements.txt`
* Модель **PatchTST** используется для временных рядов и требует установки из репозитория IBM (см. выше).

---

