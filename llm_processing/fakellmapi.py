import random
import re

# 2. Создать заглушку для взаимодействия с LLM API
print("2. Определение функции-заглушки для LLM API")

def fake_call_llm(prompt_template, news_text):
    """
    Имитирует вызов LLM API и возвращает предопределенные ответы
    на основе ключевых слов в тексте новости.
    """
    # тональность
    if "тональность" in prompt_template.lower():
        if "позитивно" in news_text.lower() or "рост" in news_text.lower() or "прибыль" in news_text.lower():
            return str(round(random.uniform(0.5, 1.0), 1)) # позитивно
        elif "негативно" in news_text.lower() or "падение" in news_text.lower() or "инфляция" in news_text.lower():
            return str(round(random.uniform(-1.0, -0.5), 1)) # негативно
        else:
            return str(round(random.uniform(-0.3, 0.3), 1)) # нейтрально

    # влияние
    elif "влияние" in prompt_template.lower():
        if "неожиданное решение" in news_text.lower() or "резко выросли" in news_text.lower() or "ажиотаж" in news_text.lower():
            return str(random.randint(7, 10)) # сильное влияние
        elif "умеренного влияния" in news_text.lower() or "стабилизировался" in news_text.lower() or "незначительное улучшение" in news_text.lower():
            return str(random.randint(1, 4)) # слабое
        else:
            return str(random.randint(4, 7)) # среднее

    # извлечение терминов
    elif "ключевые экономические термины" in prompt_template.lower():
        terms = []
        if "инфляция" in news_text.lower(): terms.append("инфляция")
        if "процентная ставка" in news_text.lower() or "ключевая ставка" in news_text.lower(): terms.append("процентная ставка")
        if "ввп" in news_text.lower(): terms.append("ВВП")
        if "рынок акций" in news_text.lower(): terms.append("рынок акций")
        if "нефть" in news_text.lower(): terms.append("нефть")
        if "золото" in news_text.lower(): terms.append("золото")
        if "безработица" in news_text.lower(): terms.append("безработица")
        if "прибыль" in news_text.lower(): terms.append("прибыль")
        if "национальной валюты" in news_text.lower(): terms.append("национальная валюта")
        return ", ".join(terms) if terms else "Нет данных"

    return "Неизвестный промпт" 

print("Функция 'fake_call_llm' готова.")
print("-" * 50)
