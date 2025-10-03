from transformers import PatchTSTConfig, PatchTSTForPrediction
from src.config import *

def create_regression_model(num_input_channels: int) -> PatchTSTForPrediction:
    """
    Создает и возвращает сконфигурированную модель PatchTST для регрессии.
    """
    config = PatchTSTConfig(
        num_input_channels=num_input_channels,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        patch_length=PATCH_LENGTH,
        patch_stride=PATCH_LENGTH,
        d_model=REG_D_MODEL,
        num_attention_heads=REG_N_HEADS,
        num_hidden_layers=REG_ENCODER_LAYERS,
        ffn_dim=256,
        dropout=REG_DROPOUT,
        head_dropout=REG_DROPOUT,
        scaling="std",
        loss="mse",
    )
    model = PatchTSTForPrediction(config)
    return model

def create_probability_model(num_input_channels: int) -> PatchTSTForPrediction:
    """
    Создает и возвращает сконфигурированную модель PatchTST для предсказания вероятностей.
    Мы используем ту же регрессионную архитектуру, но будем обучать ее на бинарных таргетах.
    """
    config = PatchTSTConfig(
        num_input_channels=num_input_channels,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        patch_length=PATCH_LENGTH,
        patch_stride=PATCH_LENGTH,
        d_model=PROB_D_MODEL,
        num_attention_heads=PROB_N_HEADS,
        num_hidden_layers=PROB_ENCODER_LAYERS,
        ffn_dim=128,
        dropout=PROB_DROPOUT,
        head_dropout=PROB_DROPOUT,
        scaling="std",
        loss="mse", # Будем использовать MSE для обучения на 0/1, это аналог Brier Score
        output_range=[-0.1, 1.1] # Ограничиваем выход, чтобы получить вероятности
    )
    model = PatchTSTForPrediction(config)
    return model