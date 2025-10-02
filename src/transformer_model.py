import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Добавляет информацию о позиции в последовательности.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor, shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    """
    Основная модель Трансформера для временных рядов.
    """
    def __init__(self, num_features: int, d_model: int, n_head: int, num_encoder_layers: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        
        # 1. Линейный слой для преобразования входных признаков в d_model
        self.input_embedding = nn.Linear(num_features, d_model)
        
        # 2. Позиционное кодирование
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 3. Основной кодировщик Трансформера
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 4. Финальный линейный слой для получения предсказания
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        src: shape [batch_size, seq_len, num_features]
        """
        # 1. Преобразуем признаки
        src = self.input_embedding(src) * math.sqrt(self.d_model)
        
        # 2. Добавляем позиционное кодирование. Нужно поменять оси для PE.
        # [batch, seq, feature] -> [seq, batch, feature]
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)
        # Возвращаем оси обратно: [seq, batch, feature] -> [batch, seq, feature]
        src = src.permute(1, 0, 2)
        
        # 3. Прогоняем через кодировщик
        output = self.transformer_encoder(src)
        
        # 4. Берем выход только последнего временного шага
        output = output[:, -1, :]
        
        # 5. Получаем финальное предсказание
        output = self.output_linear(output)
        
        return output