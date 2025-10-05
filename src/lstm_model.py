import torch
import torch.nn as nn

class PyTorchLSTM(nn.Module):
    def __init__(self, num_features: int, hidden_units: int, num_layers: int = 2, dropout: float = 0.2):
        """
        Архитектура LSTM модели на PyTorch.
        
        Args:
            num_features (int): Количество входных признаков.
            hidden_units (int): Количество нейронов в LSTM слоях.
            num_layers (int): Количество LSTM слоев.
            dropout (float): Вероятность Dropout.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True, # Очень важный параметр!
            num_layers=num_layers,
            dropout=dropout
        )
        self.linear = nn.Linear(in_features=hidden_units, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход данных через модель.
        """
        # LSTM возвращает output и (hidden_state, cell_state)
        # Нам нужен только output последнего временного шага
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        
        # Прогоняем через линейный слой для получения предсказания
        prediction = self.linear(last_time_step_out)
        return prediction