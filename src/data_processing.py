import numpy as np
import pandas as pd
from typing import Tuple

def create_sequences(
    data: pd.DataFrame, 
    sequence_length: int, 
    target_column: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Преобразует DataFrame в последовательности для LSTM.
    
    Args:
        data (pd.DataFrame): DataFrame со всеми числовыми признаками и таргетом.
                             ВАЖНО: данные должны быть уже отмасштабированы (scaled)!
        sequence_length (int): Длина входной последовательности (lookback window).
        target_column (str): Название целевой переменной.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: X (признаки) и y (таргет) в формате numpy.
    """
    X, y = [], []
    target_idx = data.columns.get_loc(target_column)
    
    for i in range(len(data) - sequence_length):
        # Входная последовательность (окно данных)
        seq_x = data.iloc[i:i + sequence_length].values
        # Целевое значение (сразу после окна)
        seq_y = data.iloc[i + sequence_length, target_idx]
        
        X.append(seq_x)
        y.append(seq_y)
        
    return np.array(X), np.array(y)