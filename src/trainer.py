import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import copy

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_function,
    optimizer,
    device: str,
    epochs: int,
    model_path: str
):
    """
    Функция для обучения и валидации PyTorch модели.
    """
    best_val_loss = np.inf
    early_stopping_counter = 0
    patience = 10 # Сколько эпох ждать улучшения

    for epoch in range(epochs):
        model.train() # Переводим модель в режим обучения
        
        train_loss = 0.0
        # Оборачиваем train_loader в tqdm для прогресс-бара
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for X_batch, y_batch in progress_bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Прямой проход
            y_pred = model(X_batch).squeeze(1)
            loss = loss_function(y_pred, y_batch)
            train_loss += loss.item()

            # Обратный проход
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix(loss=loss.item())

        # Валидация
        model.eval() # Переводим модель в режим оценки
        val_loss = 0.0
        with torch.no_grad(): # Отключаем расчет градиентов
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch).squeeze(1)
                loss = loss_function(y_pred, y_batch)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Логика сохранения лучшей модели и ранней остановки
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"  Validation loss decreased. Saving model to {model_path}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= patience:
            print(f"  Early stopping after {patience} epochs with no improvement.")
            break
            
    # Загружаем веса лучшей модели
    model.load_state_dict(torch.load(model_path))
    return model