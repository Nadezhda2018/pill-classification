# Файл: src/eval.py
import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(model, loader, device):
    """
    Прогоняет модель по датасету и возвращает истинные метки и предсказания.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    return np.array(all_labels), np.array(all_preds)

def get_metrics_report(y_true, y_pred, class_names):
    """
    Генерирует словарь с метриками для каждого класса.
    """
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    return report