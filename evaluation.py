import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

def evaluate_model(y_true, y_pred):
    """
    Menghitung metrik evaluasi: akurasi, presisi, recall, dan f1-score.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

def generate_classification_report(y_true, y_pred, target_names=None):
    """
    Menghasilkan classification report dalam bentuk string.
    """
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)

def plot_confusion_matrix(y_true, y_pred, labels, figsize=(5,4)):
    """
    Menampilkan confusion matrix dalam bentuk heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    return plt
