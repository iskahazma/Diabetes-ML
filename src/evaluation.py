# src/evaluation.py
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import numpy as np

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """Train and evaluate a classification model."""

    # Train
    model.fit(X_train, y_train)

    # Predict labels
    y_pred = model.predict(X_test)

    # Predict probabilities for ROC-AUC
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        # Normalize 0–1
        y_proba = (scores - scores.min()) / (scores.max() - scores.min())

    # Metrics — NO pos_label used (auto-detect)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print("=" * 50)
    print(f"Model: {model_name}")
    print("=" * 50)
    print("Accuracy:", round(acc,4))
    if roc is not None:
        print("ROC AUC :", round(roc,4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("=" * 50)

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC-AUC": roc
    }
