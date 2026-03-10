"""
Train a simple Random Forest classifier on the Iris dataset
and save the model + metadata for Hugging Face upload.
"""

import json
import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


def train(n_estimators: int = 100, max_depth: int = 5, random_state: int = 42):
    print("=== Loading dataset ===")
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = list(iris.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    print("\n=== Training model ===")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    print("\n=== Evaluating model ===")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5)

    print(f"Test Accuracy : {accuracy:.4f}")
    print(f"CV Accuracy   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # ── Save artefacts ────────────────────────────────────────────────────────
    out = Path("model")
    out.mkdir(exist_ok=True)

    model_path = out / "iris_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {model_path}")

    metrics = {
        "test_accuracy": round(accuracy, 4),
        "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
        "cv_std_accuracy": round(float(cv_scores.std()), 4),
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "class_names": class_names,
        "feature_names": list(iris.feature_names),
    }
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {metrics_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Iris classifier")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    metrics = train(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
    print("\n=== Done ===")
    print(json.dumps(metrics, indent=2))
