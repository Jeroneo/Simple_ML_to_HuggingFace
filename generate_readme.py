"""
Generate a Hugging Face model card (README.md) from metrics.json
"""

import json
from pathlib import Path
from datetime import datetime


def generate_readme(metrics_path: str = "model/metrics.json") -> str:
    metrics = json.loads(Path(metrics_path).read_text())

    accuracy    = metrics["test_accuracy"]
    cv_mean     = metrics["cv_mean_accuracy"]
    cv_std      = metrics["cv_std_accuracy"]
    classes     = metrics["class_names"]
    features    = metrics["feature_names"]
    n_est       = metrics["n_estimators"]
    max_d       = metrics["max_depth"]
    train_n     = metrics["train_samples"]
    test_n      = metrics["test_samples"]
    date        = datetime.utcnow().strftime("%Y-%m-%d")

    readme = f"""---
license: mit
tags:
  - sklearn
  - classification
  - iris
  - random-forest
  - tabular
library_name: sklearn
---

# 🌸 Iris Classifier — Random Forest

A simple **Random Forest** classifier trained on the classic
[Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html).
Deployed automatically via GitHub Actions.

## 📊 Evaluation Results

| Metric | Value |
|---|---|
| Test Accuracy | **{accuracy:.4f}** |
| CV Accuracy (5-fold) | **{cv_mean:.4f} ± {cv_std:.4f}** |
| Train samples | {train_n} |
| Test samples | {test_n} |

## 🏗️ Model Details

| Parameter | Value |
|---|---|
| Algorithm | Random Forest |
| n_estimators | {n_est} |
| max_depth | {max_d} |

## 📥 Usage

```python
import pickle, requests, numpy as np
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(repo_id="YOUR_HF_USERNAME/iris-classifier", filename="iris_classifier.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Predict  (sepal length, sepal width, petal length, petal width)
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(sample)
class_names = {classes}
print(class_names[prediction[0]])  # -> 'setosa'
```

## 📋 Features

The model uses {len(features)} features:
{chr(10).join(f"- `{f}`" for f in features)}

## 🏷️ Classes

{", ".join(f"`{c}`" for c in classes)}

---
*Last trained: {date}*
"""

    Path("model/README.md").write_text(readme)
    print("README.md generated.")
    return readme


if __name__ == "__main__":
    generate_readme()
    print(Path("model/README.md").read_text())
