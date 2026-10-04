"""
train_model.py
----------------
Trains and compares a RandomForestClassifier and an XGBClassifier for
HelioLead AI lead scoring, evaluates with precision/recall/F1/ROC-AUC,
and saves the best model + its feature list for the Streamlit dashboard.
"""

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, roc_auc_score, RocCurveDisplay,
)
from xgboost import XGBClassifier

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss", random_state=42, n_jobs=-1,
    ),
}

results = {}
fig, ax = plt.subplots(figsize=(6, 5))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        "precision_class1": report["1"]["precision"],
        "recall_class1": report["1"]["recall"],
        "f1_class1": f1,
        "roc_auc": auc,
    }

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {auc:.3f}")

    RocCurveDisplay.from_predictions(y_test, y_proba, name=name, ax=ax)

ax.set_title("ROC Curve - RandomForest vs XGBoost")
plt.tight_layout()
plt.savefig("figures/roc_comparison.png", dpi=150)

with open("models/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

# Pick the better model by F1 on the positive (adopted) class
best_name = max(results, key=lambda k: results[k]["f1_class1"])
best_model = models[best_name]
print(f"\nBest model by F1 (adopted class): {best_name}")

joblib.dump(best_model, "models/heliolead_model.joblib")
joblib.dump(list(X_train.columns), "models/feature_columns.joblib")

# Feature importance plot for the chosen model
importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=True).tail(15)
fig, ax = plt.subplots(figsize=(8, 6))
importances.plot(kind="barh", ax=ax, color="#4C72B0")
ax.set_title(f"Top 15 Feature Importances ({best_name})")
plt.tight_layout()
plt.savefig("figures/feature_importance.png", dpi=150)

print("Saved model to models/heliolead_model.joblib")
print("Saved metrics to models/metrics.json")
print("Saved plots to figures/")
