#!/usr/bin/env python3
"""
Robust explain_and_metrics script (fixed index alignment).
Run from project root or webapp: python3 explain_and_metrics.py
"""
import os, sys, json
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Resolve project root (directory containing this script)
THIS = os.path.abspath(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(THIS), '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'family_history.csv')
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts')

if not os.path.exists(DATA_PATH):
    alt = os.path.abspath(os.path.join(os.path.dirname(THIS), '..', 'data', 'family_history.csv'))
    if os.path.exists(alt):
        DATA_PATH = alt
        ARTIFACTS_DIR = os.path.abspath(os.path.join(os.path.dirname(THIS), '..', 'artifacts'))
    else:
        print("ERROR: cannot find data/family_history.csv. looked at:", DATA_PATH, alt)
        sys.exit(2)

if not os.path.isdir(ARTIFACTS_DIR):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

print("Using DATA_PATH:", DATA_PATH)
print("Using ARTIFACTS_DIR:", ARTIFACTS_DIR)

# Try to import shap, optional
try:
    import shap
    SHAP_AVAILABLE = True
    print("shap available")
except Exception:
    SHAP_AVAILABLE = False
    print("shap NOT available")

# Load data
df = pd.read_csv(DATA_PATH)
if 'disease_label' not in df.columns:
    print("ERROR: data does not have 'disease_label' column. Columns:", df.columns.tolist())
    sys.exit(3)

X = df.drop(columns=['disease_label'])
y = df['disease_label']

# Load model
MODEL = None
for f in os.listdir(ARTIFACTS_DIR):
    if f.endswith('.joblib'):
        MODEL = joblib.load(os.path.join(ARTIFACTS_DIR, f))
        print("Loaded model:", f)
        break
if MODEL is None:
    print("ERROR: no .joblib model found in", ARTIFACTS_DIR)
    sys.exit(4)

# Train/test split (same as training)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Reset indices so positional arrays align with DataFrame rows
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Predictions/probs (positional)
y_pred = MODEL.predict(X_test)
if hasattr(MODEL, "predict_proba"):
    y_prob = MODEL.predict_proba(X_test)[:,1]
else:
    try:
        from scipy.special import expit
        raw = MODEL.decision_function(X_test)
        y_prob = expit(raw)
    except Exception:
        y_prob = y_pred.astype(float)

# Metrics
metrics = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred, zero_division=0)),
    'recall': float(recall_score(y_test, y_pred, zero_division=0)),
    'f1': float(f1_score(y_test, y_pred, zero_division=0)),
    'roc_auc': float(roc_auc_score(y_test, y_prob)),
    'brier_score': float(brier_score_loss(y_test, y_prob))
}

clf_report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred).tolist()

out = {'metrics': metrics, 'classification_report': clf_report, 'confusion_matrix': cm}
with open(os.path.join(ARTIFACTS_DIR, 'metrics_summary.json'), 'w') as fh:
    json.dump(out, fh, indent=2)
print("Saved metrics_summary.json")

# ROC
from sklearn.metrics import roc_curve
fpr, tpr, thr = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'ROC AUC = {metrics["roc_auc"]:.3f}')
plt.plot([0,1],[0,1],'--', linewidth=0.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACTS_DIR, 'roc_curve.png'))
print("Saved roc_curve.png")

# Calibration
plt.figure(figsize=(6,5))
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibration')
plt.plot([0,1],[0,1],'--', linewidth=0.8)
plt.xlabel('Predicted probability')
plt.ylabel('True probability (fraction)')
plt.title('Calibration Curve')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACTS_DIR, 'calibration.png'))
print("Saved calibration.png")

# Save confusion matrix plot if seaborn available
try:
    import seaborn as sns
    import matplotlib.pyplot as plt2
    plt2.figure(figsize=(4,3))
    sns.heatmap(pd.DataFrame(cm), annot=True, fmt='d', cmap='Blues')
    plt2.xlabel('Predicted'); plt2.ylabel('Actual'); plt2.title('Confusion Matrix')
    plt2.tight_layout()
    plt2.savefig(os.path.join(ARTIFACTS_DIR, 'confusion_matrix.png'))
    print("Saved confusion_matrix.png")
except Exception:
    print("seaborn not available or failed; skipping confusion matrix image.")

# Explanations (SHAP if available, else linear contributions)
feature_names = X.columns.tolist()
def compute_linear_contributions(model, X_ref, x_row):
    try:
        if hasattr(model, 'named_steps'):
            scaler = model.named_steps.get('scaler', None)
            clf = model.named_steps.get('clf', model)
        else:
            scaler = None
            clf = model
        if scaler is not None:
            X_ref_scaled = scaler.transform(X_ref)
            x_scaled = scaler.transform(x_row.to_frame().T)[0]
            coef = np.array(clf.coef_)[0]
            intercept = float(clf.intercept_[0]) if hasattr(clf, 'intercept_') else 0.0
            mean_scaled = X_ref_scaled.mean(axis=0)
            contrib = coef * (x_scaled - mean_scaled)
            return {feat: float(c) for feat,c in zip(feature_names, contrib)}, float(intercept)
        else:
            coef = np.array(clf.coef_)[0]
            mean = X_ref.mean(axis=0).values
            contrib = coef * (x_row.values - mean)
            intercept = float(clf.intercept_[0]) if hasattr(clf, 'intercept_') else 0.0
            return {feat: float(c) for feat,c in zip(feature_names, contrib)}, float(intercept)
    except Exception as e:
        raise

explanations = {}
X_background = X_train.sample(min(100, len(X_train)), random_state=42)

# Build examples using positional indexing
examples = {
    'low_risk_example': X_test[np.array(y_prob) < 0.2].head(1),
    'mid_risk_example': X_test[(np.array(y_prob) >= 0.4) & (np.array(y_prob) < 0.6)].head(1),
    'high_risk_example': X_test[np.array(y_prob) > 0.8].head(1)
}

for name, df_ex in examples.items():
    if df_ex.shape[0] == 0:
        explanations[name] = {'error': 'no example'}
        continue
    # get positional row index (iloc 0 corresponds to same position in y_prob & y_pred)
    pos = df_ex.index[0]  # since X_test has reset index, this is positional
    x_row = df_ex.iloc[0]
    example_prob = float(y_prob[pos])
    example_label = int(y_pred[pos])
    if SHAP_AVAILABLE:
        try:
            explainer = shap.Explainer(MODEL.predict_proba, X_background, feature_names=feature_names)
            shap_vals = explainer(df_ex)[0].values[:,1]
            contribs = {f: float(v) for f,v in zip(feature_names, shap_vals)}
            base = float(explainer(df_ex).base_values[0][1])
            explanations[name] = {
                'method': 'shap',
                'probability': example_prob,
                'predicted_label': example_label,
                'base_value': base,
                'contributions': contribs
            }
            continue
        except Exception:
            pass
    try:
        contribs, intercept = compute_linear_contributions(MODEL, X_background, x_row)
        explanations[name] = {
            'method': 'linear_contrib',
            'probability': example_prob,
            'predicted_label': example_label,
            'intercept': intercept,
            'contributions': contribs
        }
    except Exception as e:
        explanations[name] = {'error': str(e)}

with open(os.path.join(ARTIFACTS_DIR, 'explanations.json'), 'w') as fh:
    json.dump(explanations, fh, indent=2)
print("Saved explanations.json")
