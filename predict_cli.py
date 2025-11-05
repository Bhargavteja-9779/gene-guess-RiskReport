import joblib
import pandas as pd
import sys
import json
import os

# Find the joblib in artifacts (relative to current working dir)
MODEL_PATH = None
art_dir = os.path.join(os.getcwd(), 'artifacts')
if os.path.isdir(art_dir):
    for f in os.listdir(art_dir):
        if f.endswith('.joblib'):
            MODEL_PATH = os.path.join(art_dir, f)
            break

if MODEL_PATH is None:
    print("No model found in artifacts/. Run training first.")
    sys.exit(1)

model = joblib.load(MODEL_PATH)

features = ['age','first_deg_relatives','second_deg_relatives',
            'consanguinity','known_marker','smoker','bmi']

def sanitize(d):
    # Ensure types and clamp expected ranges
    out = {}
    out['age'] = int(max(0, min(120, float(d.get('age', 0)))))
    out['first_deg_relatives'] = int(max(0, min(50, float(d.get('first_deg_relatives', 0)))))
    out['second_deg_relatives'] = int(max(0, min(100, float(d.get('second_deg_relatives', 0)))))
    # binary fields must be 0 or 1
    for k in ('consanguinity','known_marker','smoker'):
        val = int(float(d.get(k, 0)))
        out[k] = 1 if val >= 1 else 0
    out['bmi'] = float(max(10.0, min(60.0, float(d.get('bmi', 20.0)))))
    return out

def predict_from_dict(d):
    d2 = sanitize(d)
    df = pd.DataFrame([d2], columns=features)
    prob = model.predict_proba(df)[0][1]  # probability of label=1
    pred = int(prob >= 0.5)
    return pred, prob, d2

if len(sys.argv) > 1:
    # Expect a JSON string as argument
    try:
        data = json.loads(sys.argv[1])
        pred, prob, d2 = predict_from_dict(data)
        print("Sanitized input:", d2)
        print(f"Predicted label: {pred}, risk probability: {prob:.4f}")
    except Exception as e:
        print("Failed to parse JSON:", e)
else:
    print("Interactive mode. Enter values or press enter to use example.")
    example = {'age':45,'first_deg_relatives':1,'second_deg_relatives':1,
               'consanguinity':0,'known_marker':0,'smoker':0,'bmi':24.5}
    for f in features:
        raw = input(f + f" ({type(example[f]).__name__}) [default {example[f]}]: ").strip()
        if raw != '':
            try:
                example[f] = type(example[f])(raw)
            except:
                print(f"Could not parse {f}, using default {example[f]}")
    pred, prob, d2 = predict_from_dict(example)
    print("Sanitized input:", d2)
    print(f"Predicted label: {pred}, risk probability: {prob:.4f}")
