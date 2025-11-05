from flask import Flask, render_template, request, jsonify
import joblib, os, json, pandas as pd
app = Flask(__name__)

# locate model and artifacts
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
artifacts_dir = os.path.join(base_dir, 'artifacts')
MODEL = None
for f in os.listdir(artifacts_dir):
    if f.endswith('.joblib'):
        MODEL = joblib.load(os.path.join(artifacts_dir, f))
        break
if MODEL is None:
    raise RuntimeError("Model not found in artifacts. Run training first.")

# Load metrics and example explanations if available
metrics_path = os.path.join(artifacts_dir, 'metrics_summary.json')
explanations_path = os.path.join(artifacts_dir, 'explanations.json')
metrics_summary = {}
explanations = {}
if os.path.exists(metrics_path):
    with open(metrics_path) as fh:
        metrics_summary = json.load(fh)
if os.path.exists(explanations_path):
    with open(explanations_path) as fh:
        explanations = json.load(fh)

# keep the last predicted sample contributions in-memory (simple)
last_prediction_info = {}

features = ['age','first_deg_relatives','second_deg_relatives',
            'consanguinity','known_marker','smoker','bmi']

def sanitize_form(form):
    try:
        out = {}
        out['age'] = int(max(0, min(120, float(form.get('age', 0)))))
        out['first_deg_relatives'] = int(max(0, min(50, float(form.get('first_deg_relatives', 0)))))
        out['second_deg_relatives'] = int(max(0, min(100, float(form.get('second_deg_relatives', 0)))))
        for k in ('consanguinity','known_marker','smoker'):
            val = int(float(form.get(k, 0)))
            out[k] = 1 if val >= 1 else 0
        out['bmi'] = float(max(10.0, min(60.0, float(form.get('bmi', 20.0)))))
        return out, None
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    return render_template('index.html', metrics=metrics_summary.get('metrics', {}))

@app.route('/predict', methods=['POST'])
def predict():
    data, err = sanitize_form(request.form)
    if err:
        return jsonify({'error': 'Invalid input: ' + err}), 400
    # DataFrame for model
    df = pd.DataFrame([data], columns=features)
    try:
        if hasattr(MODEL, "predict_proba"):
            prob = float(MODEL.predict_proba(df)[0][1])
        else:
            prob = float(MODEL.predict(df)[0])
        label = int(prob >= 0.5)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # generate per-feature contributions: try load explanations.json precomputed; if not, compute a simple linear contribution
    contribs = {}
    if os.path.exists(explanations_path):
        # try to find closest example by predicted label
        try:
            loaded = json.load(open(explanations_path))
            # choose high_risk_example if prob>=0.5 else low
            key = 'high_risk_example' if prob >= 0.5 else 'low_risk_example'
            if key in loaded and 'contributions' in loaded[key]:
                contribs = loaded[key]['contributions']
        except Exception:
            contribs = {}
    # Fallback calculation: for linear pipeline-like model, use coef*(x-mean)
    if not contribs:
        try:
            # attempt to compute simple contribution from pipeline if accessible
            model = MODEL
            # get coef from classifier inside pipeline
            if hasattr(model, 'named_steps'):
                clf = model.named_steps.get('clf', model)
                scaler = model.named_steps.get('scaler', None)
            else:
                clf = model
                scaler = None
            # load training data means if available
            # attempt to read training data to compute mean
            df_train = pd.read_csv(os.path.join(base_dir, 'data', 'family_history.csv'))
            X_train = df_train.drop(columns=['disease_label'])
            if scaler is not None:
                X_train_scaled = scaler.transform(X_train)
                x_scaled = scaler.transform(df)[0]
                coef = getattr(clf, 'coef_', None)
                if coef is not None:
                    contribs = {f: float(c) for f,c in zip(features, (coef[0] * (x_scaled - X_train_scaled.mean(axis=0))))}
            else:
                coef = getattr(clf, 'coef_', None)
                if coef is not None:
                    contribs = {f: float(c) for f,c in zip(features, (coef[0] * (df.iloc[0].values - X_train.mean(axis=0).values)))}
        except Exception:
            contribs = {}

    # store last prediction info for /explain call
    last_prediction_info['data'] = data
    last_prediction_info['probability'] = prob
    last_prediction_info['label'] = label
    last_prediction_info['contributions'] = contribs

    resp = {'label': label, 'probability': prob, 'sanitized_input': data, 'contributions': contribs}
    return jsonify(resp)

@app.route('/explain', methods=['GET'])
def explain():
    if not last_prediction_info:
        return jsonify({'error': 'No prediction yet. Call /predict first.'}), 400
    return jsonify(last_prediction_info)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
