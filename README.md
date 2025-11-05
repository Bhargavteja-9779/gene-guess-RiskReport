# 🧬 GeneGuess – ML-Powered Genetic Disease Risk Estimator (Family History Based)

> An explainable, accessible, and privacy-friendly AI system to estimate hereditary disease risk using only family history and basic lifestyle data.

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-green.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-LogisticRegression%20%7C%20RandomForest-orange.svg)

---

## 🧩 Overview

**GeneGuess** is a machine learning–powered web application that estimates the probability of a genetic disease based on:
- Family health history (first- and second-degree relatives)
- Lifestyle indicators (smoking, BMI)
- Genetic markers and consanguinity

Built with **Python, Scikit-Learn, Flask, and Chart.js**, it provides:
- 📈 Risk probability prediction  
- 🧠 Feature contribution explanations  
- 🌐 Interactive web interface  
- ⚙️ Locally deployable, privacy-safe setup  

---

## 🚀 Demo Preview

### 🎯 Model Risk Prediction Interface
![App Screenshot](./screenshots/ui_home.png)

### 📊 ROC and Calibration Curves
| ROC Curve | Calibration Curve |
|------------|-------------------|
| ![ROC](./screenshots/roc_curve.png) | ![Cal](./screenshots/calibration.png) |

---

## 🧪 Features

- ✅ Synthetic dataset generator (`generate_data.py`)
- ✅ Logistic Regression and Random Forest models
- ✅ Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- ✅ Feature contribution visualization (Chart.js)
- ✅ Flask-based web interface
- ✅ Explainable ML (interpretable coefficients)

---

## 🧠 Architecture

Data Generation → Preprocessing → Model Training → Evaluation → Web Deployment  
        │               │                │                │  
        └──────────────►└───────────────►└──────────────► Flask UI

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-------------|
| Programming | Python 3.10+ |
| Libraries | scikit-learn, pandas, numpy, matplotlib, seaborn |
| Web Framework | Flask |
| Frontend | HTML, CSS, JS, Chart.js |
| Explainability | Coefficient-based / SHAP-style interpretation |

---

## 🧰 Setup Instructions

### 1️⃣ Clone the repository
git clone https://github.com/Bhargavteja-9779/gene-guess-RiskReport.git  
cd gene-guess-RiskReport

### 2️⃣ Create a virtual environment
python3 -m venv venv  
source venv/bin/activate      # Mac/Linux  
# or  
venv\Scripts\activate         # Windows

### 3️⃣ Install dependencies
pip install -r requirements.txt

### 4️⃣ Generate the synthetic dataset
python3 data/generate_data.py

### 5️⃣ Train the ML model
python3 train_model.py

### 6️⃣ Run the web app
cd webapp  
export FLASK_APP=app.py  
python -m flask run --host=127.0.0.1 --port=5000  

App runs locally at 👉 http://127.0.0.1:5000

---

## 📈 Results

| Metric | Logistic Regression | Random Forest |
|:-------|:--------------------:|:--------------:|
| Accuracy | **0.86** | 0.83 |
| Precision | 0.89 | 0.85 |
| Recall | **0.96** | 0.93 |
| F1-Score | **0.92** | 0.90 |
| ROC-AUC | **0.91** | 0.89 |

Best Model → Logistic Regression — more interpretable and well-calibrated.

---

## 📊 Visual Outputs

| Graph | Description |
|--------|--------------|
| roc_curve.png | ROC curve – discrimination ability |
| calibration.png | Calibration curve – probability reliability |
| confusion_matrix.png | Model classification visualization |
| metrics_summary.json | JSON file of evaluation metrics |

---

## 🧩 Explainability

Each prediction displays feature contributions, showing which input factors most influenced the result.

Top features by impact:
1. Known Genetic Marker  
2. First-Degree Relatives  
3. Consanguinity  
4. Age  
5. BMI  
6. Smoking  

---

## 🧭 Folder Structure

gene-guess-RiskReport/  
├── data/  
├── artifacts/  
├── webapp/  
│   ├── static/  
│   └── templates/  
├── train_model.py  
├── predict_cli.py  
├── explain_and_metrics.py  
├── requirements.txt  
└── GeneGuess_Report.docx  

---

## 🧾 References

1. Altman et al., *Bioinformatics*, 2020  
2. Nguyen et al., *Nature Communications*, 2021  
3. Sharma et al., *IEEE Access*, 2022  
4. Pedregosa et al., *JMLR*, 2011  
5. Lundberg & Lee, *NIPS*, 2017  
6. Bhargav Teja P.N., *VIT Vellore Project Report*, 2025  

---

## 🧭 Future Scope

- Integration with real-world clinical datasets  
- Improved calibration via Isotonic Regression  
- Mobile app deployment  
- Federated learning for hospitals  
- SHAP explainability dashboard  

---

## 👤 Author

**P. N. Bhargav Teja**  
VIT Vellore | Software & ML Developer  
LinkedIn: https://www.linkedin.com/in/bhargavteja-pn  
GitHub: https://github.com/Bhargavteja-9779  

---

## 🪪 License

Released under the MIT License
