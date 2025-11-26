# Health-Diagnosis-AI

Machine Learning and NLP based health diagnosis system that predicts the most likely disease based on user-provided symptoms.

## 📌 Project Overview

This Health Diagnosis System uses Sentence Transformer embeddings to understand symptoms written in natural language.
It compares the input with a curated medical knowledge base and returns the closest matching disease along with confidence score and useful reference information.

## 📊 Dataset

The dataset consists of disease names, medical descriptions, and symptom references encoded into embeddings.

(Custom-built dataset — not publicly available.)

## 🧠 Features

- Accepts user details (name, age, gender, weight) and symptoms.

- Predicts disease using semantic similarity (cosine similarity).

- Returns matched medical sentence for interpretability.

- Displays confidence score of prediction.

- Clean and responsive Streamlit UI.

## 🛠️ Tools & Technologies Used

- Python 🐍

- Sentence Transformers (all-MiniLM-L6-v2)

- Scikit-learn

- Streamlit (UI & deployment)

- NumPy, Pandas

- Pickle (model storage)

## 🔍Sample Input

| Parameter | Value                          |
| --------- | ------------------------------ |
| Full Name | Aayush                         |
| Age       | 23                             |
| Gender    | Male                           |
| Weight    | 72                             |
| Symptoms  | mild fever, cough, throat pain |

## 🟢 Output Example

```
Predicted Disease → Influenza
Matched Text → "Patients may experience fever, sore throat and cough."
Confidence Score → 0.83
Source → https://example-medical-link.com
```
## 🚀 How to Run Locally

### 1. Clone the repository:

```
git clone https://github.com/Aayush-infinity2/Health-Diagnosis-AI.git
cd Health-Diagnosis-AI
```

### 2. Install dependencies:

``pip install -r requirements.txt
``

### 3. Run the Streamlit app:

`` streamlit run main.py
``

## 🌍 Deployment

Deployed via Streamlit Cloud.
Push your repository and connect it with the Streamlit platform — it will automatically handle deployment.

## 👨‍💻 Authors

### Aayush Sharma

## ⚠️ Disclaimer

This project is intended only for educational and research purposes.
It is not a medical device and does not replace professional healthcare advice.
Please consult certified medical professionals for real health concerns.



