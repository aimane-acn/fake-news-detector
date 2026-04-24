# 🧠 Fake News Detection API (DistilBERT)

A full-stack AI project that detects fake news using NLP and Deep Learning with a fine-tuned DistilBERT model.

---

## 🚀 Features

* 🔍 Detect fake vs reliable news
* 🌐 URL scraping (automatic article extraction)
* 🧠 Fine-tuned DistilBERT model
* ⚡ FastAPI backend
* 🎨 Simple web interface

---

## 🏗️ Tech Stack

* Python, FastAPI
* PyTorch, HuggingFace Transformers
* BeautifulSoup (web scraping)
* HTML / CSS / JavaScript

---

## 📦 Installation

```bash
git clone https://github.com/aimane-acn/fake-news-detector.git
cd fake-news-detector

python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

---

## 📊 Dataset Setup

This project uses the **ISOT Fake News Dataset**.

👉 Download from Kaggle:
https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

Place the files like this:

```
dataset/
 ├── Fake.csv
 └── True.csv
```

---

## 🧠 Train the Model

```bash
python train.py
```

This will:

* Train DistilBERT
* Save the model in `bert_model/`

---

## 🚀 Run the API

```bash
uvicorn main:app --reload
```

Open in browser:

```
http://127.0.0.1:8000
```

---

## 📡 API Endpoints

* POST `/predict` → classify text
* POST `/predict-url` → classify article from URL
* GET `/health` → system status
* GET `/metrics` → model performance

---

## ⚠️ Important Notes

* ❗ You must train the model before running the API
* ❗ Dataset is not included (download manually)
* ❗ Model files are not included (generated after training)

---

## 📌 Future Improvements

* Multilingual support (Arabic / French)
* Model optimization
* Docker deployment
* Cloud deployment (Render / AWS)

---

## 👨‍💻 Author

Aimane Achibane

---

⭐ If you like this project, consider giving it a star on GitHub!
