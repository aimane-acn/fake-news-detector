# 🧠 Fake News Detection API (DistilBERT)

A full-stack AI project that detects fake news using NLP and Deep Learning.

## 🚀 Features

* 🔍 Detect fake vs reliable news
* 🌐 URL scraping (auto article extraction)
* 🧠 DistilBERT fine-tuned model
* ⚡ FastAPI backend
* 🎨 Modern web UI

## 🏗️ Tech Stack

* Python, FastAPI
* PyTorch, Transformers
* BeautifulSoup (web scraping)
* HTML, CSS, JavaScript

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector

pip install -r requirements.txt
```

## 🧠 Train the model

```bash
python train.py
```

## 🚀 Run the API

```bash
uvicorn main:app --reload
```

Open in browser:

```
http://127.0.0.1:8000
```

## 📊 API Endpoints

* POST `/predict`
* POST `/predict-url`
* GET `/health`
* GET `/metrics`

## ⚠️ Notes

* Train the model before running the API
* Dataset not included (add your own Kaggle dataset)

## 📌 Future Improvements

* Multilingual support (Arabic/French)
* Better scraping robustness
* Deployment (Docker, cloud)

## 👨‍💻 Author

Nouredine
