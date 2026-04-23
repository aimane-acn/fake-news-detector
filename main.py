"""
Fake News Detection API — DistilBERT Version
=============================================
Uses OOP, NLP, FastAPI, Decorators, Async Requests, Web Scraping, MLOps
Dataset: ISOT Fake News Dataset (dataset/Fake.csv + dataset/True.csv)
Run `python train.py` first, then `uvicorn main:app`
"""

import re
import os
import time
import logging
import hashlib
import functools
import threading
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

import httpx
import uvicorn
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

torch.set_num_threads(1)

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"[DEVICE] Using: {DEVICE}")


# ─────────────────────────────────────────────
# Decorators
# ─────────────────────────────────────────────
def timing(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"[TIMING] {func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper


def retry(max_attempts=3, delay=1.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"[RETRY] {func.__name__} attempt {attempt}/{max_attempts} failed: {e}")
                    if attempt < max_attempts:
                        time.sleep(delay)
                    else:
                        raise
        return wrapper
    return decorator


def cache_result(func):
    _cache = {}
    _lock = threading.Lock()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()
        with _lock:
            if key not in _cache:
                _cache[key] = func(*args, **kwargs)
                logger.info(f"[CACHE] Stored {func.__name__}")
            else:
                logger.info(f"[CACHE] Hit {func.__name__}")
        return _cache[key]
    return wrapper


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────
@dataclass
class Article:
    title: str
    content: str
    url: Optional[str] = None
    source: Optional[str] = None
    scraped_at: datetime = field(default_factory=datetime.now)

    def full_text(self) -> str:
        return f"{self.title}. {self.content}"

    def word_count(self) -> int:
        return len(self.content.split())


@dataclass
class PredictionResult:
    label: str
    confidence: float
    processing_time_ms: float
    article_word_count: int


# ─────────────────────────────────────────────
# NLP Preprocessor Class
# ─────────────────────────────────────────────
class TextPreprocessor:
    """Light cleaning — DistilBERT handles tokenization itself."""

    def __init__(self):
        self.stopwords = {
            "the", "a", "an", "is", "in", "it", "of", "and", "to", "was", "for",
            "on", "are", "as", "at", "be", "by", "this", "with", "from", "or", "but"
        }

    def clean(self, text: str) -> str:
        text = str(text)
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"\(reuters\)", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract_features_manual(self, text: str) -> dict:
        exclamations = text.count("!")
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        word_count = len(text.split())
        return {
            "exclamations": exclamations,
            "caps_ratio": round(caps_ratio, 4),
            "word_count": word_count,
        }


# ─────────────────────────────────────────────
# PyTorch Dataset Class
# ─────────────────────────────────────────────
class NewsDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int],
                 tokenizer: DistilBertTokenizer, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────
# Web Scraper Class
# ─────────────────────────────────────────────
class NewsScraper:
    HEADERS = {"User-Agent": "Mozilla/5.0 (FakeNewsBot/1.0)"}
    TIMEOUT = 8

    @retry(max_attempts=3, delay=1.0)
    def scrape(self, url: str) -> Article:
        resp = requests.get(url, headers=self.HEADERS, timeout=self.TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        title = soup.find("title")
        title_text = title.get_text(strip=True) if title else "No title"

        paragraphs = soup.find_all("p")
        content = " ".join(p.get_text(strip=True) for p in paragraphs)
        if len(content) < 50:
            content = soup.get_text()[:1000]

        return Article(title=title_text, content=content, url=url)

    async def scrape_async(self, url: str) -> Article:
        async with httpx.AsyncClient(headers=self.HEADERS, timeout=self.TIMEOUT) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            title = soup.find("title")
            title_text = title.get_text(strip=True) if title else "No title"

            paragraphs = soup.find_all("p")
            content = " ".join(p.get_text(strip=True) for p in paragraphs)
            if len(content) < 50:
                content = soup.get_text()[:1000]

            return Article(title=title_text, content=content, url=url)


# ─────────────────────────────────────────────
# DistilBERT Classifier Class (MLOps)
# ─────────────────────────────────────────────
class BERTFakeNewsClassifier:
    """
    Fine-tunes distilbert-base-uncased for binary fake news classification.
    Labels: 0 = RELIABLE, 1 = FAKE
    """

    MODEL_DIR  = "bert_model"
    BERT_NAME  = "distilbert-base-uncased"
    MAX_LEN    = 128
    BATCH_SIZE = 8
    EPOCHS     = 1
    LR         = 2e-5

    def __init__(self, preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor
        self.tokenizer: Optional[DistilBertTokenizer] = None
        self.model: Optional[DistilBertForSequenceClassification] = None
        self.is_trained = False
        self.metrics: dict = {}

    def _load_tokenizer(self):
        if self.tokenizer is None:
            logger.info("[BERT] Loading tokenizer…")
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.BERT_NAME)

    @timing
    def train(self, texts: list[str], labels: list[int]) -> dict:
        self._load_tokenizer()

        logger.info(f"[BERT] Cleaning {len(texts)} texts…")
        cleaned = [self.preprocessor.clean(t) for t in texts]

        X_train, X_test, y_train, y_test = train_test_split(
            cleaned, labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )

        train_dataset = NewsDataset(X_train, y_train, self.tokenizer, self.MAX_LEN)
        test_dataset  = NewsDataset(X_test,  y_test,  self.tokenizer, self.MAX_LEN)

        train_loader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        test_loader  = DataLoader(test_dataset,  batch_size=self.BATCH_SIZE)

        logger.info("[BERT] Loading distilbert-base-uncased for sequence classification…")
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.BERT_NAME,
            num_labels=2,
        ).to(DEVICE)

        optimizer = AdamW(self.model.parameters(), lr=self.LR, weight_decay=0.01)
        total_steps = len(train_loader) * self.EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        for epoch in range(1, self.EPOCHS + 1):
            self.model.train()
            total_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels_tensor  = batch["label"].to(DEVICE)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_tensor,
                )
                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

                if (batch_idx + 1) % 50 == 0:
                    logger.info(
                        f"[BERT] Epoch {epoch}/{self.EPOCHS} "
                        f"Step {batch_idx+1}/{len(train_loader)} "
                        f"Loss: {loss.item():.4f}"
                    )

            avg_loss = total_loss / len(train_loader)
            logger.info(f"[BERT] Epoch {epoch} avg loss: {avg_loss:.4f}")

        # ── Evaluation ──
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in test_loader:
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds   = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["label"].numpy())

        acc    = accuracy_score(all_labels, all_preds)
        report = classification_report(
            all_labels, all_preds,
            target_names=["Reliable", "Fake"],
            output_dict=True,
        )

        self.metrics = {
            "accuracy":   round(acc, 4),
            "train_size": len(X_train),
            "test_size":  len(X_test),
            "report":     report,
        }
        self.is_trained = True
        logger.info(f"[BERT] Test accuracy: {acc:.4f}")
        return self.metrics

    def save(self):
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        self.model.save_pretrained(self.MODEL_DIR)
        self.tokenizer.save_pretrained(self.MODEL_DIR)
        logger.info(f"[BERT] Saved to {self.MODEL_DIR}/")

    def load(self):
        if not os.path.exists(self.MODEL_DIR):
            raise FileNotFoundError(f"'{self.MODEL_DIR}' not found. Run `python train.py` first.")
        logger.info(f"[BERT] Loading from {self.MODEL_DIR}/…")
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.MODEL_DIR)
        self.model     = DistilBertForSequenceClassification.from_pretrained(self.MODEL_DIR).to(DEVICE)
        self.model.eval()
        self.is_trained = True
        logger.info("[BERT] Model loaded ✓")

    @timing
    def predict(self, text: str) -> tuple[str, float]:
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        self.model.eval()
        cleaned = self.preprocessor.clean(text)

        encoding = self.tokenizer(
            cleaned,
            max_length=self.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids      = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        pred       = int(np.argmax(probs))
        confidence = float(np.max(probs))
        label      = "FAKE" if pred == 1 else "RELIABLE"
        return label, confidence


# ─────────────────────────────────────────────
# Dataset Loader Class
# ─────────────────────────────────────────────
class DatasetBuilder:
    def load_kaggle_dataset(
        self,
        fake_path: str = "dataset/Fake.csv",
        true_path: str = "dataset/True.csv",
        sample_size: Optional[int] = None,
    ) -> tuple[list[str], list[int]]:
        fake = pd.read_csv(fake_path)
        real = pd.read_csv(true_path)

        fake["label"] = 1
        real["label"] = 0

        if sample_size:
            fake = fake.sample(sample_size, random_state=42)
            real = real.sample(sample_size, random_state=42)

        df = pd.concat([fake, real], ignore_index=True)
        df = df.dropna(subset=["title", "text"])
        df["content"] = df["title"].fillna("") + ". " + df["text"].fillna("")
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(
            f"[DATASET] {len(df)} articles — "
            f"fake={df['label'].sum()}, real={(df['label']==0).sum()}"
        )
        return df["content"].tolist(), df["label"].tolist()


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Fake News Detection API — DistilBERT",
    description="Fine-tuned distilbert-base-uncased for fake news detection.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

preprocessor    = TextPreprocessor()
classifier      = BERTFakeNewsClassifier(preprocessor)
scraper         = NewsScraper()
dataset_builder = DatasetBuilder()


@app.on_event("startup")
def startup_event():
    if os.path.exists(BERTFakeNewsClassifier.MODEL_DIR):
        classifier.load()
    else:
        raise RuntimeError(
            "No trained model found. Run `python train.py` first."
        )


# ─── Pydantic Schemas ───
class TextInput(BaseModel):
    text: str

class URLInput(BaseModel):
    url: str

class TrainInput(BaseModel):
    texts: list[str]
    labels: list[int]  # 1=fake, 0=reliable


# ─── Endpoints ───
@app.get("/", response_class=HTMLResponse)
async def root():
    file_path = os.path.join(os.path.dirname(__file__), "templates/index.html")
    if not os.path.exists(file_path):
        return "<h1>index.html not found ❌</h1>"
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
    
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "distilbert-base-uncased",
        "device": str(DEVICE),
        "model_trained": classifier.is_trained,
        "timestamp": datetime.now().isoformat(),
    }

@app.post("/predict")
async def predict(body: TextInput):
    if not body.text.strip():
        raise HTTPException(400, "Text cannot be empty.")
    t0 = time.perf_counter()
    label, confidence = classifier.predict(body.text)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    features = preprocessor.extract_features_manual(body.text)
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "processing_time_ms": round(elapsed_ms, 2),
        "features": features,
    }

@app.post("/predict-url")
async def predict_url(body: URLInput):
    try:
        article = await scraper.scrape_async(body.url)
    except Exception as e:
        raise HTTPException(400, f"Could not scrape URL: {e}")
    t0 = time.perf_counter()
    label, confidence = classifier.predict(article.full_text())
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "url": body.url,
        "title": article.title,
        "word_count": article.word_count(),
        "label": label,
        "confidence": round(confidence, 4),
        "processing_time_ms": round(elapsed_ms, 2),
    }

@app.post("/train")
async def train_endpoint(body: TrainInput):
    if len(body.texts) != len(body.labels):
        raise HTTPException(400, "texts and labels must have the same length.")
    if len(body.texts) < 10:
        raise HTTPException(400, "Provide at least 10 samples.")
    metrics = classifier.train(body.texts, body.labels)
    classifier.save()
    return {"status": "trained", "metrics": metrics}

@app.get("/metrics")
async def metrics():
    if not classifier.metrics:
        raise HTTPException(400, "Model not yet trained.")
    return classifier.metrics

@app.get("/demo-data")
async def demo_data():
    texts, labels = dataset_builder.load_kaggle_dataset(sample_size=3)
    return {"count": len(texts), "texts": texts[:4], "labels": labels[:4]}

@app.post("/retrain")
async def retrain(sample_size: int = 500):
    texts, labels = dataset_builder.load_kaggle_dataset(sample_size=sample_size)
    metrics = classifier.train(texts, labels)
    classifier.save()
    return {"status": "retrained", "metrics": metrics}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)