# train.py — run once: python train.py
import logging
from main import preprocessor, classifier, dataset_builder, BERTFakeNewsClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Loading dataset…")
    texts, labels = dataset_builder.load_kaggle_dataset(sample_size=500)

    logger.info("Training model…")
    classifier.train(texts, labels)

    logger.info("Saving model…")
    classifier.save()

    print(f"\n✅ Model saved to '{BERTFakeNewsClassifier.MODEL_DIR}/'")
    print(f"📊 Accuracy: {classifier.metrics['accuracy']}")
    print(f"📦 Train size: {classifier.metrics['train_size']}")
    print(f"🧪 Test size:  {classifier.metrics['test_size']}")