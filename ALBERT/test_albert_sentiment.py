import torch
from torch.utils.data import DataLoader
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from sklearn.metrics import classification_report
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

# Hyperparameters
BATCH_SIZE = 8
MAX_LENGTH = 64
MODEL_DIR = "albert_twitter_sentiment_model"  # Directory of the saved fine-tuned model
DATASET_PATH = "/Users/onuraltinkurt/repos/SWE599/twitter_dataset.csv"

# Load and preprocess the Twitter dataset
def load_and_preprocess_twitter_dataset(file_path):
    data = pd.read_csv(file_path, header=None, encoding='latin1')
    data.columns = ["label", "id", "date", "query", "user", "text"]

    # Map labels: 0 -> 0 (negative), 4 -> 1 (positive)
    data["label"] = data["label"].map({0: 0, 4: 1})

    # Sample 500 random test examples
    test_data = data.sample(n=10000, random_state=None)
    return test_data

test_data = load_and_preprocess_twitter_dataset(DATASET_PATH)

# Load ALBERT tokenizer and model
tokenizer = AlbertTokenizer.from_pretrained(MODEL_DIR)
model = AlbertForSequenceClassification.from_pretrained(MODEL_DIR)

# Tokenize the test dataset
def tokenize_function(data):
    return tokenizer(
        data["text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

tokenized_test = tokenize_function(test_data)

dataloader_test = DataLoader(
    dataset=list(zip(tokenized_test["input_ids"], tokenized_test["attention_mask"], test_data["label"])),
    batch_size=BATCH_SIZE
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Evaluation function
def evaluate_model():
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(dataloader_test):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=["negative", "positive"])
    print("Classification Report:\n", report)

if __name__ == "__main__":
    print("Testing the model on a new dataset...")
    evaluate_model()

