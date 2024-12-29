import torch
from torch.utils.data import DataLoader
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AdamW, get_scheduler
from sklearn.metrics import classification_report
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

# Hyperparameters
EPOCHS = 1
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
MAX_LENGTH = 64
MODEL_NAME = "albert-base-v2"
DATASET_PATH = "/Users/onuraltinkurt/repos/SWE599/twitter_dataset.csv"

# Load and preprocess the Twitter dataset
def load_and_preprocess_twitter_dataset(file_path):
    data = pd.read_csv(file_path, header=None, encoding='latin1')
    data.columns = ["label", "id", "date", "query", "user", "text"]

    # Map labels: 0 -> 0 (negative), 4 -> 1 (positive)
    data["label"] = data["label"].map({0: 0, 4: 1})

    # Shuffle and split the dataset
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data = train_data.sample(n=50000)  # Use 1000 samples for training
    test_data = test_data.sample(n=5000) 
    return train_data, test_data

train_data, test_data = load_and_preprocess_twitter_dataset(DATASET_PATH)

# Load ALBERT tokenizer and model
tokenizer = AlbertTokenizer.from_pretrained(MODEL_NAME)
model = AlbertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Tokenize datasets
def tokenize_function(data):
    return tokenizer(
        data["text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

tokenized_train = tokenize_function(train_data)
tokenized_test = tokenize_function(test_data)

dataloader_train = DataLoader(
    dataset=list(zip(tokenized_train["input_ids"], tokenized_train["attention_mask"], train_data["label"])),
    batch_size=BATCH_SIZE,
    shuffle=True
)

dataloader_test = DataLoader(
    dataset=list(zip(tokenized_test["input_ids"], tokenized_test["attention_mask"], test_data["label"])),
    batch_size=BATCH_SIZE
)

# Move model to GPU if available
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu") 
model.to(device)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
num_training_steps = len(dataloader_train) * EPOCHS
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Training function
def train_model():
    model.train()
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        for input_ids, attention_mask, labels in tqdm(dataloader_train):
            optimizer.zero_grad()

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()

        print(f"Loss: {loss.item():.4f}")

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

    report = classification_report(all_labels, all_preds, target_names=["negative", "positive"])
    print("Classification Report:\n", report)

if __name__ == "__main__":
    train_model()
    evaluate_model()

    # Save the fine-tuned model
    model.save_pretrained("albert_twitter_sentiment_model")
    tokenizer.save_pretrained("albert_twitter_sentiment_model")
    print("Model and tokenizer saved to albert_twitter_sentiment_model/")

