import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.utils.data import DataLoader
from dataset import TwitterDataset, load_twitter_dataset
from model import SentimentClassifier

# Hyperparameters
dataset_path = "/Users/onuraltinkurt/repos/SWE599/twitter_dataset.csv"
BATCH_SIZE = 16
MAX_LENGTH = 128
MODEL_PATH = "/Users/onuraltinkurt/repos/SWE599/BERT/sentiment-training-twitter-pytorch/training-twitter-pytorch/outputs/models/sentiment_model-epoch_4.pth"

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    _, val_data = load_twitter_dataset(dataset_path)
    val_data = val_data.sample(n=10000, random_state=42)  # Sample exactly 500 rows
    print(f"Number of test samples: {len(val_data)}")  # Verify the test data size

    # Create dataset and dataloader
    val_dataset = TwitterDataset(val_data, max_length=MAX_LENGTH)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Load trained model
    model = SentimentClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))  # Map to CPU if necessary
    model.to(device)

    # Evaluate model
    preds, labels = evaluate_model(model, val_dataloader, device)

    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    f1 = f1_score(labels, preds, average='binary')

    # Print metrics
    print("Classification Report:\n", classification_report(labels, preds, target_names=["negative", "positive"]))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

