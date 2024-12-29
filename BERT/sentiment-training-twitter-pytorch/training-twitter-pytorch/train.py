import os
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from tqdm import tqdm
from model import SentimentClassifier
from dataset import TwitterDataset, load_twitter_dataset

# Hyperparameters
EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
MAX_LENGTH = 128

# dataset_path = "/Users/onuraltinkurt/repos/SWE599/twitter_dataset.csv"
dataset_path = "../datasets/twitter/training.1600000.processed.noemoticon.csv"

data_resample_ratio = None

output_dir = "./outputs"
model_save_dir = os.path.join(output_dir, "models")
log_save_dir = os.path.join(output_dir, "logs")

def train(model, dataloader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_data, val_data = load_twitter_dataset(dataset_path)

    if data_resample_ratio is not None:
        n_train_samples = len(train_data)
        n_val_samples = len(val_data)

        sel_n_train_samples = int(n_train_samples * data_resample_ratio)
        sel_n_val_samples = int(n_val_samples * data_resample_ratio)

        train_data = train_data[: sel_n_train_samples]
        val_data = val_data[: sel_n_val_samples]

    print("N Samples Training: {}".format(len(train_data)))
    print("N Samples Validation: {}".format(len(val_data)))

    train_dataset = TwitterDataset(train_data, max_length=MAX_LENGTH)
    val_dataset = TwitterDataset(val_data, max_length=MAX_LENGTH)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize model
    model = SentimentClassifier()
    model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = len(train_dataloader) * EPOCHS
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    if not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir)

    if not os.path.isdir(log_save_dir):
        os.makedirs(log_save_dir)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss = train(model, train_dataloader, optimizer, scheduler, loss_fn, device)
        val_loss, val_accuracy = evaluate(model, val_dataloader, loss_fn, device)

        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        train_log_file = os.path.join(log_save_dir, "training.csv")
        val_logs_file = os.path.join(log_save_dir, "validation.csv")
        model_file = os.path.join(model_save_dir, "sentiment_model-epoch_{}.pth".format(epoch))

        with open(train_log_file, "a") as fp:
            fp.write("{}".format(train_loss))

        with open(val_logs_file, "a") as fp:
            fp.write("{},{}".format(val_loss, val_accuracy))

        # Save the model
        torch.save(model.state_dict(), model_file)

        print("Model saved as {}".format(model_file))

    # Save the model
    final_model_file = os.path.join(model_save_dir, "sentiment_model-final.pth")

    torch.save(model.state_dict(), final_model_file)

    print("Final model saved as {}".format(final_model_file))
