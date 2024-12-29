import os
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from transformers import AdamW, get_scheduler
from tqdm import tqdm
from model import SentimentClassifier
from dataset import TwitterDataset, load_twitter_dataset


assert torch.cuda.is_available()

WORLD_SIZE = torch.cuda.device_count()
GPU_IDS = list(range(WORLD_SIZE))

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 2e-5
MAX_LENGTH = 128

# DATASET_PATH = "/Users/onuraltinkurt/repos/SWE599/twitter_dataset.csv"
DATASET_PATH = "../datasets/twitter/training.1600000.processed.noemoticon.csv"

DATA_RESAMPLE_RATIO = None

OUTPUT_DIR = "./outputs"
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "models")
LOG_SAVE_DIR = os.path.join(OUTPUT_DIR, "logs")


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

def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank, world_size, freeze_bert, resume):

    ddp_setup(rank, world_size)

    # Load dataset
    train_data, val_data = load_twitter_dataset(DATASET_PATH)

    if DATA_RESAMPLE_RATIO is not None:
        n_train_samples = len(train_data)
        n_val_samples = len(val_data)

        sel_n_train_samples = int(n_train_samples * DATA_RESAMPLE_RATIO)
        sel_n_val_samples = int(n_val_samples * DATA_RESAMPLE_RATIO)

        train_data = train_data[: sel_n_train_samples]
        val_data = val_data[: sel_n_val_samples]

    print("N Samples Training: {}".format(len(train_data)))
    print("N Samples Validation: {}".format(len(val_data)))

    train_dataset = TwitterDataset(train_data, max_length=MAX_LENGTH)
    val_dataset = TwitterDataset(val_data, max_length=MAX_LENGTH)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=DistributedSampler(train_dataset, shuffle=True))
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=DistributedSampler(val_dataset))

    # Initialize model
    model = SentimentClassifier()

    if resume != "":
        if os.path.isfile(resume):
            print("Loading Model: {}".format(resume))
            model.load_state_dict(torch.load(resume, weights_only=True))
        else:
            print("Could not found Model: {}".format(resume))
            destroy_process_group()
            exit()

    model.to(rank)

    if freeze_bert:
        model.freeze_bert_layers()

    model = DDP(model, device_ids=[rank,])

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = len(train_dataloader) * EPOCHS
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        train_dataloader.sampler.set_epoch(epoch)
        val_dataloader.sampler.set_epoch(epoch)

        train_loss = train(model, train_dataloader, optimizer, scheduler, loss_fn, rank)
        val_loss, val_accuracy = evaluate(model, val_dataloader, loss_fn, rank)

        if rank == 0:
            print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

            train_log_file = os.path.join(LOG_SAVE_DIR, "training.csv")
            val_logs_file = os.path.join(LOG_SAVE_DIR, "validation.csv")
            model_file = os.path.join(MODEL_SAVE_DIR, "sentiment_model-epoch_{}.pth".format(epoch))

            with open(train_log_file, "a") as fp:
                fp.write("{},{}\n".format(epoch, train_loss))

            with open(val_logs_file, "a") as fp:
                fp.write("{},{},{}\n".format(epoch, val_loss, val_accuracy))

            # Save the model
            torch.save(model.module.state_dict(), model_file)

            print("Model saved as {}".format(model_file))

    # Save the model
    if rank == 0:
        final_model_file = os.path.join(MODEL_SAVE_DIR, "sentiment_model-final.pth")

        torch.save(model.module.state_dict(), final_model_file)

        print("Final model saved as {}".format(final_model_file))

    destroy_process_group()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog='Training', description='Model Training')
    parser.add_argument('--resume', help='model path', type=str, default="")
    parser.add_argument('--freeze-bert', action='store_true')
    args = parser.parse_args()

    freeze_bert = args.freeze_bert
    resume = args.resume

    if not os.path.isdir(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    if not os.path.isdir(LOG_SAVE_DIR):
        os.makedirs(LOG_SAVE_DIR)

    mp.spawn(main, args=(WORLD_SIZE, freeze_bert, resume), nprocs=WORLD_SIZE)
