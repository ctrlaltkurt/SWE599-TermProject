import torch
from transformers import BertTokenizer
from model import SentimentClassifier

# Parameters
MODEL_PATH = "outputs/models/sentiment_model-epoch_4.pth"
BERT_MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128

def predict_sentiment(text, model, tokenizer, device):
    model.eval()

    # Tokenize input text
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return pred, probs.squeeze().tolist()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    # Load trained model
    model = SentimentClassifier()
    #model.load_state_dict(torch.load(MODEL_PATH))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.to(device)

    # Input text
    text = input("Enter a tweet for sentiment analysis: ")

    # Predict sentiment
    pred, probs = predict_sentiment(text, model, tokenizer, device)
    label_map = {0: "Negative", 1: "Positive"}

    print(f"Sentiment: {label_map[pred]}\nProbabilities: {probs}")

