import torch
import torch.nn as nn
from transformers import BertModel

class SentimentClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_labels=2):
        super(SentimentClassifier, self).__init__()
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.3)
        
        # Classification head
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use the [CLS] token representation
        cls_output = outputs.pooler_output

        # Apply dropout and classification layer
        logits = self.classifier(self.dropout(cls_output))

        return logits

    def freeze_bert_layers(self, freeze=True):
        # Freeze BERT layers if freeze=True
        for param in self.bert.parameters():
            param.requires_grad = not freeze

if __name__ == "__main__":
    # Example usage
    model = SentimentClassifier()

    # Freeze BERT layers
    model.freeze_bert_layers(freeze=True)

    # Example input tensors
    input_ids = torch.randint(0, 30522, (2, 128))  # Random token IDs for batch of 2
    attention_mask = torch.ones((2, 128))  # Attention mask

    # Forward pass
    outputs = model(input_ids, attention_mask)
    print("Logits:", outputs)

