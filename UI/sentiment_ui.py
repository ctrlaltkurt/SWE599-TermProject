from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QTextEdit, QComboBox
from transformers import pipeline, BertTokenizer, BertForSequenceClassification, AlbertTokenizer, AlbertForSequenceClassification
import torch
import joblib

class SentimentAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentiment Analysis UI")
        self.setGeometry(100, 100, 600, 400)

        # Logistic Regression Paths
        self.lr_model_path = "/Users/onuraltinkurt/repos/SWE599/logisticregression/final_sentiment_model.pkl"
        self.lr_vectorizer_path = "/Users/onuraltinkurt/repos/SWE599/logisticregression/vectorizer.pkl"

        # BERT Paths
        self.bert_model_path = "/Users/onuraltinkurt/repos/SWE599/BERT/sentiment-training-twitter-pytorch/training-twitter-pytorch/outputs/models/sentiment_model-epoch_4.pth"
        self.bert_tokenizer_name = "bert-base-uncased"

        # ALBERT Paths
        self.albert_model_path = "/Users/onuraltinkurt/repos/SWE599/ALBERT/albert_twitter_sentiment_model"

        print("Loading models...")

        # Load Logistic Regression Model
        self.lr_model = joblib.load(self.lr_model_path)
        self.lr_vectorizer = joblib.load(self.lr_vectorizer_path)
        print("Logistic Regression model loaded.")

        # Load BERT Model and Tokenizer
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_tokenizer_name)
        self.bert_model = BertForSequenceClassification.from_pretrained(self.bert_tokenizer_name, num_labels=2)
        self.bert_model.load_state_dict(torch.load(self.bert_model_path, map_location=torch.device("cpu")))
        self.bert_model.eval()
        self.bert_pipeline = pipeline("text-classification", model=self.bert_model, tokenizer=self.bert_tokenizer)
        print("BERT model loaded.")

        # Load ALBERT Model and Tokenizer
        self.albert_tokenizer = AlbertTokenizer.from_pretrained(self.albert_model_path)
        self.albert_model = AlbertForSequenceClassification.from_pretrained(self.albert_model_path)
        self.albert_model.eval()
        self.albert_pipeline = pipeline("text-classification", model=self.albert_model, tokenizer=self.albert_tokenizer)
        print("ALBERT model loaded.")

        # Main widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout
        self.layout = QVBoxLayout()

        # Model selection dropdown
        self.model_label = QLabel("Select Model:")
        self.layout.addWidget(self.model_label)

        self.model_selector = QComboBox()
        self.model_selector.addItems(["Logistic Regression", "BERT", "ALBERT"])
        self.layout.addWidget(self.model_selector)

        # Text input
        self.text_input_label = QLabel("Enter Text:")
        self.layout.addWidget(self.text_input_label)

        self.text_input = QTextEdit()
        self.layout.addWidget(self.text_input)

        # Predict button
        self.predict_button = QPushButton("Analyze Sentiment")
        self.predict_button.clicked.connect(self.predict_sentiment)
        self.layout.addWidget(self.predict_button)

        # Result display
        self.result_label = QLabel("")
        self.layout.addWidget(self.result_label)

        self.central_widget.setLayout(self.layout)

    def predict_sentiment(self):
        input_text = self.text_input.toPlainText().strip()
        selected_model = self.model_selector.currentText()

        if not input_text:
            self.result_label.setText("Please enter some text.")
            return

        try:
            if selected_model == "Logistic Regression":
                features = self.lr_vectorizer.transform([input_text])
                prediction = self.lr_model.predict(features)[0]
                self.result_label.setText(f"Logistic Regression Prediction: {'positive' if prediction == 'positive' else 'negative'}")
            elif selected_model == "BERT":
                result = self.bert_pipeline(input_text)[0]
                sentiment = 'positive' if result['label'] == 'LABEL_1' else 'negative'
                self.result_label.setText(f"BERT Prediction: {sentiment} - Score: {result['score']:.2f}")
            elif selected_model == "ALBERT":
                result = self.albert_pipeline(input_text)[0]
                sentiment = 'positive' if result['label'] == 'LABEL_1' else 'negative'
                self.result_label.setText(f"ALBERT Prediction: {sentiment} - Score: {result['score']:.2f}")
            else:
                self.result_label.setText("Model not recognized.")
        except Exception as e:
            self.result_label.setText(f"Error: {str(e)}")

# Run the application
if __name__ == "__main__":
    app = QApplication([])
    window = SentimentAnalysisApp()
    window.show()
    app.exec()

