import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib

# Automatically download necessary NLTK resources
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Preprocessing function
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Handle missing or invalid text values
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(filtered_tokens)

# Load the dataset
data = pd.read_csv(
    "twitter_dataset.csv",  # Replace with your file name
    encoding="ISO-8859-1",  # Encoding for this dataset
    names=["target", "ids", "date", "flag", "user", "text"]  # Column names
)

# Filter dataset to include only positive and negative labels
data = data[data['target'].isin([0, 4])]

# Shuffle the dataset to avoid grouped classes
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Handle missing or invalid values in the 'text' column
data = data.dropna(subset=['text'])
data['text'] = data['text'].astype(str)

# Map target labels
data['target'] = data['target'].map({0: 'negative', 4: 'positive'})

# Preprocess the text
data['cleaned_text'] = data['text'].apply(preprocess_text)

# Vectorize the text
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X = vectorizer.fit_transform(data['cleaned_text'])
y = data['target']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model on all training data
model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Evaluation - Classification Report:\n", classification_report(y_test, y_pred))

# Save the vectorizer and model for future use
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(model, "final_sentiment_model.pkl")
print("Model and vectorizer saved.")
