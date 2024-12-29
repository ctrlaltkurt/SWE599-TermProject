import joblib

# Load the trained model and vectorizer
model = joblib.load("final_sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Define a function to predict sentiment
def predict_sentiment(text):
    # Transform the input text using the saved vectorizer
    features = vectorizer.transform([text])
    # Predict sentiment using the trained model
    prediction = model.predict(features)[0]
    return prediction

# Interactive mode
if __name__ == "__main__":
    print("Interactive Sentiment Analysis")
    print("Type your text below (type 'exit' to quit):\n")
    
    while True:
        user_input = input("Enter text: ")
        if user_input.lower() == "exit":
            print("Exiting sentiment analysis. Goodbye!")
            break

        sentiment = predict_sentiment(user_input)
        print(f"Predicted Sentiment: {sentiment}\n")

