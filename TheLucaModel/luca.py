from flask import Flask, request, jsonify, render_template
import os
import random
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

app = Flask(__name__)

# Sample training dataset (You can replace this with more comprehensive data)
training_data = [
    {"intent": "greet", "patterns": ["Hello", "Hi", "Hey"], "response": ["Hello! How can I help you?"]},
    {"intent": "goodbye", "patterns": ["Bye", "Goodbye", "See you later"], "response": ["Goodbye! Have a great day!"]},
    {"intent": "thanks", "patterns": ["Thanks", "Thank you", "Much appreciated"], "response": ["You're welcome!"]}
]

# Extract patterns and intents for model training
X_train = []
y_train = []
responses = {}

for data in training_data:
    intent = data["intent"]
    responses[intent] = data["response"]
    for pattern in data["patterns"]:
        X_train.append(pattern)
        y_train.append(intent)

# NLP Preprocessing
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Model Training
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Save Model and Vectorizer
pickle.dump(classifier, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

@app.route('/')
def chat_ui():
    return render_template("index.html")

@app.route('/train', methods=['GET'])
def train_ui():
    return render_template("train.html")

@app.route('/metrics', methods=['GET'])
def accuracy_ui():
    return render_template("metrics.html")

@app.route('/chat', methods=['POST'])
def process_user_input():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"response": "Please type a message."})

    # Load Model and Vectorizer
    classifier = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    
    # Vectorize user input
    input_vectorized = vectorizer.transform([user_input])

    # Predict Intent
    intent = classifier.predict(input_vectorized)[0]
    response = random.choice(responses.get(intent, ["I didn't understand that."]))

    return jsonify({"response": response})

@app.route('/train-model', methods=['POST'])
def train_model():
    # Receive new training data
    data = request.json

    if not data or "intent" not in data or "patterns" not in data:
        return jsonify({"message": "Invalid data format."})

    # Update training dataset
    new_intent = data["intent"]
    new_patterns = data["patterns"]

    X_train.extend(new_patterns)
    y_train.extend([new_intent] * len(new_patterns))

    # Retrain the model
    X_train_vectorized = vectorizer.fit_transform(X_train)
    classifier.fit(X_train_vectorized, y_train)

    # Save updated model and vectorizer
    pickle.dump(classifier, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

    return jsonify({"message": "Model retrained successfully!"})

if __name__ == "__main__":
    app.run(debug=True)
