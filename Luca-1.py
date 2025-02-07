import torch
import torch.nn as nn
import torch.optim as optim
import random
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Ensure necessary NLTK data is downloaded
nltk.download("punkt")

# Define training data for intent classification
training_data = [
    ("hello", "greeting"),
    ("hi", "greeting"),
    ("hey", "greeting"),
    ("goodbye", "goodbye"),
    ("bye", "goodbye"),
    ("thank you", "thanks"),
    ("thanks", "thanks"),
    ("what's the weather", "weather"),
    ("is it raining", "weather"),
    ("what is your name", "name"),
    ("who are you", "name")
]

# Preprocess data
vectorizer = CountVectorizer(tokenizer=word_tokenize)
X_train = vectorizer.fit_transform([text for text, _ in training_data]).toarray()
y_train = [label for _, label in training_data]
label_dict = {label: i for i, label in enumerate(set(y_train))}
y_train = np.array([label_dict[label] for label in y_train])

# Context dictionary (stores last few interactions per user)
context = {}
context_limit = 3  # Number of recent interactions to store

# Define a simple neural network for intent classification
class IntentClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(IntentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 8)
        self.fc2 = nn.Linear(8, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

# Train intent classifier
input_size = X_train.shape[1]
num_classes = len(label_dict)
model = IntentClassifier(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train_intent_classifier():
    for epoch in range(100):
        inputs = torch.tensor(X_train, dtype=torch.float32)
        labels = torch.tensor(y_train, dtype=torch.long)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

def predict_intent(text, user_id):
    text_vector = vectorizer.transform([text]).toarray()
    inputs = torch.tensor(text_vector, dtype=torch.float32)
    outputs = model(inputs)
    predicted_label = torch.argmax(outputs).item()
    intent = list(label_dict.keys())[predicted_label]
    
    # Store context
    if user_id not in context:
        context[user_id] = []
    context[user_id].append(intent)
    if len(context[user_id]) > context_limit:
        context[user_id].pop(0)  # Keep only recent interactions
    
    return intent

# Predefined responses
responses = {
    "greeting": ["Hello! How can I help you?", "Hi there!", "Hey! What's up?"],
    "goodbye": ["Goodbye! Have a great day!", "See you later!", "Take care!"],
    "thanks": ["You're welcome!", "Anytime!", "Glad to help!"],
    "weather": ["I can't check the weather, but you can try a weather website!"],
    "name": ["I'm a chatbot created to assist you!", "Call me ChatBot!"]
}

def generate_response(intent, user_id):
    if user_id in context and context[user_id].count(intent) > 1:
        return f"You've asked about {intent} before. Would you like more details?"
    return random.choice(responses[intent])

def chatbot_response(user_input, user_id="default_user"):
    intent = predict_intent(user_input, user_id)
    return generate_response(intent, user_id)

def chat():
    print("Chatbot: Hello! Type 'quit' to exit.")
    user_id = "user1"
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Chatbot: Goodbye!")
            break
        response = chatbot_response(user_input, user_id)
        print("Chatbot:", response)

if __name__ == "__main__":
    train_intent_classifier()
    chat()
