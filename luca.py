from flask import Flask, render_template, request, jsonify, session
import re
import csv
import random
import json
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session handling

# Load responses from a CSV file
responses = {}
with open('responses.csv', mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        key, *values = row
        responses[key] = values

# File to store unrecognized inputs
unrecognized_file = 'unrecognized_inputs.json'

# Ensure the JSON file exists
if not os.path.exists(unrecognized_file):
    with open(unrecognized_file, 'w') as f:
        json.dump([], f)

# File to store emotional inputs
emotional_file = 'emotional_inputs.json'

# Ensure the emotional inputs JSON file exists
if not os.path.exists(emotional_file):
    with open(emotional_file, 'w') as f:
        json.dump([], f)

# Helper function to normalize user input
def normalize_input(user_input):
    # Convert to lowercase and remove symbols except spaces
    user_input = re.sub(r'[^\w\s]', '', user_input.lower())
    return user_input

# Helper function to extract a name from the input
def extract_name(user_input):
    # Match a phrase like "my friend <name>", "angry with <name>", or "with <name>"
    match = re.search(r"(?:my friend|angry with|with) (\b[A-Za-z]+\b)", user_input, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()  # Extract and capitalize the name
    return None

# Emotion keyword mapping
emotion_keywords = {
    "happy": ["happy", "excited", "joyful", "glad"],
    "sad": ["sad", "unhappy", "depressed", "down"],
    "angry": ["angry", "mad", "furious", "annoyed"],
    "confused": ["confused", "lost", "unsure", "puzzled"],
    "bored": ["bored", "disinterested", "uninterested", "tired"]
}

# Helper function to detect emotions in user input
def detect_emotion(user_input):
    for emotion, keywords in emotion_keywords.items():
        if any(keyword in user_input for keyword in keywords):
            return emotion
    return None

# Store emotional input in a file
def store_emotional_input(user_input, emotion):
    with open(emotional_file, 'r+') as f:
        data = json.load(f)
        data.append({"input": user_input, "emotion": emotion})
        f.seek(0)
        json.dump(data, f, indent=4)

# Initialize or update conversation context
def update_context(key, value):
    if 'context' not in session:
        session['context'] = {}
    session['context'][key] = value

# Retrieve context by key
def get_context(key):
    return session.get('context', {}).get(key)

# Helper function to find the best match for the user's input
def get_response(user_input):
    normalized_input = normalize_input(user_input)

    # Check for specific phrases like "angry with my friend <name>"
    name = extract_name(user_input)
    if name:
        update_context('friend_name', name)
        update_context('conversation_state', {'topic': 'friend', 'expected_response': 'details'})
        return f"Oh. What did {name} do to you?"

    # Detect emotion from user input
    emotion = detect_emotion(normalized_input)
    if emotion:
        store_emotional_input(user_input, emotion)
        if emotion == "happy":
            update_context('conversation_state', {'topic': 'happy', 'expected_response': 'details'})
            return "I'm glad to hear that! What's making you so happy?"
        elif emotion == "sad":
            update_context('conversation_state', {'topic': 'sad', 'expected_response': 'confirmation'})
            return "I'm sorry you're feeling sad. Do you want to talk about it?"
        elif emotion == "angry":
            update_context('conversation_state', {'topic': 'angry', 'expected_response': 'details'})
            return "I see you're angry. Want to share what's bothering you?"
        elif emotion == "confused":
            update_context('conversation_state', {'topic': 'confused', 'expected_response': 'details'})
            return "Being confused is tough. Maybe I can help clarify things?"
        elif emotion == "bored":
            update_context('conversation_state', {'topic': 'bored', 'expected_response': 'details'})
            return "Feeling bored? Maybe try something new or exciting!"

    # Handle responses based on the current conversation state
    state = get_context('conversation_state')
    if state:
        if state['expected_response'] == 'confirmation' and 'yes' and 'ok' and 'sure' in normalized_input:
            if state['topic'] == 'sad':
                update_context('conversation_state', {'topic': 'sad', 'expected_response': 'details'})
                return "I'm here to listen. Can you share what happened?"
        elif state['expected_response'] == 'details':
            update_context('conversation_state', None)  # Reset state after getting details
            return "Thanks for sharing that. How are you feeling now?"

    # Use context to make responses more dynamic
    friend_name = get_context('friend_name')
    if friend_name and 'friend' in normalized_input:
        return f"Are you still thinking about {friend_name}?"

    # Default behavior if no special case is matched
    for key, value in responses.items():
        if re.search(key, normalized_input):
            return random.choice(value)  # Randomize responses for more dynamic flow

    # If no match is found, save the input to the unrecognized file
    with open(unrecognized_file, 'r+') as f:
        data = json.load(f)
        if normalized_input not in data:  # Avoid duplicate entries
            data.append(normalized_input)
            f.seek(0)
            json.dump(data, f, indent=4)
    
    return random.choice(responses.get("default", ["Sorry, I didn't catch that!"]))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"response": "Say something! I can't read minds... yet."})

    response = get_response(user_input)

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
