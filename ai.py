import re
import random
from datetime import datetime

class ConversationalAI:
    def __init__(self):
        self.memory = {}  # Stores user-specific information and situations
        self.emotional_state = "neutral"  # Tracks the AI's emotional state
        self.user_emotional_state = "neutral"  # Tracks the user's emotional state
        self.dialogue_state = "greeting"  # Tracks the current state of the conversation
        self.user_name = None  # Stores the user's name
        self.last_interaction_time = None  # Tracks the last interaction time
        self.current_topic = None  # Tracks the current topic of conversation

    def update_emotional_state(self, user_input):
        """Updates the AI's and user's emotional state based on user input."""
        positive_words = ["happy", "joy", "love", "excited", "good", "great", "awesome"]
        negative_words = ["sad", "angry", "hate", "bad", "upset", "frustrated", "depressed"]

        # Analyze user input for emotional cues
        if any(word in user_input.lower() for word in positive_words):
            self.user_emotional_state = "happy"
        elif any(word in user_input.lower() for word in negative_words):
            self.user_emotional_state = "sad"
        else:
            self.user_emotional_state = "neutral"

        # Adjust AI's emotional state based on user's emotions
        if self.user_emotional_state == "happy":
            self.emotional_state = random.choice(["happy", "excited"])
        elif self.user_emotional_state == "sad":
            self.emotional_state = random.choice(["concerned", "sad"])
        else:
            self.emotional_state = "neutral"

    def remember_user_info(self, key, value):
        """Stores user-specific information in memory."""
        self.memory[key] = value

    def recall_user_info(self, key):
        """Retrieves user-specific information from memory."""
        return self.memory.get(key, None)

    def remember_situation(self, situation):
        """Stores a situation in memory."""
        if "situations" not in self.memory:
            self.memory["situations"] = []
        self.memory["situations"].append(situation)

    def recall_situations(self):
        """Retrieves all remembered situations."""
        return self.memory.get("situations", [])

    def handle_greeting(self, user_input):
        """Handles the greeting state of the conversation."""
        if not self.user_name:
            name_match = re.search(r"my name is (\w+)", user_input.lower())
            if name_match:
                self.user_name = name_match.group(1)
                self.remember_user_info("name", self.user_name)
                return f"Nice to meet you, {self.user_name}! How can I help you today?"
            else:
                return "Hello! What's your name?"
        else:
            return f"Hi again, {self.user_name}! How can I assist you today?"

    def handle_emotion(self, user_input):
        """Handles emotional responses based on the user's emotional state."""
        if self.user_emotional_state == "sad":
            self.remember_situation(f"{self.user_name} was feeling sad: {user_input}")
            return random.choice([
                f"I'm sorry to hear that you're feeling sad, {self.user_name}. What happened?",
                "That sounds tough. Do you want to talk about it?"
            ])
        elif self.user_emotional_state == "happy":
            self.remember_situation(f"{self.user_name} was feeling happy: {user_input}")
            return random.choice([
                f"That's great to hear, {self.user_name}! What made you so happy?",
                "I'm glad you're feeling happy! Tell me more about it."
            ])
        else:
            return "How are you feeling today?"

    def handle_memory(self, user_input):
        """Handles memory-related queries."""
        if "remember" in user_input.lower():
            key_match = re.search(r"remember (\w+)", user_input.lower())
            value_match = re.search(r"is (\w+)", user_input.lower())
            if key_match and value_match:
                key = key_match.group(1)
                value = value_match.group(1)
                self.remember_user_info(key, value)
                return f"Okay, I'll remember that {key} is {value}."
            else:
                return "I didn't catch that. What should I remember?"
        elif "recall" in user_input.lower():
            key_match = re.search(r"recall (\w+)", user_input.lower())
            if key_match:
                key = key_match.group(1)
                value = self.recall_user_info(key)
                if value:
                    return f"I remember that {key} is {value}."
                else:
                    return f"I don't recall anything about {key}."
            else:
                return "What should I recall?"
        else:
            return None

    def handle_general_conversation(self, user_input):
        """Handles general conversation topics."""
        if "how are you" in user_input.lower():
            return self.handle_emotion(user_input)
        elif "time" in user_input.lower():
            current_time = datetime.now().strftime("%H:%M")
            return f"The current time is {current_time}."
        elif "thank you" in user_input.lower():
            return "You're welcome!"
        elif self.user_emotional_state == "sad":
            return self.handle_emotion(user_input)
        elif self.current_topic:
            return self.continue_topic(user_input)
        else:
            return self.start_new_topic(user_input)

    def start_new_topic(self, user_input):
        """Starts a new topic based on user input."""
        topics = {
            "work": "How is work going?",
            "family": "How is your family doing?",
            "hobby": "What hobbies have you been enjoying lately?",
            "weather": "How's the weather where you are?"
        }
        for topic, response in topics.items():
            if topic in user_input.lower():
                self.current_topic = topic
                return response
        return random.choice([
            "That's interesting! Tell me more.",
            "I see. What else is on your mind?",
            "Hmm, I'm not sure I understand. Can you elaborate?"
        ])

    def continue_topic(self, user_input):
        """Continues the current topic of conversation."""
        if self.current_topic == "work":
            return "How do you feel about your work lately?"
        elif self.current_topic == "family":
            return "What's new with your family?"
        elif self.current_topic == "hobby":
            return "Have you tried anything new in your hobbies recently?"
        elif self.current_topic == "weather":
            return "Has the weather been affecting your plans?"
        else:
            return "Tell me more about that."

    def respond(self, user_input):
        """Generates a response based on the current dialogue state and user input."""
        self.update_emotional_state(user_input)
        self.last_interaction_time = datetime.now()

        # Handle memory-related queries
        memory_response = self.handle_memory(user_input)
        if memory_response:
            return memory_response

        # Dialogue management
        if self.dialogue_state == "greeting":
            response = self.handle_greeting(user_input)
            if self.user_name:
                self.dialogue_state = "general"
            return response
        elif self.dialogue_state == "general":
            return self.handle_general_conversation(user_input)
        else:
            return "I'm not sure how to respond to that."

# Main loop for interacting with the AI
def main():
    ai = ConversationalAI()
    print("AI: Hello! I'm your conversational AI. How can I assist you today?")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("AI: Goodbye! Have a great day!")
            break
        response = ai.respond(user_input)
        print(f"AI: {response}")

if __name__ == "__main__":
    main()