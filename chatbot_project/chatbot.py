import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load intents
with open("intents.json") as file:
    data = json.load(file)

patterns = []
tags = []
responses = {}

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])
    responses[intent["tag"]] = intent["responses"]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

# Chat loop
print("Chatbot is running! Type 'quit' to stop.")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        break

    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, X)
    best_match = similarity.argmax()

    tag = tags[best_match]
    reply = random.choice(responses[tag])

    print("Bot:", reply)