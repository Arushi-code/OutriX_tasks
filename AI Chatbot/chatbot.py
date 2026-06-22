import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Load model and data
model = load_model('chatbot_model.h5')

with open('intents.json', 'r') as file:
    intents = json.load(file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)

    sentence_words = [
        lemmatizer.lemmatize(word.lower())
        for word in sentence_words
    ]

    return sentence_words

def bag_of_words(sentence):

    sentence_words = clean_up_sentence(sentence)

    bag = [0] * len(words)

    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1

    return np.array(bag)

def predict_class(sentence):

    bow = bag_of_words(sentence)

    result = model.predict(
        np.array([bow]),
        verbose=0
    )[0]

    max_index = np.argmax(result)

    return classes[max_index]

def get_response(intent_name):

    for intent in intents['intents']:
        if intent['tag'] == intent_name:
            return random.choice(intent['responses'])

    return "Sorry, I don't understand."

def chatbot_response(text):

    intent = predict_class(text)

    return get_response(intent)