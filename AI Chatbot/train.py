import json
import random
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load intents
with open('intents.json', 'r') as file:
    intents = json.load(file)

words = []
classes = []
documents = []

ignore_letters = ['?', '!', '.', ',']

# Process intents
for intent in intents['intents']:
    for pattern in intent['patterns']:

        word_list = nltk.word_tokenize(pattern)

        words.extend(word_list)

        documents.append((word_list, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatization
words = [
    lemmatizer.lemmatize(word.lower())
    for word in words
    if word not in ignore_letters
]

words = sorted(set(words))
classes = sorted(set(classes))

# Save vocabulary
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []

output_empty = [0] * len(classes)

for document in documents:

    bag = []

    pattern_words = [
        lemmatizer.lemmatize(word.lower())
        for word in document[0]
    ]

    for word in words:
        bag.append(1 if word in pattern_words else 0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)

training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Build model
model = Sequential()

model.add(Dense(
    128,
    input_shape=(len(train_x[0]),),
    activation='relu'
))

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(
    len(train_y[0]),
    activation='softmax'
))

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Train model
model.fit(
    train_x,
    train_y,
    epochs=200,
    batch_size=5,
    verbose=1
)

# Save model
model.save('chatbot_model.h5')

print("Model trained successfully!")