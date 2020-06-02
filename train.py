import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, Embedding
import tensorflow.keras.utils as ku

with open("./irish-lyrics-eof.txt", "r") as f:
    lyrics = f.read()

lyrics = lyrics.lower().split("\n")

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(lyrics)

word_index = tokenizer.word_index

total_words = len(word_index) + 1

input_sequences = []

for line in lyrics:
    sequence = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(sequence)):
        n_gram_sequence = sequence[:i+1]
        input_sequences.append(n_gram_sequence)

max_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences,
                                maxlen=max_len,
                                padding='pre')

x, y = input_sequences[:, :-1], ku.to_categorical(input_sequences[:, -1],
                                                  num_classes=total_words)

TRAIN = False
if TRAIN:
    model = Sequential()
    model.add(Embedding(total_words, 100, input_shape=(max_len - 1,)))
    model.add(Bidirectional(LSTM(150, activation=tf.nn.relu, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dense(64, activation=tf.nn.relu))
    model.add(Dense(total_words, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    print(model.summary())

    history = model.fit(x, y, epochs=100, verbose=1)
    model.save("./models/generator.h5")
    ####train_loss = history.history['loss']
    ####train_accuracy = history.history['acc']
else:

    model = tf.keras.models.load_model("./models/generator.h5")
    next_words = 100
    seed = "rise with the"

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed])[0]
        token_list = pad_sequences([token_list], padding='pre', maxlen=max_len-1)
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in word_index.items():
            if index == predicted:
                output_word = word
                break
        seed += " " + output_word
    print("Generated Text\n", seed)

