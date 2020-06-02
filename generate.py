import tensorflow as tf
import numpy as np

next_words = 100
seed = "Help me Obi Wan Kenobi, you're my only hope"

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed])[0]
    token_list = pad_sequences(token_list, padding='pre', maxlen=max_len)
    predicted = model.predict(token_list, verbose=0)
    output_word = ""
    for word, index in word_index.items():
        if index == predicted:
            output_word = word
            break
    seed += output_word
print("Generated Text\n", seed)
