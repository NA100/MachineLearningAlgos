from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

texts = ["I love this!", "Worst ever", "Amazing product", "Not good", "Happy with it"]
labels = [1, 0, 1, 0, 1]  # 1 = positive, 0 = negative

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, padding='post', maxlen=4)

model = Sequential([
    Embedding(input_dim=100, output_dim=16, input_length=4),
    GlobalAveragePooling1D(),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded, labels, epochs=30, verbose=0)

# Test
test_text = ["Really bad"]
test_seq = tokenizer.texts_to_sequences(test_text)
test_pad = pad_sequences(test_seq, padding='post', maxlen=4)
print("Sentiment:", "Positive" if model.predict(test_pad)[0][0] > 0.5 else "Negative")
