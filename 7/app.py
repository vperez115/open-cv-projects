import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
import os
from tkinter import Tk, filedialog
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import LossScaleOptimizer

# Enable mixed precision globally
set_global_policy('mixed_float16')
tf.config.experimental.enable_tensor_float_32_execution(False)


# Global Variables
MAX_WORDS = 10000  # Maximum number of words to keep based on frequency
MAX_LEN = 100  # Maximum length of sequences (padding)
MODEL_PATH = "sentiment_model.h5"  # Model save path
TOKENIZER_PATH = "tokenizer.pkl"  # Tokenizer save path

def open_file_dialog():
    """Open a file dialog to allow the user to select a CSV file."""
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window
    file_path = filedialog.askopenfilename(
        title="Select a CSV File",
        filetypes=[("CSV Files", "*.csv")]
    )
    return file_path

def load_dataset(file_path):
    """Load and preprocess the Sentiment140 dataset."""
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1', header=None)
    except UnicodeDecodeError:
        print("Error reading the file. Please check the encoding.")
        return None, None, None

    print(f"Loaded dataset from {file_path}")

    df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    texts = df['text'].astype(str).values
    labels = df['target'].values

    labels = np.where(labels == 4, 1, 0)

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    return padded_sequences, labels, tokenizer

def create_model(num_classes=1):
    """Create an LSTM model for sentiment analysis."""
    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
        Bidirectional(LSTM(64, return_sequences=True)),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
    ])

    # Use mixed precision compatible optimizer with loss scaling
    opt = Adam(learning_rate=0.001)
    opt = LossScaleOptimizer(opt)
    
    model.compile(optimizer=opt, 
                  loss='binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_model():
    """Train the model using the Sentiment140 dataset."""
    file_path = open_file_dialog()
    if not file_path or not os.path.exists(file_path):
        print("File not found or no file selected.")
        return

    X, y, tokenizer = load_dataset(file_path)
    if X is None:
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = create_model()
    print("Training the model...")

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=256,
        verbose=1
    )

    model.save(MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")

    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved at {TOKENIZER_PATH}")

def load_or_train_model():
    """Load an existing model or train a new one."""
    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        print("Loading existing model and tokenizer...")
        model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    else:
        print("No existing model found. Training a new model...")
        train_model()
        model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer

def test_model():
    """Test the trained model on a new dataset or single input."""
    model, tokenizer = load_or_train_model()

    while True:
        choice = input("Do you want to test with a file? (yes/no): ").strip().lower()
        
        if choice == 'yes':
            file_path = open_file_dialog()
            if not file_path or not os.path.exists(file_path):
                print("File not found or no file selected.")
                continue  # Go back to the main loop and allow the user to choose again

            df = pd.read_csv(file_path, encoding='ISO-8859-1', header=None)
            df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
            texts = df['text'].astype(str).values

            sequences = tokenizer.texts_to_sequences(texts)
            padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

            predictions = model.predict(padded_sequences)
            df['predicted_sentiment'] = (predictions > 0.5).astype(int)

            print(df[['text', 'predicted_sentiment']].head())

        else:
            while True:
                review = input("Enter a review to test (or type 'exit' to stop): ").strip()
                if review.lower() == 'exit':
                    print("Exiting review testing...")
                    return  # Exit the function and stop testing

                sequence = tokenizer.texts_to_sequences([review])
                padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

                prediction = model.predict(padded_sequence)[0][0]
                sentiment = "Positive" if prediction > 0.5 else "Negative"
                print(f"Predicted Sentiment: {sentiment} with confidence {prediction:.2f}")

                another_review = input("Do you want to test another review? (yes/no): ").strip().lower()
                if another_review != 'yes':
                    print("Stopping individual review testing...")
                    return  # Exit the function if the user does not want to test another review


def main():
    """Main function to run the program."""
    action = input("Do you want to train the model? (yes/no): ").strip().lower()
    if action == 'yes':
        train_model()
    else:
        test_model()

if __name__ == "__main__":
    main()
