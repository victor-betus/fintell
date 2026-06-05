import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import text_to_word_sequence

def tokenizer(X):
    X_tok = X.apply(text_to_word_sequence).tolist()
    print(f"Tokenized {len(X_tok)} reviews | avg length: {int(np.mean([len(s) for s in X_tok]))}")
    return X_tok
