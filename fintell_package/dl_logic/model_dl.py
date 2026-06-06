from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report
from fintell_package.params import MODEL_DIR_DL, MODEL_DIR_DL_PLOTS


def init_model(maxlen, vector_size):
    model = Sequential()
    model.add(layers.Masking(input_shape=(maxlen, vector_size)))
    model.add(layers.LSTM(20, activation='tanh'))
    model.add(layers.Dense(15, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))  # 3 classes, pas 1

    model.compile(loss='sparse_categorical_crossentropy',  # pas binary_crossentropy
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model

def plot_history(history):
    from fintell_package.run_context import RUN_TIMESTAMP
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.set_title('Model loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Val'], loc='upper right')
    ax2.plot(history.history['accuracy'])
    ax2.plot(history.history['val_accuracy'])
    ax2.set_title('Model accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Val'], loc='upper left')
    plot_path = MODEL_DIR_DL_PLOTS / f'training_history_{RUN_TIMESTAMP}.png'
    plt.savefig(plot_path)
    print(f"Final loss: {history.history['loss'][-1]:.4f} | val_loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Final accuracy: {history.history['accuracy'][-1]:.4f} | val_accuracy: {history.history['val_accuracy'][-1]:.4f}")
    return plot_path

def train_model(X_train, y_train, X_val, y_val, model):
    print(model.summary())

    es = EarlyStopping(patience=5, restore_best_weights=True)
    mc = ModelCheckpoint(MODEL_DIR_DL / 'fintell_lstm_w2v.keras', save_best_only=True, monitor='val_accuracy')

    history = model.fit(X_train, y_train,
          batch_size=32,
          epochs=100,
          validation_data=(X_val, y_val),
          callbacks=[es, mc]
         )

    plot_path = plot_history(history)

    return model, history, plot_path


def evaluate_model(X, y, model):
    y_pred = model.predict(X).argmax(axis=1)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    report = classification_report(y, y_pred)
    return accuracy, f1, report


def predict_model(X_new, model):
    return model.predict(X_new)
