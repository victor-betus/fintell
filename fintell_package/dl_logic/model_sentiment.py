from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, Callback
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report
from fintell_package.params import MODEL_DIR_DL, MODEL_DIR_DL_PLOTS, GCS_PROJECT_ID, GCS_BUCKET_NAME, MODEL_TARGET, MODEL_DL_NAME, USE_CLASS_WEIGHT, EMBEDDER_NAME
from fintell_package.run_context import RUN_TIMESTAMP
from fintell_package.data import upload_file_to_bucket
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

class GCSCheckpoint(Callback):
    def __init__(self, local_path):
        super().__init__()
        self.local_path = local_path
        self.best_val_accuracy = 0

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy', 0)
        if val_acc > self.best_val_accuracy:
            self.best_val_accuracy = val_acc
            self.model.save(str(self.local_path))
            if MODEL_TARGET == "gcs":
                upload_file_to_bucket(
                    GCS_PROJECT_ID, GCS_BUCKET_NAME,
                    str(self.local_path),
                    f"dl_checkpoints/best_model_{RUN_TIMESTAMP}.keras"
                )
                print(f"✅ Best model uploaded to GCS (val_acc: {val_acc:.4f})")


def init_model(maxlen, vector_size, model_dl_name=MODEL_DL_NAME):

    if model_dl_name == 'lstm':
        model = Sequential()
        model.add(layers.Masking(input_shape=(maxlen, vector_size)))
        model.add(layers.LSTM(20, activation='tanh'))
        model.add(layers.Dense(15, activation='relu'))
        model.add(layers.Dense(3, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

    elif model_dl_name == 'gru':
        model = Sequential()
        model.add(layers.Masking(input_shape=(maxlen, vector_size)))
        model.add(layers.GRU(20, activation='tanh'))
        model.add(layers.Dense(15, activation='relu'))
        model.add(layers.Dense(3, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])


    elif model_dl_name == 'bigru':
        from tensorflow.keras.regularizers import l2
        from tensorflow.keras.optimizers import Adam

        model = Sequential()
        model.add(layers.Masking(input_shape=(maxlen, vector_size)))
        model.add(layers.Bidirectional(layers.GRU(128, dropout=0.2)))
        model.add(layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(3, activation='softmax'))

        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    elif model_dl_name == 'bilstm':
        model = Sequential()
        model.add(layers.Masking(input_shape=(maxlen, vector_size)))
        model.add(layers.Bidirectional(layers.LSTM(20, activation='tanh')))
        model.add(layers.Dense(15, activation='relu'))
        model.add(layers.Dense(3, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

    else:
        raise ValueError(f"Unknown model_dl_name: '{model_dl_name}'. Choose from: lstm, gru, bilstm")
    return model


def plot_history(history):
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

    checkpoint_path = MODEL_DIR_DL / f'fintell_{MODEL_DL_NAME}_{EMBEDDER_NAME}.keras'
    if USE_CLASS_WEIGHT:
        weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        cw = dict(enumerate(weights))
    else:
        cw = None

    es = EarlyStopping(patience=5, restore_best_weights=True)
    gcs_ckpt = GCSCheckpoint(checkpoint_path)

    history = model.fit(X_train, y_train,
        batch_size=64,
        epochs=100,
        validation_data=(X_val, y_val),
        callbacks=[es, gcs_ckpt],
        class_weight=cw,
        verbose=2
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
