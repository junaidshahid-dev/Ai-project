"""Models: classical baselines and the paper's 1D-CNN, CNN_Bi-LSTM and averaging
ensemble (Chowdhury et al. 2025, Tables 2 and 3)."""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DROPOUT = 0.2


def baseline_models(seed):
    return {
        "Majority": DummyClassifier(strategy="most_frequent"),
        "SVM": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True, random_state=seed)),
        "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed),
    }


def _block(x, filters, kernel, dropout):
    x = layers.Conv1D(filters, kernel, strides=1, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2, padding="same")(x)
    return layers.Dropout(DROPOUT)(x) if dropout else x


def _build(input_len, n_classes, bilstm):
    """Table 2 of the paper. Blocks 1 and 3 have no dropout; kernel 5 for the
    first three convolutions and 3 for the rest. The CNN_Bi-LSTM variant adds a
    Bi-directional LSTM (64 units per direction) before the last conv block."""
    inp = layers.Input(shape=(input_len, 1))
    x = _block(inp, 128, 5, dropout=False)
    x = _block(x, 128, 5, dropout=True)
    x = _block(x, 64, 5, dropout=False)
    x = _block(x, 64, 3, dropout=True)
    if bilstm:
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = _block(x, 32, 3, dropout=True)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return models.Model(inp, out, name="CNN_BiLSTM" if bilstm else "CNN")


DEEP_BUILDERS = {
    "CNN": lambda n, c: _build(n, c, bilstm=False),
    "CNN_BiLSTM": lambda n, c: _build(n, c, bilstm=True),
}


def train_deep(name, X_tr, y_tr, X_val, y_val, n_classes, seed, epochs=100, patience=5, verbose=0):
    """Training settings from Table 3: Adam (lr 1e-3), batch 64, up to 100
    epochs, early stopping on val_accuracy (patience 5) and LR halving on
    val_accuracy plateau (patience 3, min 1e-5). `patience` is exposed only for
    the no-augmentation control in the ablation (see run_experiments.py)."""
    tf.keras.utils.set_random_seed(seed)
    model = DEEP_BUILDERS[name](X_tr.shape[1], n_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cbs = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-5),
    ]
    hist = model.fit(X_tr[..., None], y_tr, validation_data=(X_val[..., None], y_val),
                     epochs=epochs, batch_size=64, callbacks=cbs, verbose=verbose)
    return model, hist.history


def predict_deep(model, X):
    return model.predict(X[..., None], batch_size=256, verbose=0)


def ensemble_proba(probas):
    """Averaging ensemble: mean of the member models' class probabilities
    (equivalent to the paper's Average layer at inference time)."""
    return np.mean(np.stack(probas), axis=0)
