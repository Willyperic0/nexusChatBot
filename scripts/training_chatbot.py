# --- Imports sólidos y consistentes ---
from __future__ import annotations
import os, re, json, math, random, pickle, sys
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

# TensorFlow / Keras (usar SIEMPRE tf.keras para evitar incompatibilidades)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers, optimizers

# Sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, top_k_accuracy_score

# NLTK (opcional: no obligatorio en este pipeline)
import nltk
for pkg in ["punkt", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg=="punkt" else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# Semillas para reproducibilidad
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)

DATA_PATH = os.path.join("data", "nexus_data.json")
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    raw = json.load(f)

assert 'intents' in raw, "El JSON debe tener la clave 'intents'."

texts, labels = [], []
for it in raw['intents']:
    tag = it.get('tag')
    for patt in it.get('patterns', []):
        if isinstance(patt, str) and patt.strip():
            texts.append(patt.strip())
            labels.append(tag)

print(f"Textos cargados: {len(texts)}  |  Clases únicas: {len(set(labels))}")

df = pd.DataFrame({'text': texts, 'label': labels})
df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Elimina duplicados exactos (texto + etiqueta)
before = len(df)
df = df.drop_duplicates(subset=['text', 'label'])
after = len(df)
print(f"Duplicados eliminados: {before - after}")

# Recuento por clase
label_counts = df['label'].value_counts().sort_values(ascending=False)
print("Clases totales:", label_counts.shape[0])
print("Mín/Mediana/Media/Máx ejemplos por clase:",
      int(label_counts.min()), int(label_counts.median()),
      round(label_counts.mean(), 2), int(label_counts.max()))

df.head(3)

def stratified_split_robust(df, test_size=0.2, seed=SEED):
    """
    Garantiza que clases con 1 ejemplo queden en TRAIN.
    Toma al menos 1 instancia a VAL si la clase tiene >=2 ejemplos.
    """
    train_idx, val_idx = [], []
    rng = np.random.default_rng(seed)
    for label, group in df.groupby('label'):
        idx = group.index.to_list()
        n = len(idx)
        if n <= 1:
            train_idx.extend(idx)
            continue
        n_val = max(1, int(round(n * test_size)))
        n_val = min(n - 1, n_val)  # deja al menos 1 en TRAIN
        val_choice = rng.choice(idx, size=n_val, replace=False)
        train_choice = list(set(idx) - set(val_choice))
        train_idx.extend(train_choice)
        val_idx.extend(val_choice)
    return df.loc[train_idx].reset_index(drop=True), df.loc[val_idx].reset_index(drop=True)

train_df, val_df = stratified_split_robust(df, test_size=0.2, seed=SEED)
print(f"Train: {len(train_df)} | Val: {len(val_df)} | #clases train: {train_df['label'].nunique()} | #clases val: {val_df['label'].nunique()}")

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Longitud máxima por percentil 95 (cap entre 24 y 64)
lengths = train_df['text'].str.split().apply(len).values
p95 = int(np.percentile(lengths, 95))
MAX_LEN = max(24, min(64, p95))
MAX_WORDS = 20000

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(train_df['text'].tolist())  # SOLO TRAIN

def to_pad(seqs, maxlen=MAX_LEN):
    return pad_sequences(tokenizer.texts_to_sequences(seqs), maxlen=maxlen, padding='post', truncating='post')

X_train = to_pad(train_df['text'].tolist(), MAX_LEN)
X_val   = to_pad(val_df['text'].tolist(), MAX_LEN)

with open(os.path.join("artifacts", "tokenizer.pkl"), 'wb') as f:
    pickle.dump(tokenizer, f)


print("MAX_LEN:", MAX_LEN, "| Vocab usado (<=MAX_WORDS):", min(len(tokenizer.word_index)+1, MAX_WORDS))

le = LabelEncoder()
y_train_int = le.fit_transform(train_df['label'].tolist())
y_val_int   = le.transform(val_df['label'].tolist())

num_classes = len(le.classes_)
y_train = keras.utils.to_categorical(y_train_int, num_classes=num_classes)
y_val   = keras.utils.to_categorical(y_val_int, num_classes=num_classes)

classes_array = np.arange(num_classes)
class_weights_raw = compute_class_weight(
    class_weight='balanced',
    classes=classes_array,
    y=y_train_int
)
class_weights = {i: float(w) for i, w in enumerate(class_weights_raw)}
print("Ejemplo pesos (primeros 10):", dict(list(class_weights.items())[:10]))

with open(os.path.join("artifacts", "label_encoder.pkl"), 'wb') as f:
    pickle.dump(le, f)


def build_textcnn(max_words: int, max_len: int, num_classes: int,
                  embed_dim: int = 128, conv_filters: int = 128,
                  kernel_sizes=(3,4,5), dropout_dense: float = 0.5,
                  l2_reg: float = 1e-4) -> keras.Model:
    inputs = keras.Input(shape=(max_len,), dtype='int32')
    x = layers.Embedding(input_dim=max_words, output_dim=embed_dim, input_length=max_len, name='embedding')(inputs)
    x = layers.SpatialDropout1D(0.2)(x)
    convs = []
    for k in kernel_sizes:
        c = layers.Conv1D(filters=conv_filters, kernel_size=k, padding='valid', activation='relu',
                          kernel_regularizer=regularizers.l2(l2_reg))(x)
        c = layers.GlobalMaxPooling1D()(c)
        convs.append(c)
    x = layers.Concatenate()(convs) if len(convs) > 1 else convs[0]
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout_dense)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout_dense)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='TextCNN_Intents')
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    opt = optimizers.Adam(learning_rate=3e-4)
    model.compile(optimizer=opt, loss=loss,
                  metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')])
    model.build((None, max_len))
    return model

model = build_textcnn(MAX_WORDS, MAX_LEN, num_classes)
model.summary()

early_stop = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=12,
                                     restore_best_weights=True, verbose=1)
reduce_lr  = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                         min_lr=1e-6, verbose=1)
checkpoint = callbacks.ModelCheckpoint(
    os.path.join("artifacts", "chatbot_textcnn.keras"),
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)


history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=64,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

best_model = keras.models.load_model(os.path.join("artifacts", "chatbot_textcnn.keras"), compile=False)
best_model.compile(
    optimizer=optimizers.Adam(3e-4),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.0),
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
)

val_loss, val_acc, val_top3 = best_model.evaluate(X_val, y_val, verbose=0)
print(f"VAL Accuracy: {val_acc*100:.2f}% | VAL Top-3: {val_top3*100:.2f}% | VAL Loss: {val_loss:.4f}")

# Reporte por clase (puede tardar en 514 clases)
y_val_pred = best_model.predict(X_val, verbose=0)
y_val_pred_int = np.argmax(y_val_pred, axis=1)
print(classification_report(y_val_int, y_val_pred_int, digits=3))

# Guarda el modelo entrenado en un archivo h5
model.save(os.path.join("artifacts", "chatbot_model.h5"))

