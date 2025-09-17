¡Gracias por compartir todo con tanto detalle! Voy a:

1. **Diagnosticar** por qué tu precisión se queda en \~74%.
2. Aplicar la **estrategia de los 5 porqués**.
3. Proponer una **solución estratégica** (arquitectura + preprocesado + entrenamiento) para llevarte al **85–87%**.
4. Dejarte un **notebook por celdas**, con **imports limpios y consistentes** (sin `!pip`, `!python ...`, ni mezclas entre `keras` y `tensorflow.keras`), listo para Colab.

---

## Observaciones rápidas sobre tus resultados

* Estás imprimiendo en la **celda 10** la *precisión de entrenamiento* del último epoch (`hist.history['accuracy'][-1]`), que no refleja el mejor punto. En cambio, la **celda 9** evalúa con `restore_best_weights=True` y te da **73.85%** real en *validación*.
* El **summary “unbuilt”** en tu celda 7 ocurre por cómo Keras 3 construye el modelo; no es error, pero confunde. Se corrige con `model.build((None, max_len))` antes de `model.summary()`.
* Estás mezclando `from keras...` y `from tensorflow.keras...`. Eso **rompe compatibilidad** en Keras 3/TensorFlow 2.x. Conviene **usar siempre `tf.keras`**.
* Tienes **514 clases** con \~12.6k patrones (≈24 por clase de media, y varias muy escasas). Hay **alto desbalance** y el modelo actual (Embedding + *GlobalAveragePooling*) es **demasiado simple** para capturar señales finas con **max\_len=20** y **vocab=3000**.

---

## Los 5 Porqués (root cause)

1. **¿Por qué** no llegas al 85–87%?
   Porque el modelo pierde información contextual: usa `GlobalAveragePooling1D` (promedio) con `embedding_dim=50`, `max_len=20` y `vocab=3000`, lo que **aplana excesivamente** la señal.

2. **¿Por qué** se pierde info contextual?
   Porque la arquitectura **no modela n‑gramas**/patrones locales (no hay convs ni RNNs) y el *max\_len* es **corto** para cubrir variaciones.

3. **¿Por qué** el *val acc* luce mejor que el *train acc* de la última época?
   Porque estás leyendo la **última época** en train (no la mejor) y la validación usa **mejores pesos** por *early stopping*. Además, puede haber **leakage ligero** si tokenizas con todo el corpus antes de separar (tu `Tokenizer` se ajusta con todos los textos).

4. **¿Por qué** podría haber leakage/desbalance?
   Porque no aplicas **split estratificado robusto** (controlando clases raras) ni **pesos de clase**. En 514 clases, las raras tiran el rendimiento.

5. **¿Por qué** no se corrige con los artefactos guardados (pkl)?
   Porque hay **inconsistencias**: guardas `words.pkl/classes.pkl` de un pipeline BoW que **no usas** después, y las etiquetas del modelo van por **LabelEncoder** (otro mapeo). Eso complica la inferencia/depuración.

---

## Estrategia propuesta (para alcanzar 85–87%)

1. **Higiene de datos y split**

   * Deduplicar `(texto, etiqueta)`.
   * **Tokenizer entrenado solo con TRAIN** (evita fuga).
   * Split **estratificado por clase** que garantice al menos 1 ejemplo en train para clases pequeñas; si una clase tiene solo 1 ejemplo, va a TRAIN.

2. **Representación**

   * Aumentar **vocabulario** a `MAX_WORDS=20000`.
   * Definir **longitud de secuencia** por **percentil 95** de TRAIN (p.ej. en la práctica suele quedar entre 24 y 64).
   * (Opcional) Normalizaciones leves del texto (espacios, trims), sin depender de lematizadores en español para no añadir fragilidad.

3. **Modelo**

   * Cambiar a **TextCNN** multikernel (3,4,5) + `SpatialDropout1D` + `GlobalMaxPooling` + `BatchNorm` + densas con `Dropout`. Esta arquitectura captura n‑gramas y suele ganar **10–15 puntos** sobre *bag of words* o *average embeddings* en *intent classification*.
   * `embedding_dim=128`, filtros 128–256, `label_smoothing≈0.05`, `l2` leve.

4. **Entrenamiento**

   * **Class weights** con `compute_class_weight` para combatir el desbalance.
   * Callbacks fuertes: `EarlyStopping(monitor='val_accuracy')`, `ReduceLROnPlateau(monitor='val_loss')`, `ModelCheckpoint` por `val_accuracy`.
   * Batch 64, LR 3e‑4.

5. **Evaluación**

   * Reportar **VAL accuracy** del **mejor checkpoint** (no el último epoch).
   * Añadir **Top‑3 accuracy** y `classification_report` para ver clases débiles.
   * Guardar `tokenizer.pkl`, `label_encoder.pkl` y `artifacts.json` para inferencia consistente.

> Si aún faltan 1–2 puntos, añade un **BiGRU** (opcional) y haz **ensemble suave** (0.6*TextCNN + 0.4*BiGRU). En muchos datasets grandes de intenciones esto empuja al objetivo.

---

## Notebook por celdas (todo consistente, sin `!pip`, sin `!python -m ...`)

> **Nota:** Usa **siempre** `tf.keras`. He incluido descargas silenciosas de NLTK sólidas (sin `!`) por si luego quieres experimentar; no dependemos de ellas.

---

### **Celda 1 — Imports sólidos + semillas + versiones**

```python
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
```

---

### **Celda 2 — Carga del dataset (JSON tipo intents)**

```python
# Si necesitas subir manualmente, descomenta:
# from google.colab import files
# uploaded = files.upload()

DATA_PATH = 'nexus_data_17k.json'  # cambia el nombre si tu archivo se llama distinto
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
```

---

### **Celda 3 — Limpieza, deduplicación y resumen**

```python
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
```

---

### **Celda 4 — Partición estratificada robusta (sin fuga de info)**

```python
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
```

---

### **Celda 5 — Tokenización (ajuste solo con TRAIN)**

```python
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

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("MAX_LEN:", MAX_LEN, "| Vocab usado (<=MAX_WORDS):", min(len(tokenizer.word_index)+1, MAX_WORDS))
```

---

### **Celda 6 — Etiquetas + Pesos de clase**

```python
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

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
```

---

### **Celda 7 — Modelo TextCNN (recomendado)**

```python
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
```

---

### **Celda 8 — Entrenamiento (callbacks fuertes)**

```python
early_stop = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=12,
                                     restore_best_weights=True, verbose=1)
reduce_lr  = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                         min_lr=1e-6, verbose=1)
checkpoint = callbacks.ModelCheckpoint('chatbot_textcnn.keras', monitor='val_accuracy',
                                       mode='max', save_best_only=True, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=64,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)
```

---

### **Celda 9 — Evaluación correcta (mejor checkpoint)**

```python
best_model = keras.models.load_model('chatbot_textcnn.keras', compile=False)
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
```

---

### **Celda 10 — Artefactos + función de inferencia**

```python
ARTIFACTS = {
    "model_path": "chatbot_textcnn.keras",
    "tokenizer_path": "tokenizer.pkl",
    "label_encoder_path": "label_encoder.pkl",
    "max_len": MAX_LEN,
    "max_words": MAX_WORDS
}
with open('artifacts.json', 'w', encoding='utf-8') as f:
    json.dump(ARTIFACTS, f, ensure_ascii=False, indent=2)

def predict_intent(texts: List[str], model=None, tokenizer=None, le=None, max_len:int=None, top_k:int=3):
    if model is None:
        model = keras.models.load_model(ARTIFACTS["model_path"], compile=False)
    if tokenizer is None:
        with open(ARTIFACTS["tokenizer_path"], 'rb') as f:
            tokenizer = pickle.load(f)
    if le is None:
        with open(ARTIFACTS["label_encoder_path"], 'rb') as f:
            le = pickle.load(f)
    if max_len is None:
        max_len = ARTIFACTS["max_len"]
    X = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=max_len, padding='post', truncating='post')
    probs = model.predict(X, verbose=0)
    top_idxs = np.argsort(-probs, axis=1)[:, :top_k]
    results = []
    for i, idxs in enumerate(top_idxs):
        intents = le.inverse_transform(idxs)
        confs = probs[i, idxs].tolist()
        results.append(list(zip(intents, [float(c) for c in confs])))
    return results

# Ejemplo de uso
print(predict_intent(["hola", "necesito soporte de facturación"], top_k=3))
```

---

### **Celda 11 (opcional) — BiGRU + ensemble (si necesitas el empujón final)**

```python
def build_bigru(max_words, max_len, num_classes, embed_dim=128, rnn_units=128, l2_reg=1e-4):
    inputs = keras.Input(shape=(max_len,), dtype='int32')
    x = layers.Embedding(max_words, embed_dim, input_length=max_len)(inputs)
    x = layers.SpatialDropout1D(0.2)(x)
    x = layers.Bidirectional(layers.GRU(rnn_units, return_sequences=True))(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs, name='BiGRU_Intents')
    model.compile(optimizer=optimizers.Adam(3e-4),
                  loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
                  metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')])
    model.build((None, max_len))
    return model

# Entrenamiento opcional del BiGRU:
# bigru = build_bigru(MAX_WORDS, MAX_LEN, num_classes)
# bigru.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=40, batch_size=64,
#           class_weight=class_weights, callbacks=[callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=8, restore_best_weights=True)], verbose=1)

# Ensemble (si entrenaste ambos):
# p1 = best_model.predict(X_val, verbose=0)  # TextCNN
# p2 = bigru.predict(X_val, verbose=0)       # BiGRU
# p_ens = 0.6*p1 + 0.4*p2
# y_pred_ens = p_ens.argmax(axis=1)
# print("VAL Accuracy (ensemble):", accuracy_score(y_val_int, y_pred_ens)*100)
```

---

## Ajustes finos (si necesitas afinar a tu dataset)

* **MAX\_WORDS**: prueba 30k–40k si ves muchas `<OOV>`.
* **MAX\_LEN**: si p95 queda muy bajo, fija 48–64.
* **TextCNN**: sube `conv_filters` a 256 y `embed_dim` a 256 si no hay OOM.
* **Regularización**: `dropout_dense` 0.6 si hay sobreajuste.
* **LR**: baja a `2e-4` si el *val loss* oscila.
* **Clases raras**: revisa el `classification_report` y, si procede, añade ejemplos o agrupa sinónimos de intención (si tu negocio lo permite).

---

## Qué cambia respecto a tu código original

* **Unificación de imports**: ahora **solo `tf.keras`** (evita choques Keras 3).
* **Split robusto** por clase y **tokenizador entrenado solo con TRAIN** (sin fuga).
* **Arquitectura TextCNN** con **SpatialDropout1D** + **GlobalMaxPooling** + **BatchNorm** (mejor señal que el promedio).
* **Pesos de clase + label smoothing** para lidiar con 514 clases desbalanceadas.
* **Evaluación** sobre el **mejor checkpoint**, no el último epoch.
* **Artefactos consistentes** (`tokenizer.pkl`, `label_encoder.pkl`, `artifacts.json`) para inferencia estable.

Con estos cambios, en problemas de *intent classification* con muchas clases, es frecuente ver saltos de **\~10–15 puntos** respecto a un baseline de embeddings + average pooling, lo que te debería situar dentro del rango **85–87%** si tus patrones por clase son razonablemente consistentes y no hay demasiadas etiquetas con 1–2 ejemplos.

Si te interesa, luego podemos añadir **data augmentation ligera** (e.g., *random deletion/swap* libre de idioma) y/o **ensemble** con el BiGRU para rascar puntos extra sin introducir dependencias nuevas.

¿Quieres que también te deje una celda para **matriz de confusión** y gráficos de aprendizaje?
