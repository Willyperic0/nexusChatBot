# chatbot.py
import os
import json
import pickle
import re
import random
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from difflib import get_close_matches

# =============================
# Config
# =============================
NN_THRESHOLD = 0.50
SEM_THRESHOLD = 0.40
MAX_LEN = 24  # coincide con el entrenamiento
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =============================
# Paths
# =============================
def _base_dir() -> Path:
    try:
        return Path(__file__).parent.resolve()
    except NameError:
        return Path.cwd().resolve()

BASE = _base_dir()
INTENTS_PATH = (BASE / "nexus_data.patched.json") if (BASE / "nexus_data.patched.json").exists() else (BASE / "nexus_data.json")
TOKENIZER_PATH = BASE / "tokenizer.pkl"
LABELENC_PATH = BASE / "label_encoder.pkl"
MODEL_PATH = BASE / "chatbot_model.h5"

# =============================
# Utils
# =============================
def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

# =============================
# Normalización
# =============================
_re_multi_ws = re.compile(r"\s+")
_re_nonword = re.compile(r"[^\w\s]")
_dup_re = re.compile(r"(.)\1+")

def unidecode_es(s: str) -> str:
    tr = str.maketrans("áéíóúÁÉÍÓÚñÑ", "aeiouAEIOUnN")
    return s.translate(tr)

def squash_dupes(s: str) -> str:
    return _dup_re.sub(r"\1", s)

def light_phonetic_norm(s: str) -> str:
    t = s
    t = re.sub(r"sh", "ch", t)
    t = re.sub(r"\bh", "", t)
    t = re.sub(r"\bqu", "k", t)
    t = re.sub(r"\b(c|k)ue", "ke", t)
    t = re.sub(r"\b(c|k)ui", "ki", t)
    return t

def tokenize(s: str) -> List[str]:
    t = unidecode_es(s).lower().strip()
    t = squash_dupes(t)
    t = light_phonetic_norm(t)
    t = _re_nonword.sub(" ", t)
    t = _re_multi_ws.sub(" ", t).strip()
    return t.split()

def normalize_text(s: str) -> str:
    t = apply_alias_and_syns(s)
    t = light_phonetic_norm(t)
    t = _re_nonword.sub(" ", t)
    t = _re_multi_ws.sub(" ", t).strip()
    return t

# =============================
# Sinónimos y alias
# =============================
SYNONYMS: Dict[str, List[str]] = {
    "dime": ["muestrame","indícame","quiero saber","explícame","cuéntame","enséñame","recuérdame"],
    "información": ["info","detalles","datos","explicación","resumen","descripción","características"],
    "qué": ["cuál","cuales","que","cuáles"],
    "héroes": ["personajes","clases","heroes","jugables"],
    "ítems": ["objetos","productos","items","elementos"],
    "misiones": ["retos","desafíos","quests","tareas"],
    "subasta": ["subastas","pujar","puja","auction","subaasta","subastaa"],
    "créditos": ["dinero","moneda","creditos"],
    "chaman":  ["shaman","chamán","changua","te_changua","chamn","chamann"],
    "picaro":  ["píc4ro","picar0","pic4ro","pícaro"],
}
MULTI_ALIAS = {
    "te changua": "te_changua",
    "compra inmediata": "compra_inmediata",
    "golpe con escudo": "golpe_con_escudo",
    "mano de piedra": "mano_de_piedra",
    "defensa feroz": "defensa_feroz",
    "guerrero tanque": "guerrero_tanque",
    "guerrero armas":  "guerrero_armas",
    "mago fuego":      "mago_fuego",
    "mago hielo":      "mago_hielo",
    "picaro machete":  "picaro_machete",
    "pícaro machete":  "picaro_machete",
    "picaro veneno":   "picaro_veneno",
    "pícaro veneno":   "picaro_veneno",
}
SYN_PATTS  = [(re.compile(rf"\b{re.escape(unidecode_es(v).lower())}\b"), k)
              for k, vals in SYNONYMS.items() for v in vals]
ALIAS_PATTS= [(re.compile(rf"\b{re.escape(unidecode_es(k).lower())}\b"), v)
              for k, v in MULTI_ALIAS.items()]

def apply_alias_and_syns(text: str) -> str:
    t = unidecode_es(text).lower()
    for rx, rep in ALIAS_PATTS:
        t = rx.sub(rep, t)
    for rx, rep in SYN_PATTS:
        t = rx.sub(rep, t)
    t = _re_nonword.sub(" ", t)
    t = _re_multi_ws.sub(" ", t).strip()
    return t

# =============================
# Greetings/despedidas
# =============================
GREET_WORDS = {"hola","buenas","hey","oli","saludos","saludos bot"}
BYE_WORDS   = {"adios","adiós","bye","chao","hasta","hasta luego","nos","nos vemos"}

def greet_route(text: str) -> Optional[str]:
    toks = set(tokenize(apply_alias_and_syns(text)))
    if toks & GREET_WORDS: return "saludos"
    if toks & BYE_WORDS: return "despedida"
    return None

# =============================
# Conversión a secuencia
# =============================
def text_to_sequence(sentence: str, tokenizer, max_len: int):
    seq = tokenizer.texts_to_sequences([sentence])
    return pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

def predict_class(sentence: str, tokenizer, label_encoder, model, max_len, threshold: float = NN_THRESHOLD):
    seq = text_to_sequence(sentence, tokenizer, max_len)
    probs = model.predict(seq, verbose=0)[0]
    res = [(i, float(p)) for i, p in enumerate(probs) if p >= threshold]
    res.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": label_encoder.classes_[i], "probability": f"{p:.4f}"} for i, p in res]

# =============================
# ChatRuntime con fuzzy fallback
# =============================
class ChatRuntime:
    def __init__(self):
        self.intents = _load_json(INTENTS_PATH)
        self.tokenizer = _load_pickle(TOKENIZER_PATH)
        self.label_encoder = _load_pickle(LABELENC_PATH)
        self.model = load_model(MODEL_PATH)

        # Diccionario para fuzzy matching
        self.known_phrases = {}
        for intent in self.intents.get("intents", []):
            tag = intent.get("tag")
            for pattern in intent.get("patterns", []):
                norm_pat = normalize_text(pattern)
                self.known_phrases[norm_pat] = tag
    
    def _respond_tag(self, tag: str) -> Optional[str]:
        it = next((i for i in self.intents["intents"] if i["tag"] == tag), None)
        if it and it.get("responses"):
            return random.choice(it["responses"])
        return None

    def _fuzzy_match(self, text: str, cutoff: float = 0.7) -> Optional[str]:
        normalized = normalize_text(text)
        matches = get_close_matches(normalized, self.known_phrases.keys(), n=1, cutoff=cutoff)
        if matches:
            return self.known_phrases[matches[0]]
        return None

    def get_response(self, user_sentence: str, nn_threshold: float = NN_THRESHOLD) -> str:
        # 1) saludo/despedida
        g = greet_route(user_sentence)
        if g:
            ans = self._respond_tag(g)
            if ans: return ans
            return "¡Hola! ¿En qué puedo ayudarte?" if g == "saludos" else "¡Hasta luego!"
        
        # 2) NN predict
        preds = predict_class(user_sentence, self.tokenizer, self.label_encoder, self.model, MAX_LEN, nn_threshold)
        if preds:
            tag = preds[0]["intent"]
            ans = self._respond_tag(tag)
            if ans: return ans
        
        # 3) Fuzzy match fallback
        fuzzy_tag = self._fuzzy_match(user_sentence, cutoff=0.65)
        if fuzzy_tag:
            ans = self._respond_tag(fuzzy_tag)
            if ans:
                return f"(Pregunta aproximada encontrada: '{fuzzy_tag}')\n{ans}"

        # 4) fallback general
        ans = self._respond_tag("preguntas_no_reconocidas")
        return (ans + " Puedes preguntarme sobre héroes, ítems, misiones o subasta.") if ans else \
               "Lo siento, no entendí tu pregunta. ¿Puedes reformularla?"
# =============================