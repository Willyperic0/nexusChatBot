# chatbot.py
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")          # silencia INFO/WARN de TF
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")        # evita variaciones numéricas y logs oneDNN

import json, pickle, re, random
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# =============================
# Config
# =============================
NN_THRESHOLD   = 0.50
SEM_THRESHOLD  = 0.40
NGRAM_RANGE    = (3, 5)
RANDOM_SEED    = 7
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =============================
# Rutas base
# =============================
def _base_dir() -> Path:
    try:
        return Path(__file__).parent.resolve()
    except NameError:
        return Path.cwd().resolve()

BASE = _base_dir()
INTENTS_PATH = (BASE / "nexus_data.patched.json") if (BASE / "nexus_data.patched.json").exists() else (BASE / "nexus_data.json")
WORDS_PATH   = BASE / "words.pkl"
CLASSES_PATH = BASE / "classes.pkl"
MODEL_PATH   = BASE / "chatbot_model.h5"

# =============================
# Utilidades de carga
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
_re_nonword  = re.compile(r"[^\w\s]")
_dup_re      = re.compile(r"(.)\1+")  # colapsa 'aa...' -> 'a'

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
# Vocabulario de dominio (fallback temprano)
# =============================
DOMAIN_TOKENS = set("""
guerrero tanque armas mago fuego hielo picaro veneno machete chaman medico
acciones poder vida defensa ataque daño armaduras armadura armas arma
cofres creditos subasta compra inmediata barra vida torneo
misiles magma vulcano pared lluvia cono bola cortada machetazo planazo
flor agonia piquete toque vinculo canto curacion neutralizacion reanimacion
espada escudo piedra orbe fatuo baculo venas daga vision machete cierra raiz yerbabuena
defensa enfurecido magma ardiente tunica corona ventisca desterrado atleta sangre cruel
piel caminante ecos ancestrales bata cirujano pantalon expedicion
""".split())

# =============================
# Greetings/despedidas — rutas deterministas
# =============================
GREET_WORDS = {"hola","buenas","hey","oli","saludos","saludos bot"}
BYE_WORDS   = {"adios","adiós","bye","chao","hasta","hasta luego","nos","nos vemos"}
def greet_route(text: str) -> Optional[str]:
    toks = set(tokenize(apply_alias_and_syns(text)))
    if toks & GREET_WORDS: return "saludos"
    if toks & BYE_WORDS:   return "despedida"
    return None

# =============================
# BoW + NN
# =============================
def load_intents_words_classes():
    intents = _load_json(INTENTS_PATH)
    words   = _load_pickle(WORDS_PATH)
    classes = _load_pickle(CLASSES_PATH)
    return intents, words, classes

def build_word_index(words: List[str]) -> Dict[str, int]:
    return {w: i for i, w in enumerate(words)}

def bag_of_words(sentence: str, words: List[str], word_index: Dict[str, int]) -> np.ndarray:
    toks = tokenize(normalize_text(sentence))
    bag = np.zeros(len(words), dtype=np.float32)
    for w in set(toks):
        idx = word_index.get(w)
        if idx is not None:
            bag[idx] = 1.0
    return bag

_model = None
def get_model():
    global _model
    if _model is not None:
        return _model
    try:
        from keras.models import load_model
        _model = load_model(MODEL_PATH)
    except Exception:
        _model = None
    return _model

def predict_class(sentence: str, words: List[str], classes: List[str], word_index: Dict[str,int], threshold: float = NN_THRESHOLD):
    m = get_model()
    if m is None:
        return []
    bow = bag_of_words(sentence, words, word_index)
    probs = m.predict(np.array([bow]), verbose=0)[0]
    res = [[i, float(p)] for i, p in enumerate(probs) if p >= threshold]
    res.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[i], "probability": f"{p:.4f}"} for i, p in res]

# =============================
# Semántica char-ngrams (tolerante a fallos)
# =============================
_vectorizer, _tfidf_matrix, _tags_list = None, None, None
_SKLEARN_OK = False

def ensure_semantic_index(intents_json):
    global _vectorizer, _tfidf_matrix, _tags_list, _SKLEARN_OK
    if _vectorizer is not None or _SKLEARN_OK:
        return
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity  # noqa
    except Exception:
        _SKLEARN_OK = False
        _vectorizer = _tfidf_matrix = _tags_list = None
        return

    patterns, tags = [], []
    for it in intents_json.get("intents", []):
        tag = it.get("tag")
        for p in it.get("patterns", []):
            patterns.append(normalize_text(apply_alias_and_syns(p)))
            tags.append(tag)

    _vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=NGRAM_RANGE, min_df=1)
    _tfidf_matrix = _vectorizer.fit_transform(patterns) if patterns else None
    _tags_list = tags
    _SKLEARN_OK = True

def semantic_search(sentence: str, threshold: float = SEM_THRESHOLD, top_k: int = 5) -> Tuple[Optional[str], float]:
    if _tfidf_matrix is None or _vectorizer is None:
        return None, 0.0
    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return None, 0.0

    s = normalize_text(apply_alias_and_syns(sentence))
    sims = _vectorizer.transform([s])
    sims = cosine_similarity(sims, _tfidf_matrix)[0]
    idxs = np.argsort(sims)[-top_k:][::-1]
    cand = [(_tags_list[i], float(sims[i])) for i in idxs if sims[i] >= threshold]
    if not cand:
        return None, 0.0

    # Reranking con bonus/malus para evitar cruces HIELO↔FUEGO
    bonus = 0.15
    malus = 0.18
    new = []
    for t, sc in cand:
        adj = sc
        if "hielo" in s:
            if any(k in t for k in ["mago_hielo","bola_de_hielo","cono_de_hielo","lluvia_de_hielo"]): adj += bonus
            if any(k in t for k in ["mago_fuego","vulcano","misiles_de_magma","pared_de_fuego"]):     adj -= malus
        if "fuego" in s or "magma" in s or "vulcano" in s:
            if any(k in t for k in ["mago_fuego","vulcano","misiles_de_magma","pared_de_fuego"]):     adj += bonus
            if any(k in t for k in ["mago_hielo","bola_de_hielo","cono_de_hielo","lluvia_de_hielo"]): adj -= malus
        new.append((t, adj))
    new.sort(key=lambda x: x[1], reverse=True)
    return new[0]

# =============================
# Resolución directa por NOMBRE (equipo + acciones)
# =============================
def _contains_phrase(tokens: List[str], phrase: str) -> bool:
    p = phrase.split()
    n = len(p)
    for i in range(len(tokens)-n+1):
        if tokens[i:i+n] == p:
            return True
    return False

# nombres en minúsculas y SIN acentos (tokenize unidecodea)
NAME_TO_TAG = {
    # Armas
    "espada de dos manos": "arma_guerrero_armas_espada_de_dos_manos",
    "piedra de afilar":    "arma_guerrero_armas_piedra_de_afilar",
    "espada de una mano":  "arma_guerrero_tanque_espada_de_una_mano",
    "escudo de dragon":    "arma_guerrero_tanque_escudo_de_dragon",
    "orbe de manos ardientes": "arma_mago_fuego_orbe_de_manos_ardientes",
    "fuego fatuo":              "arma_mago_fuego_fuego_fatuo",
    "baculo de permafrost": "arma_mago_hielo_baculo_de_permafrost",
    "venas heladas":        "arma_mago_hielo_venas_heladas",
    "machete bendito":      "arma_picaro_machete_machete_bendito",
    "cierra sangrienta":    "arma_picaro_machete_cierra_sangrienta",
    "daga purulenta":       "arma_picaro_veneno_daga_purulenta",
    "vision borrosa":       "arma_picaro_veneno_vision_borrosa",
    "raiz china":           "arma_chaman_raiz_china",
    "yerbabuena":           "arma_chaman_yerbabuena",
    "kit de urgencias":     "arma_medico_kit_de_urgencias",
    "reanimador":           "arma_medico_reanimador",
    # Armaduras (OJO: tags con ñ exacta del dataset)
    "defensa del enfurecido":"armadura_guerrero_tanque_defensa_del_enfurecido",
    "magma ardiente":        "armadura_guerrero_tanque_magma_ardiente",
    "puno lucido":           "armadura_guerrero_armas_puño_lucido",
    "puños en llamas":       "armadura_guerrero_armas_puños_en_llamas",
    "tunica arcana":         "armadura_mago_fuego_tunica_arcana",
    "caida de fuego":        "armadura_mago_fuego_caida_de_fuego",
    "corona de hielo":       "armadura_mago_hielo_corona_de_hielo",
    "ventisca":              "armadura_mago_hielo_ventisca",
    "pie de atleta":         "armadura_picaro_machete_pie_de_atleta",
    "sangre cruel":          "armadura_picaro_machete_sangre_cruel",
    "mano del desterrado":   "armadura_picaro_veneno_mano_del_desterrado",
    "atadura carmesi":       "armadura_picaro_veneno_atadura_carmesi",
    "bata de cirujano":      "armadura_medico_bata_de_cirujano",
    "pantalon de expedicion medica": "armadura_medico_pantalon_de_expedicion_medica",
    "casco de ecos ancestrales": "armadura_chaman_casco_de_ecos_ancestrales",
    "piel de caminante del bosque": "armadura_chaman_piel_de_caminante_del_bosque",
}
# Acciones -> tags reales del dataset (accion_<heroe>_nivel_<N>)
ACTION_NAME_TO_TAG = {
    # Guerrero Tanque
    "golpe con escudo": "accion_guerrero_tanque_nivel_2",
    "mano de piedra":   "accion_guerrero_tanque_nivel_5",
    "defensa feroz":    "accion_guerrero_tanque_nivel_8",
    # Guerrero Armas
    "embate sangriento":   "accion_guerrero_armas_nivel_2",
    "lanza de los dioses": "accion_guerrero_armas_nivel_5",
    "golpe de tormenta":   "accion_guerrero_armas_nivel_8",
    # Mago Fuego
    "misiles de magma": "accion_mago_fuego_nivel_2",
    "vulcano":          "accion_mago_fuego_nivel_5",
    "pared de fuego":   "accion_mago_fuego_nivel_8",
    # Mago Hielo
    "lluvia de hielo": "accion_mago_hielo_nivel_2",
    "cono de hielo":   "accion_mago_hielo_nivel_5",
    "bola de hielo":   "accion_mago_hielo_nivel_8",  # <- clave para cortar fallo a Vulcano
    # Pícaro Veneno
    "flor de loto": "accion_picaro_veneno_nivel_2",
    "agonia":       "accion_picaro_veneno_nivel_5",
    "piquete":      "accion_picaro_veneno_nivel_8",
    # Pícaro Machete
    "cortada":    "accion_picaro_machete_nivel_2",
    "machetazo":  "accion_picaro_machete_nivel_5",
    "planazo":    "accion_picaro_machete_nivel_8",
    # Chamán
    "toque de la vida": "accion_chaman_nivel_2",
    "vinculo natural":  "accion_chaman_nivel_5",
    "canto del bosque": "accion_chaman_nivel_8",
    # Médico
    "curacion directa":            "accion_medico_nivel_2",
    "neutralizacion de efectos":   "accion_medico_nivel_5",
    "reanimacion":                 "accion_medico_nivel_8",
    # Variantes con tildes (por si llegan así)
    "vínculo natural":  "accion_chaman_nivel_5",
    "curación directa": "accion_medico_nivel_2",
    "neutralización de efectos": "accion_medico_nivel_5",
    "reanimación":      "accion_medico_nivel_8",
}

def direct_name_route(text: str) -> Optional[str]:
    tokens = tokenize(apply_alias_and_syns(text))
    # Equipo primero (armas / armaduras)
    for name, tag in NAME_TO_TAG.items():
        if _contains_phrase(tokens, name):
            return tag
    # Acciones después (tags reales del dataset)
    for name, tag in ACTION_NAME_TO_TAG.items():
        if _contains_phrase(tokens, name):
            return tag
    return None

# =============================
# Router por keyword (héroe/economía/UI)
# =============================
def build_keyword_router(intents_json) -> Dict[str, str]:
    def tag_exists(tag: str) -> bool:
        return any(i.get("tag") == tag for i in intents_json.get("intents", []))
    def first(*tags):
        for t in tags:
            if t and tag_exists(t):
                return t
        return ""
    r: Dict[str, str] = {}
    r["guerrero_tanque"] = first(
        "acciones_por_nivel_guerrero_tanque",
        "armas_por_heroe_guerrero_tanque",
        "acciones_guerrero_tanque_resumen",
        "armaduras_por_heroe_guerrero_tanque",
        "acciones_por_nivel_guerrero_tanque_canon",
    )
    r["guerrero_armas"] = first(
        "acciones_por_nivel_guerrero_armas",
        "armas_por_heroe_guerrero_armas",
        "acciones_guerrero_armas_resumen",
        "armaduras_por_heroe_guerrero_armas",
        "acciones_por_nivel_guerrero_armas_canon",
    )
    r["mago_fuego"] = first(
        "acciones_por_nivel_mago_fuego",
        "armas_por_heroe_mago_fuego",
        "acciones_mago_fuego_resumen",
        "armaduras_por_heroe_mago_fuego",
        "acciones_por_nivel_mago_fuego_canon",
    )
    r["mago_hielo"] = first(
        "acciones_por_nivel_mago_hielo",
        "armas_por_heroe_mago_hielo",
        "acciones_mago_hielo_resumen",
        "armaduras_por_heroe_mago_hielo",
        "acciones_por_nivel_mago_hielo_canon",
    )
    r["picaro_machete"] = first(
        "acciones_por_nivel_picaro_machete",
        "armas_por_heroe_picaro_machete",
        "acciones_picaro_machete_resumen",
        "armaduras_por_heroe_picaro_machete",
        "acciones_por_nivel_picaro_machete_canon",
    )
    r["picaro_veneno"] = first(
        "acciones_por_nivel_picaro_veneno",
        "armas_por_heroe_picaro_veneno",
        "acciones_picaro_veneno_resumen",
        "armaduras_por_heroe_picaro_veneno",
        "acciones_por_nivel_picaro_veneno_canon",
    )
    r["chaman"] = first(
        "acciones_por_nivel_chaman",
        "armas_por_heroe_chaman",
        "acciones_chaman_resumen",
        "armaduras_por_heroe_chaman",
        "acciones_por_nivel_chaman_canon",
    )
    r["medico"] = first(
        "acciones_por_nivel_medico",
        "armas_por_heroe_medico",
        "acciones_medico_resumen",
        "armaduras_por_heroe_medico",
        "acciones_por_nivel_medico_canon",
    )
    # Economía / UI
    r["subasta"] = first("subasta") or "subasta"
    r["pujar"] = r["subasta"]
    r["puja"] = r["subasta"]
    r["compra_inmediata"] = first("subasta_compra_inmediata", "subasta") or "subasta"
    r["creditos"] = first("creditos") or "creditos"
    r["cofres"] = first("cofres") or "cofres"
    r["hola"] = first("saludos") or "saludos"
    r["barra_de_vida"] = first("barra_vida", "barra_vida_canon") or "barra_vida"
    return {k: v for k, v in r.items() if v}

def keyword_route(user_sentence: str, router: Dict[str, str]) -> Optional[str]:
    s = normalize_text(user_sentence)
    toks = s.split()
    stoks = set(toks)

    wants_weapons = ("armas" in stoks) or ("arma" in stoks)
    wants_armors  = ("armaduras" in stoks) or ("armadura" in stoks)
    wants_actions = ("acciones" in stoks) or ("poderes" in stoks) or ("explicame" in stoks) or ("que" in stoks) or ("qué" in stoks)

    # subasta robusta
    if "subast" in " ".join(toks) or {"subasta","pujar","puja"} & stoks:
        return router.get("subasta")
    if "compra_inmediata" in stoks or "buyout" in stoks or ("comprar" in stoks and "ya" in stoks):
        return router.get("compra_inmediata") or router.get("subasta")

    heroes = ["guerrero_tanque","guerrero_armas","mago_fuego","mago_hielo","picaro_machete","picaro_veneno","chaman","medico"]
    for h in heroes:
        if h in stoks:
            if wants_weapons:
                return f"armas_por_heroe_{h}"
            if wants_armors:
                return f"armaduras_por_heroe_{h}"
            if wants_actions and router.get(h):
                return router[h]
            return router.get(h)

    if {"colores","indicador","salud"} & stoks:
        return router.get("barra_de_vida")

    for k in ["creditos","cofres","barra_de_vida","hola"]:
        if k in stoks and router.get(k):
            return router[k]

    if len(toks) == 1 and router.get(toks[0]):
        return router[toks[0]]

    return None

# =============================
# Runtime principal
# =============================
class ChatRuntime:
    def __init__(self):
        self.intents, self.words, self.classes = load_intents_words_classes()
        self.word_index = build_word_index(self.words)
        ensure_semantic_index(self.intents)
        self.router = build_keyword_router(self.intents)

    def _respond_tag(self, tag: str) -> Optional[str]:
        it = next((i for i in self.intents["intents"] if i["tag"] == tag), None)
        if it and it.get("responses"):
            return random.choice(it["responses"])
        return None

    def get_response(self, user_sentence: str, nn_threshold: float = NN_THRESHOLD) -> str:
        # -1) saludo/despedida directos (con fallback inline si faltan intents)
        g = greet_route(user_sentence)
        if g:
            ans = self._respond_tag(g)
            if ans:
                return ans
            return "¡Hola! ¿En qué puedo ayudarte?" if g == "saludos" else "¡Hasta luego!"

        # -0) subasta ultra-robusta
        if "subast" in " ".join(tokenize(user_sentence)):
            ans = self._respond_tag("subasta")
            if ans: return ans

        # 00) Nombre exacto de arma/armadura/acción
        dtag = direct_name_route(user_sentence)
        if dtag:
            ans = self._respond_tag(dtag)
            if ans: return ans

        # 0) Ruta por keyword
        kw = keyword_route(user_sentence, self.router)
        if kw:
            ans = self._respond_tag(kw)
            if ans: return ans

        # 🚧 Guardarraíl de dominio ANTES de NN/semántica
        toks = tokenize(normalize_text(user_sentence))
        dom_hits = [w for w in toks if w in DOMAIN_TOKENS]
        dom_ratio = (len(dom_hits) / max(1, len(toks)))
        if not dom_hits or dom_ratio < 0.10:
            ans = self._respond_tag("preguntas_no_reconocidas")
            return ans if ans else "Lo siento, no entendí tu pregunta. ¿Puedes reformularla?"

        # 🛡️ Forzador de equipo si dice armas/armaduras + héroe
        tokens_join = " ".join(toks)
        heroes = ["guerrero_tanque","guerrero_armas","mago_fuego","mago_hielo","picaro_machete","picaro_veneno","chaman","medico"]
        h_found = next((h for h in heroes if h in tokens_join), None)
        if h_found:
            if " armas " in f" {tokens_join} " or tokens_join.endswith(" armas"):
                forced = f"armas_por_heroe_{h_found}"
                ans = self._respond_tag(forced)
                if ans: return ans
            if " armadura " in f" {tokens_join} " or " armaduras " in f" {tokens_join} ":
                forced = f"armaduras_por_heroe_{h_found}"
                ans = self._respond_tag(forced)
                if ans: return ans

        # Refuerzo: 'costo/efecto/detalle' + nombre de acción -> intenta nombre otra vez
        lowq = unidecode_es(user_sentence).lower()
        if any(k in lowq for k in ["costo y efecto","costo","efecto","detalle"]):
            dtag2 = direct_name_route(user_sentence)
            if dtag2:
                ans = self._respond_tag(dtag2)
                if ans: return ans

        # 1) NN
        preds = predict_class(user_sentence, self.words, self.classes, self.word_index, nn_threshold)
        if not preds:
            # 2) Semántica
            tag, score = semantic_search(user_sentence, threshold=SEM_THRESHOLD)
            if tag:
                ans = self._respond_tag(tag)
                if ans: return ans
            # 3) Fallback
            ans = self._respond_tag("preguntas_no_reconocidas")
            return (ans + " Puedes preguntarme sobre héroes, ítems, misiones o subasta.") if ans else "Lo siento, no entendí tu pregunta. ¿Puedes reformularla?"

        # Caso NN ok
        tag = preds[0]["intent"]
        ans = self._respond_tag(tag)
        return ans if ans else "No tengo una respuesta configurada para eso todavía."
