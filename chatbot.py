import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import random
import re
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Descargas NLTK ---
nltk.download('punkt')

# --- Carga de recursos ---
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('nexus_data.json', 'r', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# --- Diccionario de sinónimos ampliado ---
synonyms = {
    "dime": ["muéstrame", "indícame", "quiero saber", "quiero información sobre", "explicame", "cuéntame", "enséñame", "recuérdame", "muestrame"],
    "información": ["info", "detalles", "datos", "explicación", "contenido", "resumen", "descripción", "características"],
    "cómo": ["de qué manera", "de qué forma", "manera de", "modo de", "método", "proceso"],
    "qué": ["cual", "cuál", "cuáles", "que"],
    "héroes": ["personajes", "jugables", "clases", "heroes"],
    "armas": ["equipamiento", "espadas", "bastones", "dagas", "mazas", "arcos", "ballestas"],
    "ítems": ["objetos", "productos", "cosas", "items", "elementos"],
    "pociones": ["bebidas", "elixires", "remedios", "pocion"],
    "pergaminos": ["documentos", "libros", "scrolls", "pergamino"],
    "amuletos": ["talismanes", "collares", "amuletos de poder"],
    "cristales": ["gemas", "piedras energéticas", "cristales de energia"],
    "épica": ["poder especial", "habilidad única", "epica"],
    "misiones": ["retos", "desafíos", "quests", "tareas"],
    "créditos": ["dinero", "moneda", "creditos"],
    "cofres": ["recompensas", "cajas", "tesoros"],
    "turnos": ["rondas", "acciones", "fase"]
}

# --- Normalización de texto ---
def normalize_text(text):
    text = unidecode(text)  # quita tildes
    text = text.lower()      # minúsculas
    text = re.sub(r'[^\w\s]', '', text)  # quita puntuación
    return text

# Sustituye sinónimos
def replace_synonyms(sentence):
    for key, vals in synonyms.items():
        for val in vals:
            sentence = re.sub(r'\b' + re.escape(val) + r'\b', key, sentence)
    return sentence

# Tokeniza y lematiza
def clean_up_sentence(sentence):
    sentence = normalize_text(sentence)
    sentence = replace_synonyms(sentence)
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Bolsa de palabras
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predicción de clase con NN
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.5
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# --- Construcción de TF-IDF para similitud semántica ---
patterns_list = []
tags_list = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        normalized = normalize_text(replace_synonyms(pattern))
        patterns_list.append(normalized)
        tags_list.append(intent['tag'])

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(patterns_list)

# Similitud semántica
def semantic_search(sentence):
    sentence_norm = normalize_text(replace_synonyms(sentence))
    sentence_vec = vectorizer.transform([sentence_norm])
    cosine_sim = cosine_similarity(sentence_vec, tfidf_matrix)
    max_idx = np.argmax(cosine_sim)
    max_score = cosine_sim[0][max_idx]
    if max_score >= 0.5:
        return tags_list[max_idx], max_score
    return None, 0

# Respuesta inteligente con fallback y similitud
def get_response(intents_list, intents_json, user_sentence=None, threshold=0.5):
    # Fallback por NN
    if len(intents_list) == 0 or float(intents_list[0]['probability']) < threshold:
        # Fallback por similitud semántica
        if user_sentence:
            tag, score = semantic_search(user_sentence)
            if tag:
                intent = next((i for i in intents_json['intents'] if i['tag']==tag), None)
                if intent:
                    return random.choice(intent['responses'])
        
        # Fallback por preguntas no reconocidas
        fallback_intent = next((i for i in intents_json['intents'] if i['tag']=='preguntas_no_reconocidas'), None)
        if fallback_intent:
            response = random.choice(fallback_intent['responses'])
            response += " Por ejemplo, puedes preguntarme sobre héroes, armas, ítems, misiones o cofres."
            return response
        return "Lo siento, no entendí tu pregunta. ¿Puedes reformularla?"
    
    tag = intents_list[0]['intent']
    intent = next((i for i in intents_json['intents'] if i['tag']==tag), None)
    if intent:
        return random.choice(intent['responses'])