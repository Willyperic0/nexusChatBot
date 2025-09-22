# 🤖 NexusBattle Chatbot API

Servidor backend para el chatbot de **NexusBattle**, construido con **FastAPI** y un modelo entrenado en TensorFlow/Keras.
Permite procesar preguntas de los usuarios, interpretar intenciones y devolver respuestas predefinidas o aproximadas usando fuzzy matching.

---

## 🚀 Guía de Instalación y Ejecución

### **1. Preparar el entorno**

1. Crea un entorno virtual:

```bash
python -m venv mi_venv
```

2. Actívalo según tu sistema operativo:

* **Linux / macOS**:

  ```bash
  source mi_venv/bin/activate
  ```
* **Windows (PowerShell)**:

  ```powershell
  mi_venv\Scripts\activate
  ```

3. Verás `(mi_venv)` al inicio de tu terminal, confirmando que el entorno está activo.

---

### **2. Instalar dependencias**

Instala todo desde `requirements.txt`:

```bash
pip install --no-cache-dir -r requirements.txt
```

Incluye:

* `fastapi`
* `uvicorn`
* `tensorflow`
* `numpy`
* `scikit-learn`
* `nltk`
* `unidecode`

---

### **3. Entrenar el modelo (opcional)**

Si actualizas el dataset (`nexus_data.json`), debes reentrenar:

```bash
python scripts/training_chatbot.py
```

Esto generará en la carpeta `artifacts/`:

* `chatbot_model.h5`
* `tokenizer.pkl`
* `label_encoder.pkl`
* otros archivos auxiliares (`words.pkl`, `classes.pkl`, etc.)

---

### **4. Ejecutar el servidor FastAPI**

Levanta el backend con:

```bash
python main.py
```

El servidor quedará disponible en:

* **API Docs (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

---

### **5. Probar el chatbot**

#### Usando Postman o cURL

* **Endpoint:**

  ```
  POST http://localhost:8000/api/chat/
  ```

* **Headers:**

  ```
  Content-Type: application/json
  ```

* **Body (JSON):**

  ```json
  {
    "message": "mago fueg acciones 2 5 8"
  }
  ```

* **Respuesta (ejemplo):**

  ```json
  {
    "reply": "Mago Fuego — Acciones por nivel: Nivel 2: Misiles de magma..."
  }
  ```

---

### **6. Testing rápido**

Para ejecutar un lote de pruebas desde consola:

```bash
python tests/test_chatbot.py
```

Puedes pasar preguntas personalizadas:

```bash
python tests/test_chatbot.py "hola" "qué son los créditos" "acciones mago fuego"
```

También puedes exportar los resultados:

```bash
python tests/test_chatbot.py --txt resultados.txt
```

---

## 📂 Estructura del Proyecto

```
nexusChatBot/
├── main.py                  # Punto de entrada FastAPI
├── requirements.txt         # Dependencias
├── .env                     # Variables de entorno
├── data/                    # Dataset base
│   └── nexus_data.json
├── artifacts/               # Modelos y archivos entrenados
│   ├── chatbot_model.h5
│   ├── tokenizer.pkl
│   ├── label_encoder.pkl
│   └── ...
├── scripts/
│   └── training_chatbot.py  # Script de entrenamiento
├── src/core/
│   ├── controllers/
│   │   └── chat_controller.py
│   ├── dtos/
│   │   └── chat_dto.py
│   ├── models/
│   │   └── chatbot_runtime.py
│   └── ...
└── tests/
    └── test_chatbot.py
```

---

## ⚡ Notas importantes

* Si cambias `nexus_data.json`, **reentrena el modelo**.
* El campo obligatorio en la API es siempre `"message"`.
* El servidor se ejecuta en `http://localhost:8000` por defecto.
* Usa `reload=True` en desarrollo para autorecargar cambios.

---

👨‍💻 Hecho con FastAPI + TensorFlow para NexusBattle.