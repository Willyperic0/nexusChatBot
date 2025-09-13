## **Guía de Instalación y Ejecución**

### **1. Preparar el entorno**

1. Recomiendo generar un entorno virtual (virtual environment) para aislar las dependencias del proyecto:

   ```bash
   python3 -m venv mi_venv
   ```

   > Nota: Puedes reemplazar `mi_venv` por el nombre que desees para tu entorno.

2. Activar el entorno virtual:

   * En **Linux / macOS**:

     ```bash
     source mi_venv/bin/activate
     ```
   * En **Windows (PowerShell)**:

     ```powershell
     mi_venv\Scripts\activate
     ```

3. Verás que tu terminal muestra el nombre del entorno al inicio, indicando que todas las instalaciones se harán dentro de este entorno.

---

### **2. Instalar dependencias**

Instala todas las librerías necesarias utilizando `requirements.txt`:

```bash
pip install --no-cache-dir -r requirements.txt
```

> Esto asegurará que se instalen `nltk`, `tensorflow`, `numpy`, `streamlit`, `scikit-learn` y cualquier otra dependencia listada.

---

### **3. Preparar los datos y entrenar el modelo**

1. Ejecuta el script de entrenamiento:

   ```bash
   python training_chatbot.py
   ```

   * Esto generará los archivos:

     * `words.pkl`
     * `classes.pkl`
     * `chatbot_model.h5`
   * El modelo tiene la siguiente arquitectura:

     * Capa de entrada: 128 neuronas
     * Capa oculta: 64 neuronas
     * Capa de salida: cantidad de clases entrenadas

2. Estos archivos permiten al chatbot predecir intenciones y dar respuestas basadas en tu dataset.

---

### **4. Ejecutar el chatbot**

1. Para probar la lógica del chatbot desde la consola:

   ```bash
   python chatbot.py
   ```

   * Esto cargará los archivos generados y el diccionario de datos para que los métodos funcionen correctamente.

2. Para iniciar la interfaz web con Streamlit:

   ```bash
   streamlit run index.py
   ```

   * Se abrirá automáticamente en tu navegador, mostrando la interfaz del chatbot para interactuar con él.

---

### **5. Notas importantes**

* Asegúrate de que `nexus_data.json` esté en la misma carpeta que los scripts, ya que contiene el dataset de intenciones y respuestas.
* Mantén el entorno virtual activado mientras trabajas en el proyecto para evitar conflictos con otras librerías del sistema.
* Si agregas o modificas intenciones en el dataset, recuerda reentrenar el modelo (`training_chatbot.py`) para que los cambios surtan efecto.

