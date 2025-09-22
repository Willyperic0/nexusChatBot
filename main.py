import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.core.controllers import chat_controller

app = FastAPI(
    title="NexusBattle Chatbot API",
    version="1.0.0",
    description="Servidor para el chatbot de NexusBattle"
)

# --- CORS middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # tu frontend Angular
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas del chatbot
app.include_router(chat_controller.router, prefix="/api/chat", tags=["Chat"])

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
