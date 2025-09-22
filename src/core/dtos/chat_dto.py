from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str = Field(..., example="Hola, ¿qué héroes hay disponibles?")

class ChatResponse(BaseModel):
    reply: str = Field(..., example="¡Hola! Los héroes disponibles son...")
