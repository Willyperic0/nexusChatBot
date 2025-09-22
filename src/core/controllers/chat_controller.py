from fastapi import APIRouter, HTTPException
from src.core.dtos.chat_dto import ChatRequest, ChatResponse
from src.core.models.chatbot_runtime import ChatRuntime

router = APIRouter()
runtime = ChatRuntime()

@router.post("/", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Endpoint principal del chatbot.
    Recibe un mensaje de usuario y devuelve la respuesta generada.
    """
    try:
        answer = runtime.get_response(req.message)
        return ChatResponse(reply=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
