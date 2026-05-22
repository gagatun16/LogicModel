from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from logic import chat, create_empty_state

app = FastAPI()

# Сессии пользователей
sessions = {}


# Модель запроса
class ChatRequest(BaseModel):
    session_id: str
    message: str


# Модель ответа
class ChatResponse(BaseModel):
    session_id: str
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat_api(request: ChatRequest):

    session_id = request.session_id

    # если новый пользователь
    if session_id not in sessions:
        sessions[session_id] = create_empty_state()

    state = sessions[session_id]

    try:
        reply = chat(request.message, state)

        # сохраняем обновленный state
        sessions[session_id] = state

        return ChatResponse(
            session_id=session_id,
            reply=reply
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{session_id}")
def get_state(session_id: str):

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Сессия не найдена")

    return sessions[session_id]


@app.delete("/history/{session_id}")
def clear_state(session_id: str):

    if session_id in sessions:
        del sessions[session_id]

    return {"message": "История очищена"}


@app.get("/health")
def health():
    return {"status": "ok"}
