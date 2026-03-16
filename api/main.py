from fastapi import FastAPI
from api.routes.chat import router as chat_router
from dotenv import load_dotenv
load_dotenv()
app = FastAPI(
    title="AtoZ AI Assistant",
    version="0.1.0"
)

@app.get("/")
def root():
    return {"message": "AtoZ AI Assistant API running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# Register routers
app.include_router(chat_router)
