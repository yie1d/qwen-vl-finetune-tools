from fastapi import FastAPI

from routers.qwen import qwen

app = FastAPI()

app.include_router(qwen.router)
