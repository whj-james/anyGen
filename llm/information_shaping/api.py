from typing import List

from fastapi import FastAPI

from schema import Dialog
from chat_on_docs import ChatDocAPI


app = FastAPI(title="LLM API",
              description="Currently for Regulation Rules",
              version="0.0.1",
              docs_url='/api/generate/regulations/docs',
              redoc_url='/api/generate/regulations/redoc',
              openapi_url='/api/generate/regulations/openapi.json')


@app.post("/api/generate/regulations/chat")
async def chat_with_history(dialog: Dialog):
    chatdoc = ChatDocAPI()
    result = chatdoc.chat(dialog)

    return result
