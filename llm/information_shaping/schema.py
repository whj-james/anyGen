from typing import List

from pydantic import BaseModel


class DocRef(BaseModel):
    ref_id: str
    content: str


class ChatHistory(BaseModel):
    user: str
    agent: str


class Dialog(BaseModel):
    user: str
    chat_history: List[ChatHistory]


class RepliedDialog(Dialog):
    agent: str
    refs: List[DocRef]
