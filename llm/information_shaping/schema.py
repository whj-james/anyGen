from typing import List

from pydantic import BaseModel, Field


class DocRef(BaseModel):
    ref_id: str
    content: str


class ChatHistory(BaseModel):
    user: str
    agent: str


_condense_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question about summaries or specifications of building planning rules.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""


_qa_template = """You are an AI assistant for answering questions about the inspection of building code compliance.
You are given the following extracted parts of a long document and a question. Provide a brief summary of the key Planning Rules,
along with the page number or section where each rule can be found.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""


class Dialog(BaseModel):
    user: str
    chat_history: List[ChatHistory]
    chat_condense_template: str | None = Field(_condense_template)
    doc_qa_template: str | None = Field(_qa_template)


class RepliedDialog(Dialog):
    agent: str
    refs: List[DocRef]
