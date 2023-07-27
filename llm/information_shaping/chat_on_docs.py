import os
import pickle

from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question about summaries or specifications of building planning rules.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI assistant for answering questions about the inspection of building code compliance.
You are given the following extracted parts of a long document and a question. Provide a brief summary of the key Planning Rules,
along with the page number or section where each rule can be found.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])


def get_chain(vectorstore):
    qa_chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
        vectorstore.as_retriever(search_kwargs={"score_threshold": 0.6, "k": 5}),
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        condense_question_llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'),
        return_source_documents=True,
    )
    return qa_chain


class ChatDoc:
    def __init__(self):
        with open('./assets/vectorstore.pkl', mode='rb') as fd:
            vectorstore = pickle.load(fd)
        self.qa_chain = get_chain(vectorstore)
        self.chat_history = []
    
    def chat(self, q: str):
        a = self.qa_chain({"question": q, "chat_history": self.chat_history})
        self.chat_history.append((q, a['answer']))

        answer = f'AI: {a["answer"]}'
        refs = []
        for i, ref in enumerate(a['source_documents']):
            ref_id = os.path.basename(f'REF{i}:\n{ref.metadata["source"]}')
            refs.append(ref_id)
        return f"{answer}\nREFS: {','.join(refs)}"


def chat_cli():
    with open('./assets/vectorstore.pkl', mode='rb') as fd:
        vectorstore = pickle.load(fd)

    chat_history = []
    qa_chain = get_chain(vectorstore)
    while (q := input('Human:\n').lower()) != 'bye':
        a = qa_chain({"question": q, "chat_history": chat_history})
        chat_history.append((q, a['answer']))
        print(f'AI:\n{a["answer"]}')
        for ref in a['source_documents']:
            print(f'REF:\n{ref}')
    print('AI: See you.')


if __name__ == '__main__':
    chat_cli()

