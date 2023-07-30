import os
import pickle
from typing import List

from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

from schema import Dialog, DocRef, RepliedDialog, ChatHistory, _qa_template, _condense_template


CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_condense_template)
QA_PROMPT = PromptTemplate(template=_qa_template, input_variables=["question", "context"])


def get_chain(vectorstore, doc_qa_template: str, chat_condense_template: str):
    chat_condense_prompt = PromptTemplate.from_template(template=chat_condense_template)
    doc_qa_prompt = PromptTemplate(template=doc_qa_template, input_variables=["question", "context"])
    qa_chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
        vectorstore.as_retriever(search_kwargs={"score_threshold": 0.6, "k": 5}),
        combine_docs_chain_kwargs={"prompt": doc_qa_prompt},
        condense_question_prompt=chat_condense_prompt,
        condense_question_llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'),
        return_source_documents=True,
    )
    return qa_chain


class ChatDoc:
    def __init__(self):
        with open('./assets/vectorstore.pkl', mode='rb') as fd:
            vectorstore = pickle.load(fd)
        self.qa_chain = get_chain(vectorstore, _qa_template, _condense_template)
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
    qa_chain = get_chain(vectorstore, _qa_template, _condense_template)
    while (q := input('Human:\n').lower()) != 'bye':
        a = qa_chain({"question": q, "chat_history": chat_history})
        chat_history.append((q, a['answer']))
        print(f'AI:\n{a["answer"]}')
        for ref in a['source_documents']:
            print(f'REF:\n{ref}')
    print('AI: See you.')


class ChatDocAPI:
    def __init__(self):
        with open('./assets/vectorstore.pkl', mode='rb') as fd:
            self.vectorstore = pickle.load(fd)
    
    def chat(self, dialog: Dialog):
        qa_chain = get_chain(self.vectorstore, dialog.doc_qa_template, dialog.chat_condense_template)

        chat_history = [(hist.user, hist.agent) for hist in dialog.chat_history]
        res = qa_chain({"question": dialog.user, "chat_history": chat_history})

        refs = []
        for src in res['source_documents']:
            ref_id = os.path.basename(f'{src.metadata["source"]}')
            ref = DocRef(ref_id=ref_id, content=src.page_content)
            refs.append(ref)
        
        question = dialog.user
        answer = res['answer']
        dialog.chat_history.append(ChatHistory(user=question, agent=answer))
        dialog = RepliedDialog(user=dialog.user, 
                               chat_history=dialog.chat_history, 
                               agent=answer, 
                               refs=refs, 
                               chat_condense_template=dialog.chat_condense_template,
                               doc_qa_template=dialog.doc_qa_template)
        return dialog
