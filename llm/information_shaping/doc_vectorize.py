import os
import pickle

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings


def data_vectorize():
    # Load Data
    raw_docs = []
    with os.scandir('./assets') as entries:
        for entry in entries:
            if entry.is_file() and entry.name.startswith('page_'):
                loader = TextLoader(entry.path)
                raw_docs.extend(loader.load())
    # Split text
    text_splitter = CharacterTextSplitter()
    documents = text_splitter.split_documents(raw_docs)

    # Load Data to vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.similarity_search

    # Save vectorstore
    with open("./assets/vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == '__main__':
    data_vectorize()
