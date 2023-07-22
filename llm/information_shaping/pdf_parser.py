
from pathlib import Path
from pdfminer.high_level import extract_text

import pickle

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings


def pdf_parser():
    data_dir = Path('assets')
    input_path = data_dir / Path('Amended Guyra Development Control Plan 2015 - June 2020_S-3045.pdf')
    assert input_path.is_file()

    text = extract_text(input_path)
    text = text.replace('Page ', '#'*8 + 'Page')
    pages = text.split('\x0c')

    start_idx = 13
    content_pages = pages[start_idx:]
    for i, p in enumerate(content_pages, start=start_idx):
        page_text_path = data_dir / f'page_{i}'
        with open(page_text_path, encoding='utf-8', mode='w') as fd:
            print(p, file=fd)

    with open(data_dir / 'content.txt', encoding='utf-8', mode='w') as fd:
        fd.writelines(content_pages)

if __name__ == '__main__':
    pdf_parser()