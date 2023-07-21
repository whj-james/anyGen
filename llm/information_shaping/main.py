from pathlib import Path
from pdfminer.high_level import extract_text


if __name__ == '__main__':
    path = Path('assets') / Path('Amended Guyra Development Control Plan 2015 - June 2020_S-3045.pdf')
    assert path.is_file()

    text = extract_text(path)
    pages = text.split('\x0c')

    for p in pages:
        print(p)
        input('go on? ') 