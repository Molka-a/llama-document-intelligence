import io

import docx
import pymupdf
import pytest


@pytest.fixture
def make_pdf():
    def _make(text: str = "Llama Document Intelligence") -> bytes:
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), text)
        data = doc.tobytes()
        doc.close()
        return data
    return _make


@pytest.fixture
def make_docx():
    def _make(paragraphs=("Molka Amara", "Data Scientist"), table=None, author="Tester") -> bytes:
        document = docx.Document()
        for p in paragraphs:
            document.add_paragraph(p)
        if table:
            t = document.add_table(rows=len(table), cols=len(table[0]))
            for r, row in enumerate(table):
                for c, value in enumerate(row):
                    t.cell(r, c).text = value
        document.core_properties.author = author
        buffer = io.BytesIO()
        document.save(buffer)
        return buffer.getvalue()
    return _make
