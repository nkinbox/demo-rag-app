import fitz  # PyMuPDF

doc = fitz.open('test-compliance.pdf')
print([{"page": i + 1, "text": page.get_text()} for i, page in enumerate(doc)])