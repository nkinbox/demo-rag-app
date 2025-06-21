import os
import uuid
import fitz  # PyMuPDF
import openai
import weaviate
import spacy
from weaviate.classes.config import Property, DataType, Configure

from flask import Flask, request, render_template, redirect, url_for

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load spaCy NLP model
nlp = spacy.load("en_core_web_lg")

# Initialize Weaviate client
client = weaviate.connect_to_local()

# Create vector DB class if not exists
def init_weaviate_schema():
    if not client.collections.exists("ComplianceChunk"):
        client.collections.create(
            "ComplianceChunk",
            vectorizer_config=None,
            properties=[
                Property(name="file_id", data_type=DataType.TEXT),
                Property(name="page", data_type=DataType.INT),
                Property(name="chunk", data_type=DataType.TEXT),
            ]
        )

init_weaviate_schema()


def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    return [{"page": i + 1, "text": page.get_text()} for i, page in enumerate(doc)]


def chunk_with_spacy(text, max_tokens=1000, overlap=2):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
    chunks, current_chunk, current_len = [], [], 0

    for sent in sentences:
        current_chunk.append(sent)
        current_len += len(sent.split())
        if current_len >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:]
            current_len = sum(len(s.split()) for s in current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def embed_and_store(file_id, pdf_path):
    pages = extract_pdf_text(pdf_path)
    all_chunks = []

    for page_data in pages:
        chunks = chunk_with_spacy(page_data["text"])
        for chunk in chunks:
            all_chunks.append({"chunk": chunk, "page": page_data["page"]})

    texts = [c["chunk"] for c in all_chunks]
    embeddings = openai.Embedding.create(input=texts, model="text-embedding-ada-002")["data"]

    for i, c in enumerate(all_chunks):
        client.data_object.create(
            class_name="ComplianceChunk",
            vector=embeddings[i]["embedding"],
            data_object={
                "file_id": file_id,
                "page": c["page"],
                "chunk": c["chunk"]
            }
        )


def delete_from_weaviate(file_id):
    client.batch.delete_objects(
        class_name="ComplianceChunk",
        where={"path": ["file_id"], "operator": "Equal", "valueString": file_id}
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'upload' in request.form:
            file = request.files['file']
            if file and file.filename.endswith(".pdf"):
                file_id = str(uuid.uuid4())
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_id + ".pdf")
                file.save(file_path)
                embed_and_store(file_id, file_path)
        elif 'delete' in request.form:
            file_id = request.form['delete']
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file_id + ".pdf"))
            delete_from_weaviate(file_id)

        return redirect(url_for('index'))

    files = [
        f.replace(".pdf", "") for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith(".pdf")
    ]
    return render_template("index.html", files=files)


if __name__ == "__main__":
    app.run(debug=True)
