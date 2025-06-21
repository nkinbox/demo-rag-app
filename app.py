import os
import uuid
import fitz  # PyMuPDF
from openai import OpenAI
import weaviate
import spacy
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import Filter
from weaviate.classes.aggregate import GroupByAggregate

from flask import Flask, request, render_template, redirect, url_for

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

VECTOR_DB = 'ComplianceChunk'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load spaCy NLP model
nlp = spacy.load("en_core_web_lg")

# Create vector DB class if not exists
def init_weaviate_schema():
    # Initialize Weaviate client
    client = weaviate.connect_to_local()
    try:
        if not client.collections.exists(VECTOR_DB):
            client.collections.create(
                VECTOR_DB,
                vectorizer_config=None,
                properties=[
                    Property(name="file_id", data_type=DataType.TEXT),
                    Property(name="page", data_type=DataType.INT),
                    Property(name="chunk", data_type=DataType.TEXT),
                ]
            )
    finally:
        client.close()
    

init_weaviate_schema()


def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_textpage().extractText()
        pages.append({"page": page_num + 1, "text": text})
    
    return pages


def chunk_with_spacy(text, max_tokens=500, overlap_sentences=2):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 3]

    chunks = []
    i = 0

    while i < len(sentences):
        current_chunk = []
        token_count = 0
        j = i

        # Add sentences until max_tokens is reached
        while j < len(sentences) and token_count + len(sentences[j].split()) <= max_tokens:
            current_chunk.append(sentences[j])
            token_count += len(sentences[j].split())
            j += 1

        # Add chunk to result
        chunks.append(" ".join(current_chunk))

        # Move index forward with overlap
        if j == i:  # prevent infinite loop
            j += 1
        i = j - overlap_sentences  # move back `overlap_sentences` to include context

    return chunks


def embed_and_store(file_id, pdf_path):
    openai = OpenAI()
    pages = extract_pdf_text(pdf_path)
    all_chunks = []

    for page_data in pages:
        chunks = chunk_with_spacy(page_data["text"])
        for chunk in chunks:
            all_chunks.append({"chunk": chunk, "page": page_data["page"]})

    if not len(all_chunks):
        return
    
    texts = [c["chunk"] for c in all_chunks]
    embeddings = openai.embeddings.create(input=texts, model="text-embedding-3-large").data

    client = weaviate.connect_to_local()
    try:
        collection = client.collections.get(VECTOR_DB)
        with collection.batch.fixed_size(batch_size=100) as batch:                
            for i, c in enumerate(all_chunks):
                batch.add_object(
                    properties={
                        "file_id": file_id,
                        "page": c["page"],
                        "chunk": c["chunk"]
                    },
                    vector=embeddings[i].embedding
                )
    finally:
        client.close()


def delete_from_weaviate(file_id):
    client = weaviate.connect_to_local()
    try:
        collection = client.collections.get(VECTOR_DB)
        while True:
            groupBy = collection.aggregate.over_all(group_by=GroupByAggregate(prop="file_id"))
            file_id_count = 0
            for group in groupBy.groups:
                if group.grouped_by.value == file_id:
                    file_id_count = int(group.total_count)
            if file_id_count > 0:
                collection.data.delete_many(
                    where=Filter.by_property("file_id").equal(f"{file_id}")
                )
            else:
                break
    finally:
        client.close()


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

@app.route("/read", methods=["GET", "POST"])
def readFile():
    file_id = request.args.get('file_id', None)
    chunks = []
    if file_id:
        client = weaviate.connect_to_local()
        try:
            offset = 0
            limit = 30
            while True:
                result = client.collections.get(VECTOR_DB).query.fetch_objects(
                    filters=Filter.by_property("file_id").equal(f"{file_id}"),
                    limit=limit,
                    offset=offset
                )
                offset += limit
                count = 0
                for o in result.objects:
                    count += 1
                    chunks.append(o.properties)
                
                if not count:
                    break
        finally:
            client.close()

    return render_template("read.html", chunks=chunks)

if __name__ == "__main__":
    app.run(debug=True)
