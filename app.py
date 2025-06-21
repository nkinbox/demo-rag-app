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


def chunk_with_spacy(text, max_tokens=500, overlap=2):
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

def cleanChunk(text):
    prompt = (
        'You are a PDF cleanup and formatting assistant for compliance documents.\n'
        '\n'
        'Your task is to:\n'
        '1. Remove layout junk (like "Page 4", headers, footers, repeated section names)\n'
        '2. Fix spacing, line breaks, OCR issues\n'
        '3. Format the cleaned text into **Markdown** to make it more human-readable:\n'
        '   - Use bullet points, numbered lists, or headings where obvious\n'
        '   - Preserve paragraph breaks and logical structure\n'
        '\n'
        'Very important rules:\n'
        '- Do **not** paraphrase, reword, or summarize anything\n'
        '- Do **not** add your own content or interpretation\n'
        '- Do **not** modify legal meaning, grammar, or sentence structure\n'
        '- Only apply formatting — the wording must remain 100% the same\n'
        '\n'
        'Here is the raw text:\n'
        '---\n'
        '{text}\n'
        '---\n'
        '\n'
        'Output:\n'
        '- A clean, Markdown-formatted version of the above text\n'
        '- With original wording preserved\n'
        '- Do not include any explanation or assistant-like comments\n'
    )
    openai = OpenAI()
    response = openai.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def embed_and_store(file_id, pdf_path):
    openai = OpenAI()
    pages = extract_pdf_text(pdf_path)
    all_chunks = []

    texts = []
    for page_data in pages:
        chunks = chunk_with_spacy(page_data["text"])
        for chunk in chunks:
            cleaned_chunk = cleanChunk(chunk)
            texts.append(cleaned_chunk)
            all_chunks.append({"chunk": cleaned_chunk, "page": page_data["page"]})

    if not len(all_chunks):
        return
    
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
