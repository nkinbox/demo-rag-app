import os
import fitz  # PyMuPDF
from openai import OpenAI
import weaviate
import spacy
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.classes.aggregate import GroupByAggregate
from slugify import slugify
from flask import Flask, request, render_template, redirect, url_for, jsonify
import json

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

def extract_json(text, delimiter = ['{', '}']):
    try:
        start = text.find(delimiter[0])
        end = text.rfind(delimiter[1])
        
        if start != -1 and end != -1 and start < end:
            jsonText = text[start:end + 1]
            return json.loads(jsonText)
        else:
            return None
    except Exception:
        return None

def gptResponse(prompt, system = None, temperature = 0):
    openai = OpenAI()
    messages = []
    if system:
        messages.append({
            "role": "system",
            "content": system,
        })
    
    if isinstance(prompt, str):
        messages.append({
            "role": "user",
            "content": prompt,
        })
    else:
        for message in prompt:
            messages.append(message)

    response = openai.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=temperature
    )
    result = response.choices[0].message.content.strip()
    return result

def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_textpage().extractText()
        pages.append({"page": page_num + 1, "text": text})
    
    return pages


def chunk_with_spacy(text, max_tokens=100, overlap=2):
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
    openai = OpenAI()
    pages = extract_pdf_text(pdf_path)
    all_chunks = []

    texts = []
    for page_data in pages:
        chunks = chunk_with_spacy(page_data["text"])
        for chunk in chunks:
            texts.append(chunk)
            all_chunks.append({"chunk": chunk, "page": page_data["page"]})

    if not len(all_chunks):
        return
    
    embeddings = openai.embeddings.create(input=texts, model="text-embedding-3-large").data

    client = weaviate.connect_to_local()
    try:
        if client.collections.exists(VECTOR_DB):
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
        if client.collections.exists(VECTOR_DB):
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


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if 'upload' in request.form:
            file = request.files['file']
            if file and file.filename.endswith(".pdf"):
                original_name = os.path.splitext(file.filename)[0]
                file_id = slugify(original_name)
                filename = f"{file_id}.pdf"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                delete_from_weaviate(file_id)
                embed_and_store(file_id, file_path)
        elif 'delete' in request.form:
            file_id = request.form['delete']
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file_id + ".pdf"))
            delete_from_weaviate(file_id)

        return redirect(url_for('upload'))

    files = [
        f.replace(".pdf", "") for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith(".pdf")
    ]
    return render_template("upload.html", files=files)

@app.route("/read", methods=["GET", "POST"])
def readFile():
    file_id = request.args.get('file_id', None)
    chunks = []
    if file_id:
        client = weaviate.connect_to_local()
        try:
            offset = 0
            limit = 30
            if client.collections.exists(VECTOR_DB):
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


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/clean", methods=["GET"])
def clearVectorDB():
    client = weaviate.connect_to_local()
    try:
        client.collections.delete_all()
        init_weaviate_schema()
    finally:
        client.close()
    for f in os.listdir(app.config['UPLOAD_FOLDER']):
        if f.endswith(".pdf"):
            file_id = f.replace(".pdf", "")
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file_id + ".pdf"))
    
    return redirect(url_for('upload'))

@app.route("/openai", methods=["POST"])
def openai():
    prompt = request.form.get('prompt')
    result = gptResponse(prompt)
    return jsonify({
        'data': result
    })

@app.route("/rag", methods=["POST"])
def rag():
    context = request.form.get('context')
    openai = OpenAI()
    response = openai.embeddings.create(
        input=context,
        model="text-embedding-3-large"
    )
    vector = response.data[0].embedding
    chunks = []
    client = weaviate.connect_to_local()
    try:
        if client.collections.exists(VECTOR_DB):
            collection = client.collections.get(VECTOR_DB)
            resp = collection.query.near_vector(
                near_vector=vector,
                limit=10,
                return_metadata=MetadataQuery(distance=True)
            )
            for o in resp.objects:
                chunks.append({
                    'file_id': o.properties['file_id'],
                    'page': o.properties['page'],
                    'text': o.properties['chunk'],
                    'distance': o.metadata.distance,
                })
    finally:
        client.close()

    return jsonify({
        'chunks': chunks
    })


def extractPolicies(file_name, page_text, product_text):
    policies = []
    system_role = (
        'You are a compliance policy-matching assistant.\n'
        '\n'
        'Your task is to analyze a product description or ad text and determine '
        'whether it may violate or trigger any rules based on a specific policy page from a regulation document.\n'
        '\n'
        'You will be provided with:\n'
        '- The text of one page from a PDF containing advertising or content policies\n'
        '- The text of a product description or advertisement\n'
        '\n'
        'You must only use the policy content from the provided PDF page as your source.  \n'
        'Do not rely on outside knowledge.  \n'
        'Do not apply rules not found in the page.  \n'
        'Do not assume anything beyond what is clearly stated in the PDF page and the product content.\n'
        '\n'
        'Your job is to:\n'
        '- Match any specific rules, clauses, or language from the PDF policy page to the ad/product text\n'
        '- Identify which policies apply or are potentially violated\n'
        '- Use exact quotes from both the PDF and product content to justify any matches\n'
        '\n'
        'Respond in this exact JSON format:\n'
        '\n'
        '```json\n'
        '{\n'
        '  "matches": [\n'
        '    {\n'
        '      "policy_text": "<quoted sentence or phrase from the PDF policy>",\n'
        '      "product_text": "<quoted sentence or phrase from the product>",\n'
        '      "violation": "<yes or no>",\n'
        '      "reason": "<why this policy is applicable or potentially violated>"\n'
        '    }\n'
        '  ]\n'
        '}\n'
        '```\n\n'
        '\n'
        'If no policy clearly applies, than respond with empty matches.\n'
        '\n'
        '```json\n'
        '{\n'
        '  "matches": []\n'
        '}\n'
        '```\n'
    )
    user_prompt = (
        f'Here is a page from the policy PDF ({file_name}):\n'
        '\n'
        '---\n'
        f'{page_text}\n'
        '---\n'
        '\n'
        'Here is the product or ad content:\n'
        '\n'
        '---\n'
        f'{product_text}\n'
        '---\n'
        '\n'
        'Please identify any relevant matches using only the PDF content above. Follow the JSON format.\n'
    )
    for i in range(3):
        try:
            response = gptResponse(user_prompt, system_role)
            jsonResp = extract_json(response)
            if jsonResp is not None and 'matches' in jsonResp:
                policies_ = []
                for match in jsonResp['matches']:
                    policies_.append({
                        'policy_text' : match['policy_text'],
                        'product_text' : match['product_text'],
                        'violation' : match['violation'],
                        'reason' : match['reason'],
                    })
                
                for row in policies_:
                    policies.append(row)
                return policies
            else:
                print("Wrong JSON")
        except Exception as e:
            print(e)

    return policies

@app.route("/bruteRag", methods=["POST"])
def bruteRag():
    chunks = []
    context = request.form.get('context')
    for f in os.listdir(app.config['UPLOAD_FOLDER']):
        if f.endswith(".pdf"):
            file_id = f.replace(".pdf", "")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_id + ".pdf")
            pages = extract_pdf_text(file_path)

            for page_data in pages:
                policies = extractPolicies(file_id, page_data["text"], context)
                for policy in policies:
                    chunks.append(policy)

    return jsonify({
        'chunks': chunks
    })


@app.route("/resolution", methods=["POST"])
def resolution():
    content = request.form.get('content')
    policy = request.form.get('policy')
    cons = Conversation(content, policy).hasViolation()

    return jsonify({
        'changes': cons
    })

class MarketingHead:
    def get_system_role(self):
        system_role = (
            'You are the Marketing Head of an e-commerce company named "Bodywell Ayurveda".\n'
            'BODYWELL® Ayurveda is India’s leading online ayurvedic store. They offer best quality ayurvedic products and supplements for Immunity, kids, male wellness, female wellness, joints, lactation, weight loss, cardio, diabetes, cholesterol, liver, and more.\n'
            'They are registered with the Ministry of AYUSH and adhere to GMP (Good Manufacturing Practices) standards, ensuring authenticity and safety in every product.\n\n'
            'The General Counsel has approached you after the company received a legal notice concerning some content on the website. While the specific issue isn\'t fully clear, the notice includes a list of compliance policies your content must adhere to.\n\n'
            'The product description or promotional content in question is currently generating high user engagement and driving significant traffic to the website. You believe that altering or removing it could impact business performance. While cooperating with the legal team, '
            'try to defend the value of the content, explore alternative compliant phrasings, and aim to retain the essence of the messaging that is resonating with customers.\n\n\n'
            'Remember:\n'
            '- The product has been available in the market for over six months and has received consistently positive customer feedback.\n'
            '- All claims made on the website are transparently reflected on the product packaging as well.\n'
            '- The product design, logo, and brand identity are original, trademarked, and do not imitate or resemble any well-known or popular brand.\n'
            '- The product has an average customer rating of 4.3 out of 5 across reputable online pharmacy platforms such as Apollo Pharmacy, Netmeds, and 1mg.\n'
            '- This is an Indian-origin brand, developed and manufactured in India.\n'
            '- The image displayed on the packaging accurately represents the actual product contained within.\n'
            '- The packaging does not feature any celebrity endorsement or misleading visual representation.\n\n'
            'Note: Try to keep the changes minimum'
        )
        return system_role

class GeneralCounsel:
    def get_system_role(self):
        system_role = (
            'You are General Counsel of a ecommerce company named "Bodywell Ayurveda".\n'
            'BODYWELL® Ayurveda is India’s leading online ayurvedic store. they offer best quality ayurvedic products and supplement for Immunity, kids, male wellness, female wellness, joints, lactation, weight loss, cardio, diabetes, cholesterol, liver and more.\n'
            'They are registered with the Ministry of AYUSH and adhere to GMP (Good Manufacturing Practices) standards, ensuring authenticity and safety in every product.\n\n'
            'Unfortunately, you have received a legal notice concerning the content on your website. While the exact issue is unclear, the notice includes a list of policies that your company must adhere to. Please get in touch with the Marketing Head, as they are responsible for the website content. Discuss with them which product may have triggered the violation and how to address it.\n\n\n'
            'Remember:\n'
            '- The product has been available in the market for over six months and has received consistently positive customer feedback.\n'
            '- All claims made on the website are transparently reflected on the product packaging as well.\n'
            '- The product design, logo, and brand identity are original, trademarked, and do not imitate or resemble any well-known or popular brand.\n'
            '- The product has an average customer rating of 4.3 out of 5 across reputable online pharmacy platforms such as Apollo Pharmacy, Netmeds, and 1mg.\n'
            '- This is an Indian-origin brand, developed and manufactured in India.\n'
            '- The image displayed on the packaging accurately represents the actual product contained within.\n'
            '- The packaging does not feature any celebrity endorsement or misleading visual representation.\n\n'
            'Note: Try to keep the changes minimum'
        )
        return system_role
    
    

class Conversation:
    def __init__(self, content, policy):
        self.conversation = []
        self.content = content
        self.policy = policy
    
    def get_initial_prompt(self):
        user_prompt = (
            'I am the Marketing Head. I received your email about the content below.\n'
            '\n'
            f'---\n{self.content}---\n'
            '\n'
            f'\nPolicy: "{self.policy}"\n\n'
            'I believe the content is compliant and does not need to be changed.\n'
            '\n'
            'If we don\'t agree, we will keep discussing until we both say "yes".\n'
            'Let’s keep response format consistent without any markdown:\n'
            'Agree: Yes/No\n'
            'Reason: <disagreement reason>\n'
            'Suggestion: <Suggest change, keep them minimal>'
        )
        return user_prompt
    
    def makeConversation(self):
        gc_conversation = []
        mh_conversation = []

        gc_state = 'assistant'
        mh_state = 'user'

        mh_response = self.get_initial_prompt()
        

        while True:
            gc_conversation.append({
                'role': 'user' if gc_state == 'assistant' else 'assistant',
                'content': mh_response
            })

            mh_conversation.append({
                'role': 'user' if mh_state == 'assistant' else 'assistant',
                'content': mh_response
            })
            
            gc_response = gptResponse(gc_conversation, system=GeneralCounsel().get_system_role())
            print("General ",gc_response,"\n\n\n")

            if 'agree: yes' in gc_response.lower():
                return f"{mh_response}\n\n\n{gc_response}"
            
            gc_conversation.append({
                'role': 'user' if gc_state == 'assistant' else 'assistant',
                'content': gc_response
            })
            gc_state = 'user' if gc_state == 'assistant' else 'assistant'

            mh_conversation.append({
                'role': 'user' if mh_state == 'assistant' else 'assistant',
                'content': gc_response
            })
            mh_state = 'user' if mh_state == 'assistant' else 'assistant'

            mh_response = gptResponse(mh_conversation, system=MarketingHead().get_system_role())
            print("Marketing ",mh_response,"\n\n\n")

            if 'agree: yes' in mh_response.lower():
                return f"{gc_response}\n\n\n{mh_response}"
        
    def getConclusion(self):
        agreement = self.makeConversation()
        prompt = (
            f'{agreement}\n\n\n'
            'From above conversation extract the changes to be made in below content:\n\n'
            f'{self.content}\n\n\n'
            'Format your response as:\n'
            '[{"original_line":"<original line from main content>","revised_line":"<change as per agreement in conversation>"}]'
        )

        return  extract_json(gptResponse(prompt), ['[', ']'])

    def hasViolation(self):
        prompt = (
            f'Policy statement: {self.policy}\n\n'
            'Is the below product description in violation of given policy?\n\n'
            f'{self.content}\n\n\n'
            'Respond as Yes or No only, no other explanation\n'
        )
        if 'yes' in gptResponse(prompt).lower():
            return self.getConclusion()
        return []

if __name__ == "__main__":
    app.run(debug=True)
