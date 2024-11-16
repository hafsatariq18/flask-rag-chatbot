from flask import Flask, render_template, request, jsonify
import os
import getpass
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings 
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

app = Flask(__name__)

os.environ["OPENAI_API_KEY"] = getpass.getpass()
rag_chain = None
model = ChatOpenAI(model="gpt-3.5-turbo")


@app.route("/upload", methods=["POST"])
def upload_document():
    global rag_chain
    file = request.files.get("file")
    if not file or not file.filename.endswith(".pdf"):
        return jsonify({"error": "Please upload a valid PDF file."}), 400

    # Save and process the uploaded file
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use two sentences maximum and keep the "
        "answer concise.\n\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return jsonify({"message": "Document uploaded and processed successfully!"}), 200


@app.route("/")
def index():
    return render_template('index.html')


@app.post('/ask')
def ask():
    global rag_chain
    if not rag_chain:
        return jsonify({"error": "No document uploaded yet. Please use the /upload endpoint to upload a PDF first."}), 400

    user_input = request.form.get("user_input")
    if not user_input:
        return jsonify({"error": "Please provide a question."}), 400

    response = rag_chain.invoke({"input": user_input})
    answer = response.get("answer") if response else "Sorry, I couldn't find an answer."
    return render_template('index.html', answer=answer)


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)  # Ensure uploads directory exists
    app.run(debug=True, host="0.0.0.0", port=5001)
