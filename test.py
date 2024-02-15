#!/usr/bin/env python3
from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
import argparse
import time

app = Flask(__name__)

model = os.environ.get("MODEL", "mistral")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

def initialize():
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = Ollama(model=model, callbacks=callbacks)
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

qa = initialize()

@app.route('/api/generate', methods=['POST'])
def generate_response():
    data = request.get_json()
    prompt = data.get('prompt')
    if prompt is None:
        return jsonify({'error': 'Prompt not provided'}), 400
    result = qa(prompt)
    response = result['result']
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30303)
