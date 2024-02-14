import json
import time
import requests

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
import chromadb
import argparse

model = os.environ.get("MODEL", "mistral")
# For embeddings model, the example uses a sentence-transformers model
# https://www.sbert.net/docs/pretrained_models.html 
# "The all-mpnet-base-v2 model provides the best quality, while all-MiniLM-L6-v2 is 5 times faster and still offers good quality."
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS

def main():
    # Use a different port number if necessary
    server_url = "http://123.123.123.123:30303/api/generate"

    while True:
        try:
            # Send POST request
            prompt = input("Enter a prompt (or 'exit' to quit): ")
            if prompt.strip() == "exit":
                break

            data = {"model": "sroecker/sauerkrautlm-7b-hero", "prompt": prompt}
            response = requests.post(server_url, json=data)

            # Check for successful response
            if response.status_code == 200:
                process_response(response.json())
            else:
                print(f"Error: Server returned status code {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"Error sending request: {e}")

def process_response(response_json):
    # Check for "done":true
    if not response_json["done"]:
        print("Error: Server response not complete")
        return

    # Extract and print the text response
    text_response = response_json["response"]
    print(f"> Generated text:\n{text_response}")

if __name__ == "__main__":
    main()
