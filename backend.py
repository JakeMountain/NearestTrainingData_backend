from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import openai
import os
import random
import re
from threading import Lock
from urllib.parse import quote
from urllib.parse import unquote
import os
import pinecone
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

load_dotenv()

# helper class to connect to pinecone before querying
class PineconeConnector:
    def __init__(self):
        self.PINECONE_KEY = os.getenv("PINECONE_KEY")
        self.PINECONE_ENV = os.getenv("PINECONE_ENV")
        pinecone.init(
            api_key=self.PINECONE_KEY,
            environment=self.PINECONE_ENV
        )
        self.index_name = 'semantic-search'
        # now connect to the index
        self.index = pinecone.GRPCIndex(self.index_name)

    def query(self, query_embedding):
        return self.index.query(query_embedding, top_k=5, include_metadata=True)

class Encoder:
    def __init__(self):
        # Preload encoder model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

pinecone_connector = PineconeConnector()
encoder = Encoder()

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://localhost",
    "https://localhost:3000",
    "http://https://nearest-training-data.uw.r.appspot.com/",
    "https://https://nearest-training-data.uw.r.appspot.com/"
]
origins_TEST_FULLY_PERMISSIVE = [
    '*',  # This is a test, so we want to allow all origins
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins_TEST_FULLY_PERMISSIVE,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.put("/prompt")
def read_prompt(prompt: str = ""):
        # given a user prompt as a parameter, encode the prompt using bert, query pinecone for the top 5 results,     

    # query openai API to generate a response
    OPENAI_KEY = os.getenv("OPENAI_KEY")
    openai.api_key = OPENAI_KEY
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0.5
    )
    #get the response text
    response_text = response['choices'][0]['text']
    # query OpenAI to generate a response, and return the response along with the top 5 results from pinecone.

    response_embedding = encoder.model.encode(response_text).tolist()

    # query the index
    top_k = pinecone_connector.query(response_embedding).to_dict()["matches"]
    # response_embedding = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    # top_k = [{"id":1, "score":0.9, "embedding":[1,2,3,4,5,6,7,8,9,10], "metadata": {"text": "yes"}}]
    return {"top_k":top_k, "response_text":response_text, "response_embedding":response_embedding}
    