import numpy as np
import json
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter #using for text splitter only
from langchain_community.embeddings import BedrockEmbeddings
import boto3



def get_embeddings(bedrock, index, text):
    body_text = json.dumps({"inputText": text})
    modelId = 'amazon.titan-embed-text-v1'
    accept = 'application/json'
    contentType='application/json'

    response = bedrock.invoke_model(body=body_text, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')

    results=index.add(embedding)

    return results



def split_transcript(subtitles):

    #split_into = len(subtitles)//100 # splitting into 20 for now. At some point i will make that dynamic based on character count of the input

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=20000, #Testing with hard coded 500
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    parts = text_splitter.create_documents([subtitles])

    return parts

def index_document(document):
    subtitle_doc = split_transcript(document)

    #setup titan embedding via langchain - using langchains libraries for ease of use indexing and retrieving from FAISS
    embeddings = BedrockEmbeddings(
        region_name="us-east-1"
    )
    db = FAISS.from_documents(subtitle_doc, embeddings)
    db.save_local("faiss_index")
    