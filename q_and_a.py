
import json
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter #using for text splitter only
from langchain_community.embeddings import BedrockEmbeddings
import boto3
import botocore
import os
from dotenv import load_dotenv
import streamlit as st


# loading in environment variables
load_dotenv()
# setting default session with AWS CLI Profile
boto3.setup_default_session(profile_name=os.getenv('profile_name'))
# Setup Bedrock client
config = botocore.config.Config(connect_timeout=300, read_timeout=300)
bedrock = boto3.client('bedrock-runtime' , 'us-east-1', config = config)

def invoke_llm(query, context):
    system_prompt=f"""
You are a Data Processor. You will be asked a question from a user and provided context you will use to answer that question
Answer the question provided to the best of your ability
Your response should be simple, concise, well organized and designed for human readability (using paragraph breaks and bullet points where it makes sense)
If the context doesn't contain the answer to the question, honestly say so
Make sure to surround your response in <output></output> tags

User Question:
<user_question> 
{query} 
</user_question>


return the your answer in <output> xml tags, only including the text for the answer
Make sure to surround your answer in <output></output> tags

"""

    
    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": 0.5,
        "system": system_prompt,
        "messages": [    
            {
                "role": "user",
                "content": f"""
                <context> {context} </context>
                """
            }
        ]
    }

    prompt = json.dumps(prompt)

    print(prompt)
    print("------------------------------------------------------")

    response = bedrock.invoke_model(body=prompt, modelId="anthropic.claude-3-sonnet-20240229-v1:0", accept="application/json", contentType="application/json")
    response_body = json.loads(response.get('body').read())
    llmOutput=response_body['content'][0]['text']

    print(llmOutput)

    
    output = parse_xml(llmOutput, "output")

    if output is not None:
       return output
    else:
       return llmOutput



def parse_xml(xml, tag):
  start_tag = f"<{tag}>"
  end_tag = f"</{tag}>"
  
  start_index = xml.find(start_tag)
  if start_index == -1:
    return ""

  end_index = xml.find(end_tag)
  if end_index == -1:
    return ""

  value = xml[start_index+len(start_tag):end_index]
  return value



def answer_question(query):
    with st.status("Getting Answer ", expanded=False, state="running") as status:
        # Load the vector store from the FAISS directory
        #embeddings = BedrockEmbeddings(
        #    region_name="us-east-1"
        #)
        #db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        #status.update(label="Retrieving Relevant Context", state="running", expanded=False)

        # Retrieve the most relevant segment of document
        #context = db.similarity_search(query)
        if 'text' in st.session_state:
            context = st.session_state.text
            st.write(f":heavy_check_mark: Using Active Document Context")
        else:
            embeddings = BedrockEmbeddings(
                region_name="us-east-1"
            )
            db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            context = db.similarity_search(query)
            st.write(f":heavy_check_mark: Using Vector Store Context")

        st.write(f":heavy_check_mark: Context Retrieved: {context}")

        status.update(label="Invoking LLM", state="running", expanded=False)

        #Cal the LLM wit hthe question and context
        llm_output = invoke_llm(query, context)

        status.update(label=":heavy_check_mark: Answer Generated", state="complete", expanded=False)
        st.write(":heavy_check_mark: Answer Generated")

    # Print the answer from the document
    st.write(llm_output)


