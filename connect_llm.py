import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


HF_token=os.getenv("HF_TOKEN")

hugging_face_repo_id="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(hugging_face_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=hugging_face_repo_id,
        task="text-generation",
        model_kwargs={"temperature":0.7,"max_new_tokens":500},
        huggingfacehub_api_token=HF_token
    )
    return llm

DB_FAISS_PATH = 'vectorstores/db_faiss'

custom_prompt_template="""
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=["context", "question"])
    return prompt

def connect_llm():
    llm = load_llm(hugging_face_repo_id)
    prompt = set_custom_prompt()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        verbose=True,
        return_source_documents=True
    )
    return chain

user_question = "What is the meaning of life?"

chain = connect_llm()
response = chain.invoke({"query": user_question})
print("result:", response["result"])

