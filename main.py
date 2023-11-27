import asyncio
from fastapi import FastAPI, Query, Response, status, Request
from fastapi.responses import StreamingResponse
from llama_index import VectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext
from langchain.chat_models import ChatOpenAI
from pathlib import Path
from llama_index import download_loader
from fastapi.responses import StreamingResponse
from llama_index import StorageContext, load_index_from_storage

app = FastAPI()


import os
import openai

os.environ['OPENAI_API_KEY'] = 'your_openai_key'

openai.api_key = "your_openai_key"

llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True))

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size=512)

#documents_K12 = SimpleDirectoryReader('K12').load_data()

#index_K12 = VectorStoreIndex.from_documents(documents_K12, service_context=service_context)

#index_K12.storage_context.persist(persist_dir="indexk12")


# rebuild storage context
storage_context_indexK12 = StorageContext.from_defaults(persist_dir="indexk12")

# load index
index_K12 = load_index_from_storage(storage_context_indexK12)


query_engine_K12 = index_K12.as_query_engine(
    service_context=service_context,
    similarity_top_k=5,
    streaming=True,
)

@app.get("/question_K12/")
def run_long_task(request: Request):
    queryparam = request.query_params.get("question_K12")
    queryparam = "Answer in excessive detail and the format should be paragraphs containing bullet points: "+queryparam

    def stream_generator():
        response1 = query_engine_K12.query(queryparam)
        for chunk in response1.response_gen:
            yield chunk


    return StreamingResponse(stream_generator(), media_type='text/plain')





# rebuild storage context
storage_context_indexResearch = StorageContext.from_defaults(persist_dir="indexresearch")

# load index
index_Research = load_index_from_storage(storage_context_indexResearch)

query_engine_Research = index_Research.as_query_engine(
    service_context=service_context,
    similarity_top_k=5,
    streaming=True,
)

@app.get("/question_Research/")
def run_long_task(request: Request):
    queryparam = request.query_params.get("question_Research")
    queryparam = "Answer in excessive detail and the format should be paragraphs containing bullet points: "+queryparam

    def stream_generator():
        response1 = query_engine_Research.query(queryparam)
        for chunk in response1.response_gen:
            yield chunk


    return StreamingResponse(stream_generator(), media_type='text/plain')





# rebuild storage context
storage_context_indexStudent = StorageContext.from_defaults(persist_dir="indexstudent")

# load index
index_Student = load_index_from_storage(storage_context_indexStudent)

query_engine_Student = index_Student.as_query_engine(
    service_context=service_context,
    similarity_top_k=5,
    streaming=True,
)

@app.get("/question_Student/")
def run_long_task(request: Request):
    queryparam = request.query_params.get("question_Student")
    queryparam = "Answer in excessive detail and the format should be paragraphs containing bullet points: "+queryparam

    def stream_generator():
        response1 = query_engine_Student.query(queryparam)
        for chunk in response1.response_gen:
            yield chunk


    return StreamingResponse(stream_generator(), media_type='text/plain')
