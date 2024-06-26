import os 
from dotenv import load_dotenv
from llama_index.core import StorageContext,VectorStoreIndex,load_index_from_storage
from llama_index.readers.file import PyMuPDFReader
#from llama_index.readers import PDFReader
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
import pandas as pd

load_dotenv()

def get_index(data,index_name):
    index = None
    if not os.path.exists(index_name):
        print("building index",index_name)
        index = VectorStoreIndex.from_documents(data,show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_name))
        
    return index

reader = PyMuPDFReader()

Settings.embed_model = OpenAIEmbedding()


pdf_path = os.path.join("data","Canada.pdf")




canada_pdf = reader.load(pdf_path)

canada_index = get_index(canada_pdf,"canada")
canada_engine  = canada_index.as_query_engine()