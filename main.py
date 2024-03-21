import os

from dotenv import load_dotenv
import os
import openai
import pandas as pd
from llama_index.core.query_engine import PandasQueryEngine

from prompts import new_prompt,instruction_str, context

from note_engine import note_engine


from llama_index.core.tools import QueryEngineTool ,ToolMetadata
from llama_index.core.agent import ReActAgent 
from llama_index.llms.openai import OpenAI
from pdf import canada_engine
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


load_dotenv()



# Setting the OpenAI embedding model globally
#Settings.embed_model = OpenAIEmbedding()

#Settings.embed_model = HuggingFaceEmbedding(
#model_name="BAAI/bge-small-en-v1.5"
#)


population_path = os.path.join("data","WorldPopulation2023.csv")
population_df = pd.read_csv(population_path)

population_query_engine = PandasQueryEngine(df=population_df,verbose=True)
population_query_engine.update_prompts({"pandas_prompt":new_prompt})
population_query_engine.query("what is the population of canada?")

tools = [ 
    
    note_engine,
         QueryEngineTool(query_engine=population_query_engine , metadata=ToolMetadata(
             name = "population_data",
             description="this gives information at the world population and demographics"
         ),),
         
         QueryEngineTool(query_engine=canada_engine , metadata=ToolMetadata(
             name = "canada_data",
             description="this gives detailed information about the country "
         ),),         
         
         ]

llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = ReActAgent.from_tools(tools,llm=llm,verbose=True,context=context)

while (prompt := input("Enter a prompt (q to quit) : ")) != "q":
    result = agent.query(prompt)
    print(result)
