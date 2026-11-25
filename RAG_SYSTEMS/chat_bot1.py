from langchain_core.runnables import chain
import os
from doc_load import chm 
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
key1=os.getenv("GroqKEY")
llm=ChatGroq(model="openai/gpt-oss-20b",groq_api_key=key1)

def contextgen(query:str):
    data=chm.similarity_search(query,k=2)
    context=""
    for doc in data:
       context+=doc.page_content+"\n"
    return{
      "context":context,
      "query":query     
    }

prompt1=PromptTemplate.from_template("""u are a helpful assistant , answer based on the following Context :{context} ,Question: {query}""")

chain1= contextgen | prompt1 | llm

while True:
  query=input("enter you query:")
  if query.lower()=="exit":
     break
  res=chain1.invoke(query)
  print("  ")
  print("AI: ",res.content)
  print("  ")