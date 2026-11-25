from langchain_classic.chains.qa_with_sources import vector_db
import os
import sys
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
# Ensure console can print utf-8 characters/emojis on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

key1=os.getenv("Gemini_KEY")

# Build path relative to this script's folder
file_path = os.path.join(os.path.dirname(__file__), "speech.txt")
file_path2=os.path.join(os.path.dirname(__file__), "medical_report.pdf")
file_path3=os.path.join(os.path.dirname(__file__), "data_science_syllabus.pdf")
loader = TextLoader(file_path, encoding="utf-8")
text1 = loader.load()

onedoc=text1[0]


pdfload=PyPDFLoader(file_path3)

pyd=pdfload.load()
twodoc=pyd[0]


splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)

chunks=splitter.split_text(twodoc.page_content)

# for ck in enumerate(chunks):
#    print(ck)
#    print(end="")


embedder=GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=key1
)
vector=embedder.embed_query(chunks[0])



chm=Chroma.from_texts(
   texts=chunks,
   embedding=embedder,
   persist_directory="./.db_medical"
)


print("Everthing done")
