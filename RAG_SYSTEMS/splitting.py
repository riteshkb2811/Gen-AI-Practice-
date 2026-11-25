from h11._abnf import chunk_size
from langchain_text_splitters import RecursiveCharacterTextSplitter

text="Today, I want to speak from the heart about three pillars that define the future of our nation:our youth, our education system, and our farmers."

splitter= RecursiveCharacterTextSplitter(chunk_size=50,chunk_overlap=20)

chunks=splitter.split_text(text)

print(chunks)