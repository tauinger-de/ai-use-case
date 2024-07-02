from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS

print("Hi")
loader = PyPDFLoader("https://schnittstelle.beste-gesundheit.at/files/XW/Downloads/Laufsport-Magazin/Core-Training.pdf")
pages = loader.load_and_split()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(pages, embeddings)

query = "Worum geht es hier?"
doc = db.similarity_search(query)[0]
#print(doc)

llm = OpenAI()
chain = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever())
#prompt = "Was wird durch Core-Training gesteigert?"
prompt = "Was gibt es zum Sonderpreis?"
result = chain.invoke(prompt, return_only_outputs=True)
print(result)