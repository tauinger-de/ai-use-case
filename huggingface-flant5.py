import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms.huggingface_hub import HuggingFaceHub

# Notice that “chat_history” is present in the prompt template
template = """You are a friendly chatbot engaging in a conversation with a human.
Question: {question}
Response:"""
promptTemplate = PromptTemplate.from_template(template)

llm = HuggingFaceHub(
    repo_id="google/flan-t5-xxl",
    model_kwargs={"temperature": 1.0, "max_length": 64}
)
chain = promptTemplate | llm

while True:
    query = input("User-query: ")
    # Enter 'exit' to stop
    if query.lower() == "exit":
        break
    print(chain.invoke({"question": query}))