import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms.huggingface_hub import HuggingFaceHub

# Notice that “chat_history” is present in the prompt template
template = """You are a friendly chatbot engaging in a conversation with a human.

Previous conversation:
{chat_history}

New human question: {question}
Response:"""

prompt = PromptTemplate.from_template(template)

repo_id = "google/flan-t5-small"
llm = HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature": 0.1, "max_length": 64}
)

memory = ConversationBufferMemory(memory_key="chat_history")
conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)

while True:
    query = input("User-query: ")
    # Enter 'exit' to stop
    if query.lower() == "exit":
        break
    print(conversation({"question": query})['text'])