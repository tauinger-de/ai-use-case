from langchain_huggingface.llms import HuggingFacePipeline

print("Loading model...")
hf = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-2-9b",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 10},
)


from langchain_core.prompts import PromptTemplate

promptTemplate = PromptTemplate.from_template(
    """
    Question: {question}
    Answer: Let's think step by step.
    """)

chain = promptTemplate | hf

question = "What is electroencephalography?"

print("Asking question: " + question)
print(chain.invoke({"question": question}))