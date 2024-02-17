from langchain_core.prompts.prompt import PromptTemplate

# Build prompt
chat_history_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""

DEFAULT_CHAT_HISTORY_PROMPT = PromptTemplate.from_template(chat_history_template)


# Build prompt
prompt_template = """Follow the following set of instructions that describes a task:
1. Use the context provided below to answer to the question at the end.
2. Do not add any explaination/justification for answers and do not attempt to asnwer more than than what is asked.
3. Do not add anything out of the given context.

Context:
{context}

Question: {question}
Answer:"""

DEFAULT_PROMPT_TEMPLATE = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
