from __future__ import absolute_import
import logging
import re
import os

import torch
from langchain import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain.llms import OpenAI, OpenAIChat
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.tracers import LangChainTracer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredPDFLoader
from langchain.docstore.document import Document

from src.prompt_templates import (
    # the QA prompt
    DEFAULT_PROMPT_TEMPLATE,
    # the condense question prompt
    DEFAULT_CHAT_HISTORY_PROMPT,
)


class ChatBotModel:
    def __init__(self, logger: logging.Logger, model_name=None, embedding_model_name=None):
        self.logger = logger
        if 'gpt-' in model_name:
            self.llm = OpenAI(model_name=model_name, temperature=0.1, max_tokens=-1)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto"
            )
            # generation_config = GenerationConfig.from_pretrained(model_name)
            # generation_config.max_new_tokens = 1024
            # generation_config.temperature = 0.0001
            # generation_config.top_p = 0.95
            # generation_config.do_sample = True
            # generation_config.repetition_penalty = 1.15
            
            text_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                # generation_config=generation_config,
            )
            self.llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.1})

        if embedding_model_name == "text-embedding-ada-002":
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


    def _create_retriever(self, file_path: str):
        document = self._load_file_from_path(file_path)
        page_content = "\n".join([d.page_content for d in document])
        self.logger.info("document loading completed")
        preprocessor = RecursiveCharacterTextSplitter(
            chunk_size=2500,
            chunk_overlap=500,
            separators=["\n\n", "\n"],
            keep_separator=False,
            add_start_index=False,
            strip_whitespace=False
        )
        # splitting document into chunks
        docs = preprocessor.split_documents(document)
        print(len(docs))

        vectorstore_faiss = FAISS.from_documents(documents=docs, embedding=self.embeddings)
        retriever = vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k":3})
        vectorstore_faiss.save_local(folder_path = "./database", index_name= 'fiass-db')
        # vectorstore_faiss_aws =  FAISS.load_local(folder_path = "./database", embeddings = self.embeddings, index_name= 'fiass-db')
        self.logger.info(
            f"vectorstore_faiss: number of elements in the index={vectorstore_faiss.index.ntotal}::"
        )
        return retriever, page_content

    def _load_file_from_path(self, file_path: str):
        self.logger.info(os.path.isfile(file_path))
        self.logger.info("loading file : " + file_path)
        if file_path.lower().endswith(".pdf"):
            pages = PyPDFLoader(file_path).load()
            return [Document("".join([page.page_content for page in pages]))]
        elif file_path.lower().endswith(".txt"):
            return TextLoader(file_path).load()
        else:
            raise TypeError("Only searchable Pdf and Text file are suppoorted")

    def create_qa_instance(
        self,
        file_path: str
    ):
        self.retriever,_ = self._create_retriever(file_path=file_path)
        memory_chain = ConversationBufferWindowMemory(
            memory_key="chat_history", input_key="question", output_key="answer", return_messages=True, k=2
        )
        qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=memory_chain,
            return_source_documents=True,
            # get_chat_history=_get_chat_history,
            # condense_question_llm = self.llm,
            condense_question_prompt=DEFAULT_CHAT_HISTORY_PROMPT,
            combine_docs_chain_kwargs = {"prompt": DEFAULT_PROMPT_TEMPLATE},    # the LLMChain prompt to get the answer
            chain_type="stuff",  # 'refine',
            output_key='answer',
            verbose=True
        )

        return qa