import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.documents import Document
import logging
import sys
import toml
from logging.handlers import RotatingFileHandler
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional



# App Class

class SciFiExplorer(BaseModel):
 
 model_config = ConfigDict(arbitrary_types_allowed=True) # Allows to use langchain objects with pydantic
 load_dotenv() # Loading the env variables
 filepath: str
 persist_dir: str
 collection_name: str
 chunk_size: int = Field(default=250, ge=1)
 chunk_overlap: int = Field(default=20, ge=0)
 search_type: str = Field(default="mmr")
 search_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.5, 
        }
    )
 llm: Optional[ChatOpenAI] = None
 embedding_model: Optional[OpenAIEmbeddings] = None
 vectorstore: Optional[Chroma] = None
 retriever: Optional[Any] = None

 def model_post_init(self, __context: Any) -> None:
        
        """
        Functions use: 
        Runs after Pydantic validation.
        Used to initialize the llm and embedding_model

        """
        load_dotenv()

        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_base=os.getenv("OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

        self.llm = ChatOpenAI(
            model_name="gpt-4-turbo",
            openai_api_base=os.getenv("OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.2, # Model is almost deterministic  and designed to  creativly reply  based on context from documents (not fabricate)  .
        )


# Converting a txt file to document
 def text_to_document(self) -> List[Document]:
        
        logging.debug(f"Data Ingestion process Initialized")
        loader = TextLoader(self.filepath,encoding="utf-8")
        return loader.load() 
 
# Converting multiple .txt files to documents 
 def texts_to_documents(self) -> List[Document]:
  
  logging.debug(f"Data Ingestion process Initialized")
  loader = DirectoryLoader(
        self.filepath,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

  return loader.load() 

 # Splitting the doc files to chunks
 def chunk_document(self,docs: List[Document]) -> List[Document]:

    logging.debug(f"Chunking process Initialized ")
    text_splitter = RecursiveCharacterTextSplitter(
         chunk_size = self.chunk_size,
         chunk_overlap= self.chunk_overlap
         )
    return text_splitter.split_documents(docs)
 
 # Building the vectorstore(chromadb)  and saving it to the persist directory
 def build_vectorstore(self,doc_chunks: List[Document]) -> None:
     
     logging.debug(f"builing the vectorstore process initialized")
     vectorstore =Chroma.from_documents(
     documents=doc_chunks, 
     embedding=self.embedding_model,
     collection_name= self.collection_name,
     persist_directory=self.persist_dir) 
     self.vectorstore = vectorstore

 # Setting the vectorstore either by building a new  or loading an existing one from the persist_dir
 def setup_vectorstore(self,docs: List[Document]) -> None:
        
        logging.debug(f"vector_store set up  initialized")
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir): # Checking if there is an existing collection-vectorstore
            
            self.load_vectorstore()
            logging.debug(f"vector store was loaded succesfully")
        else :
             
              self.build_vectorstore(docs)
              logging.debug(f"builing the vectorstore was successful")

 # Loading an existing vectorstore from persist directory   
 def load_vectorstore(self) -> None: 
    
    logging.debug(f"vectorstore exists, loading process started")
    self.vectorstore = Chroma(
    persist_directory=self.persist_dir,
    embedding_function=self.embedding_model,
    collection_name= self.collection_name
 )

 # Function simulating  the ingestion process using existing functions
 def ingest(self) -> None:
       
       doc = self.texts_to_documents()
       logging.debug(f"checking this length {len(doc)}")
       
       logging.debug(f"{self.filepath} is converted to Document ")
       
       chunks = self.chunk_document(doc)
       logging.debug(f"Document turned to chunks. Number of Docs: {len(chunks)} \n")

       self.setup_vectorstore(chunks)
       logging.debug(f"vector store setup is completed")

 # Setting up the vectorstore's retriever (MMR)
 def build_retriever(self) -> None:
     
     logging.debug(f"setting up the chromadb's retriever ") 
     self.retriever=self.vectorstore.as_retriever( 
        search_type= self.search_type ,
        search_kwargs= self.search_kwargs
    ) 
     logging.debug(f"retriever is built ")

  # Logging the retrieved docs
 def log_docs(self,docs) -> None:
         
         logging.debug(f"===== RETRIEVED DOCS ===== \n the number of  retrieved docs are: {len(docs)}")

         for i, doc in enumerate(docs, 1):
          logging.debug(f"\n--- Doc {i} ---\n{doc.page_content[:self.chunk_size]}")

# Retrieving docs from vectorstore  
 def retrieve_docs(self,query: str) -> Dict[str, str]:  
        
        logging.debug(f"Retrieval of  the docs has started")
        docs = self.retriever.invoke(query)  # docs = retriever.get_relevant_documents(query) for older version
        self.log_docs(docs)
        context= "\n\n".join(doc.page_content for doc in docs) # Retrieving all the context  of the docs as one whole string
        return ({"context":context,"question":query})

 # Building the chain(LCEL) with the runnable components 
 def build_chain(self) -> Runnable:
      
      logging.debug('building  the chain \n')

      # Formatting the chat prompt for the llm model  and specifying it to act as a n assistant using the "system" 
      prompt = ChatPromptTemplate.from_messages([
      (
        "system",
        "You are a creative science fiction analyst helping a writer explore classic sci-fi themes."
        "Answer strictly using the provided story excerpts. "
        "Be concise and insightful. "
        "If the answer cannot be found, say: 'I dont know based on the provided texts.'"
      ),
      (
        "human",
        "Context:\n{context}\n\n Question:\n{question}"
      )
        ])
    
     # Building the chain (LCEL-Langchain Expression Language )
      chain = (
        self.retrieve_docs   # Forwarding  the user's questions and all the docs that were retrieved as a dictionary to the prompt
        | prompt             # Formatting the prompt for the model with the retrieved  context and  the users question
        | self.llm           # LLM generates output 
        | StrOutputParser()  # Getting only the context llm generates as a string (retrieving only the response.content from the llm) 
        )
      return chain
 
 
  # Function simulating the RAG_application for exploration  by invoking the chain
 def scifi_explore(self,query: str) -> str:
      
      rag_chain = self.build_chain() # Formating  the chain(LCEL) for execution
      logging.debug('chain is ready for execution \n')
      logging.debug(f"Invoking the chain")
      response = rag_chain.invoke(query) 
      logging.debug(f"Chain invoke was successfull below are the results:")
      logging.debug(" Answer:\n%s\n", response) 
    

  # Logger Setup
 def setup_logger(self) -> None:

    config_path = "config/logger_config.toml"  # Path to the configuration file.
    logs_dir = "logs"  # Directory where log files will be stored.

    # Getting the loggers info setup
    with open(config_path, 'r', encoding='utf-8') as f:
        config = toml.load(f)

    # Assigning the logger info to format the formmater
    log_level = config.get('log', {}).get('level', 'INFO')
    log_format = config.get('log', {}).get('format', '%(asctime)s - %(levelname)s - %(message)s')
    log_file = config.get('log', {}).get('file', 'rag.log')
    os.makedirs(logs_dir, exist_ok=True)
    logging_file = os.path.join(logs_dir, log_file)

    # Creating the Rotating handlers and settting up  their format
    handler = RotatingFileHandler(logging_file, maxBytes=40000, backupCount=20, encoding='utf-8')
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())
    
  # Removing any handlers stored  previously
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

  # Setting up the new handlers for the logger   
    root_logger.addHandler(handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
 

# main app  
if __name__ == "__main__":
 filepath = "data/" # Filepath to the data
 persist_dir = 'chroma_db' # File where the vectorstore with the text embeddings is saved
 collection_name = "scifi" # Chromadb's collection name
 SciFi_Explorer = SciFiExplorer(filepath=filepath,persist_dir=persist_dir,collection_name=collection_name) #Initializing the SciFiExplorer object
 SciFi_Explorer.setup_logger() # Setting up the app logger
 SciFi_Explorer.ingest() # Loading the txt file  and turn
 SciFi_Explorer.build_retriever() #Setting the MMR for chromadb as our retriever

 while True:
    query = input("Ask a question (or 'exit'): ")
    logging.debug(f" \n The user entered his query \n")
    if query.lower() == "exit":
        break
    SciFi_Explorer.scifi_explore(query)

 



