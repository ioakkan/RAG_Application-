import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import logging
import sys
import toml
from logging.handlers import RotatingFileHandler





#App Class
class SciFiExplorer:
 
 def __init__(self,filepath:str,persist_dir,collection_name): 

    load_dotenv() #loading the env variables
 
    self.filepath = filepath
    self.persist_dir = persist_dir
    self.collection_name = collection_name
    self.embedding_model =  OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_base=os.getenv("OPENAI_ENDPOINT"),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
  
    self.vectorstore = None 
    self.llm  = ChatOpenAI(
    model_name="gpt-4-turbo", 
    openai_api_base=os.getenv("OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.5
    ) 
    self.retriever = None

# converting a txt file to document
 def text_to_document(self):
        logging.debug(f"Data Ingestion process Initialized")
        loader = TextLoader(self.filepath,encoding="utf-8")
        return loader.load() 
 
# loading-converting multiple .txt files to documents 
 def texts_to_documents(self):
  logging.debug(f"Data Ingestion process Initialized")
  loader = DirectoryLoader(
        self.filepath,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

  return loader.load() 

 #splitting the doc files to chunks
 def chunk_document(self,documents,chunk_size=250,chunk_overlap=50):
    logging.debug(f"Chunking process Initialized ")
    text_splitter = RecursiveCharacterTextSplitter(
         chunk_size = chunk_size,
         chunk_overlap= chunk_overlap
         )
    return text_splitter.split_documents(documents)
 
 # building the vectorstore(chromadb)  and saving it to the persist directory
 def build_vectorstore(self,doc_chunks):
     logging.debug(f"builing the vectorstore process initialized")

     vectorstore =Chroma.from_documents(
     documents=doc_chunks, 
     embedding=self.embedding_model,
     collection_name= self.collection_name,
     persist_directory=self.persist_dir) 
     self.vectorstore = vectorstore



 # Setting the vectorstore either by building a new  or loading an existing one from the persist_dir
 def setup_vectorstore(self,docs):
        
        logging.debug(f"vector_store set up  initialized")
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir): #checking if there is an existing collection-vectorstore
            
            self.load_vectorstore()
            logging.debug(f"vector store was loaded succesfully")
        else :
             
              self.build_vectorstore(docs)
              logging.debug(f"builing the vectorstore was successful")

  #Loading an existing vectorstore from persist directory   
 def load_vectorstore(self): 
    logging.debug(f"vectorstore exists, loading process started")
    self.vectorstore = Chroma(
    persist_directory=self.persist_dir,
    embedding_function=self.embedding_model,
    collection_name= self.collection_name
 )


 #Function simulating  the ingestion process using existing functions
 def ingest(self):
       
       doc= self.text_to_document()
       logging.debug(f"{self.filepath} is converted to Document ")

      
       chunks = self.chunk_document(doc)
       logging.debug(f"Document turned to chunks. Number of Docs: {len(chunks)} \n")

       self.setup_vectorstore(chunks)
       logging.debug(f"vector store setup is completed")

 #setting up the vectorstore's retriever (MMR)
 def build_retriever(self):
     logging.debug(f"setting up the chromadb's retriever ") 
     self.retriever=self.vectorstore.as_retriever( #return  
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.5
        }
    ) 
     logging.debug(f"retriever is built ")
   
  
# retrieving docs from vectorstore  and logging them based on query
 def retrieve_docs(self,query):  
        logging.debug(f"Retrieval of  the docs has started")
        docs = self.retriever.invoke(query)
    
        logging.debug(f"===== RETRIEVED DOCS ===== \n the number of  retrieved docs are: {len(docs)}")

        for i, doc in enumerate(docs, 1):
          logging.debug(f"\n--- Doc {i} ---\n{doc.page_content[:250]}")

        context= "\n\n".join(doc.page_content for doc in docs) #retrieving all the context  of the docs as one whole string
        return ({"context":context,"question":query})


 def build_chain(self):
      logging.debug('building  the chain \n')


      #formatting the chat prompt for the llm model  and specifying it to act as a n assistant using the "system" 
      prompt = ChatPromptTemplate.from_messages([
      (
        "system",
        "You are a helpful assistant. "
        "Answer ONLY using the provided context. "
        "If the answer is not in the context, say 'I don't know'."
      ),
      (
        "human",
        "Context:\n{context}\n\n Question:\n{question}"
      )
        ])
      
        
     # Building the chain (LCEL-Langchain Expression Language )
      chain = (
        RunnableLambda(self.retrieve_docs)  #   forwarding  the user's questions and all the docs that were retrieved as a dictionary to the prompt
        | prompt                              # formatting the prompt for the model with the retrieved  context and  the users question
        | self.llm                            # LLM generates output 
        | StrOutputParser()                   # getting only the context llm generates as a string (retrieving only the response.content from the llm) 
        )
 
      return chain
 
 
    
 def scifi_explore(self,query) :
      rag_chain = self.build_chain() #Formating  the chain(LCEL) for execution
      logging.debug('chain is ready for execution \n')
   
      logging.debug(f"Invoking the chain")
      response = rag_chain.invoke(query) 
      logging.debug(f"Chain invoke was successfull below are the results:")
      logging.debug(" Answer:\n%s\n", response) 
    

  
 def setup_logger(self):
    config_path = "config/logger_config.toml"  # Path to the configuration file.
    logs_dir = "logs"  # Directory where log files will be stored.

    #getting the loggers info setup
    with open(config_path, 'r', encoding='utf-8') as f:
        config = toml.load(f)

    #assigning the logger info to format the formmater
    log_level = config.get('log', {}).get('level', 'INFO')
    log_format = config.get('log', {}).get('format', '%(asctime)s - %(levelname)s - %(message)s')
    log_file = config.get('log', {}).get('file', 'rag.log')

    os.makedirs(logs_dir, exist_ok=True)
    logging_file = os.path.join(logs_dir, log_file)
    # Creating the Rotating handlers and settting up  their format
    handler = RotatingFileHandler(logging_file, maxBytes=40000, backupCount=10, encoding='utf-8')
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())
    
  #removing any handlers stored  previously
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
  #setting up the new handlers for the logger   
    root_logger.addHandler(handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
 

# main app  
if __name__ == "__main__":
 filepath = "data/moby_dick.txt" #filepath to the data
 persist_dir = 'chroma_db' #file where the vectorstore with the text embeddings is saved
 collection_name = "scifi" #chromadb's collection name
 SciFi_Explorer = SciFiExplorer(filepath=filepath,persist_dir=persist_dir,collection_name=collection_name) #Initializing the SciFiExplorer object
 SciFi_Explorer.setup_logger() #setting up the app logger
 SciFi_Explorer.ingest() # loading the txt file  and turn
 SciFi_Explorer.build_retriever() #setting the MMR of chromadb as our retriever

 while True:
    query = input("Ask a question (or 'exit'): ")
    logging.debug(f" \n The user entered his query \n")
    if query.lower() == "exit":
        break
    SciFi_Explorer.scifi_explore(query)

 



