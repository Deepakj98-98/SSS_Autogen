import os
from dotenv import load_dotenv
#from qdraant_client import QdrantClient, models
from langchain_text_splitters import SpacyTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
import langgraph
from langgraph.graph import StateGraph
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document

class QdrantChunking:
    def __init__(self):
        load_dotenv()
        self.model_name="sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.model_name)

    def load_and_chunk(self,fs_bucket):
        text_splitter=SpacyTextSplitter(chunk_size=200, chunk_overlap=30)
        all_docs=[]

        for file in fs_bucket.find():
            if file.filename.endswith(".txt"):
                content=fs_bucket.open_download_stream(file._id).read().decode("utf-8")
                document = Document(page_content=content, metadata={"source": file.filename})
                chunks = text_splitter.split_documents([document])
                all_docs.extend(chunks)
        
        return all_docs
    
    def store_in_qdrant(self, docs, collection_name):
        qdrant=Qdrant.from_documents(
            documents=docs,
            embedding=self.embedding_model,
            collection_name=collection_name,
            url=os.getenv("ENV_URL"),
            api_key=os.getenv("API_KEY")
        )
        print(f"Stored {len(docs)} chunks in Qdrant collection '{collection_name}'.")
        return qdrant
    
    

    def load_and_chunks(self, state):
        chunks=self.load_and_chunk(state["fs_bucket"])
        state["chunks"]=chunks
        return state
    
    def embed_and_store(self,state):
        qdrant_store=self.store_in_qdrant(state["chunks"],state["collection_name"])
        state["qdrant"]=qdrant_store
        return state
    
    def builder_graph(self,fs_bucket, collection_name):
    
        builder=StateGraph(dict)
        builder.add_node("chunk_and_load",self.load_and_chunks)
        builder.add_node("store_in_qdrant",self.embed_and_store)

        builder.set_entry_point("chunk_and_load")
        builder.add_edge("chunk_and_load","store_in_qdrant")
        builder.set_finish_point("store_in_qdrant")

        compiled_graph=builder.compile()

            # Run
        state = {
            "fs_bucket": fs_bucket, 
            "collection_name": collection_name
        }

        result = compiled_graph.invoke(state)
'''
if __name__ == "__main__":
    qc = QdrantChunking()
    result_state = qc.builder_graph("uploads", "test_check")'
    '''