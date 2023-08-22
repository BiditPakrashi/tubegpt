from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings

from typing import List


class ChromaDB():
    def __init__(self):
        pass

    def save_embeddings(self,docs:List,embeddings, persist_dir):
 
        self.db = Chroma.from_documents(
            documents=docs, embedding= embeddings, persist_directory=persist_dir
        )
        self.embedding = embeddings

    def presist_db(self):
        self.db.persist()

    def load_db(self, persist_dir):
        self.db = Chroma(persist_directory=persist_dir, embedding_function=self.embeddings)


## example
#chroma = ChromaDB()
#embeddings = OpenAIEmbeddings()
#MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
#embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
#chroma.save_embeddings(['test'],embeddings,"./tubegptdb")
#chroma.presist_db()
