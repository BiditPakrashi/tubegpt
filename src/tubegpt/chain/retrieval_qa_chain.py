
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

class RetrievalQAChain():
    def __init__(self) -> None:
        pass

    def create_chain(self,retreiver,reader, prompt):
        chain_type_kwargs = {"prompt": prompt}
        self.chain = RetrievalQA.from_chain_type(
            #llm=ChatOpenAI(temperature=0),
            llm=reader,
            chain_type="stuff",
            retriever=retreiver,
            chain_type_kwargs=chain_type_kwargs)
        
    def get_chain(self):
        return self.chain
        
    def query(self,question):
        response = self.chain.run(question)
        return response