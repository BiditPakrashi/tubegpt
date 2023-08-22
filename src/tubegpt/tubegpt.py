
from src.tubegpt.audio.simple_audio_info_extractor import SimpleAudioInfoExtractor
from src.tubegpt.chain.llm_chain import LLMChain
from src.tubegpt.chain.retrieval_qa_chain import RetrievalQAChain
from src.tubegpt.embedding.chromadb import ChromaDB
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.tubegpt.prompt.model_prompt import ModelPrompt
from src.tubegpt.video.simple_video_information_extractor import SimpleVideoInfoExtractor
import os
import openai       
from dotenv import load_dotenv, find_dotenv

# Environment Variables
TIME_WINDOW = 10 # data for audio and vidio will be consolidated in this time (in secs)
CAPTION_PER_IMAGE =  3 #for LAVIS blip 2 model
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

class TubeGPT():
    def __init__(self,vid_url,save_dir) -> None:
        self.vid_url:str = vid_url
        self.save_dir:str = save_dir
        self.audio_chain:RetrievalQAChain
        self.vision_description_chain:RetrievalQAChain
        self.vision_chain:RetrievalQAChain



    def process_audio(self,vid_url, save_dir,audio_extractor=SimpleAudioInfoExtractor(), embeddings=OpenAIEmbeddings()):
        '''
        processes audio to first get captions by using audio parser like openaiwhisper extractor provided
        then and then the documents are saved as embeddings in chromadb
        '''  
        docs = audio_extractor.audio_to_text(vid_url,save_dir=save_dir)
        self.__save_audio_embeddings(docs,embeddings=embeddings)

    # save audio embeddings
    def __save_audio_embeddings(self,docs,embeddings):
        self.audio_db = ChromaDB()
        self.audio_db.save_embeddings(docs,embeddings,self.save_dir)
    
    #2. Query audio
    def query_audio(self,question:str,reader)->str:
        '''
        queries using audio reader retrieval model and returns the response
        ''' 
        if(self.audio_chain):
            return self.audio_chain.query(question)
        
        self.__get_audio_chain(self.audio_db,reader=reader)
        return self.audio_chain.query(question)
    
    
    def __get_audio_chain(self, retreiverdb, reader) -> RetrievalQAChain:
        if(self.audio_chain):
            return self.audio_chain
        
        self.audio_chain = RetrievalQAChain()
        prompt = ModelPrompt()
        self.audio_chain.create_chain(retreiverdb=retreiverdb,reader=reader,prompt=prompt.get_audio_prompt())
        return self.audio_chain
    

    def process_video(self,file_path, video_extractor=SimpleVideoInfoExtractor(1,3),embeddings=OpenAIEmbeddings()):
        '''
        processes video to first get captions using a vision chain by using the video info extractor provided
        then the captions are converted to descriptions using an llm chain and then the documents
        are saved as embeddings in chromadb
        '''   
        docs = video_extractor.video_to_text(self.vid_url,file_path)
        description_docs = self.__caption_to_description(docs) #convert docs to description
        self.__save_video_embeddings(description_docs,embeddings)

    def __caption_to_description(self,docs_vision_caption):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
        splits = text_splitter.split_documents(docs_vision_caption)

        processed_splits = []
        for split in splits:
            result = self.__get_video_description_chain()({"image_captions": split}, return_only_outputs=True)
            processed_splits.append(result['text'])

        return processed_splits


    # save video embeddings
    def __save_video_embeddings(self,docs,embeddings):
        self.video_db = ChromaDB()
        self.video_db.save_embeddings(docs,embeddings,self.save_dir)


    def __get_video_description_chain(self, retreiverdb, reader) -> RetrievalQAChain:
        if(self.vision_description_chain):
            return self.vision_description_chain
        
        self.vision_description_chain = LLMChain()
        prompt = ModelPrompt()
        self.vision_description_chain.create_chain(llm=ChatOpenAI(temperature=0),prompt=prompt.get_text_description_prompt())
        return self.vision_description_chain
    
    #2. Query vision
    def query_vision(self,question:str,reader)->str:
        '''
        queries using vision reader retrieval model and returns the response
        ''' 
        if(self.vision_chain):
            return self.vision_chain.query(question)
        
        self.__get_vision_chain(self.video_db,reader=reader)
        return self.vision_chain.query(question)
    
    def __get_vision_chain(self, retreiverdb, reader) -> RetrievalQAChain:
        if(self.vision_chain):
            return self.vision_chain
        
        self.vision_chain = RetrievalQAChain()
        prompt = ModelPrompt()
        self.vision_chain.create_chain(retreiverdb=retreiverdb,reader=reader,prompt=prompt.get_video_prompt())
        return self.vision_chain



#urls = ["https://www.youtube.com/watch?v=_xASV0YmROc"]
#save_dir = "/Users/hbolak650/Downloads/YouTubeYoga"
#MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
#embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
#tubegpt = TubeGPT()
#tubegpt.process_audio(urls,save_dir,embeddings=embeddings)





        
        

 






vid_url = "https://www.youtube.com/watch?v=5Ay5GqJwHF8&list=PL86SiVwkw_oc8r_X6PL6VqcQ7FTX4923M&index=1"
save_dir = ""