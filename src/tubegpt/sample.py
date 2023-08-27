from tempfile import _TemporaryFileWrapper
import gradio as gr
import time

from tubegpt import TubeGPT
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings


__tubegpt_dict = {}
__url_list = []


def query(url,question,is_vision=False,query_options="audio"):
    if not url:
         return " url is empty"
    #MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    #embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    #embeddings = OpenAIEmbeddings()
    persist_directory="./db/audio"
    tubegpt = __tubegpt_dict[url]
    print(query_options)
    if(query_options=="both"):
        if not(is_vision):
             return "set query option to audio as no vision processing done"
        print("getting response using merged retreivers")
        response = tubegpt.query_merged(question=question,reader=ChatOpenAI(temperature=0))
    elif(query_options=="vision"):
        if not(is_vision):
             return "set query option to audio as no vision processing done"
        print("getting response using only vision retreiver")
        response = tubegpt.query_vision(question=question,reader=ChatOpenAI(temperature=0))
    else:
        print("getting response using only audio retreiver")
        response = tubegpt.query_audio(question=question,reader=ChatOpenAI(temperature=0))

    return response

def process(url,is_vision=False,file_desc:_TemporaryFileWrapper=None, save_dir = "/Users/hbolak650/Downloads/movie-clip",progress=gr.Progress()):
    #MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    #embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    progress(0, desc="starting processing")
    embeddings = OpenAIEmbeddings()
    tubegpt = TubeGPT([url],save_dir)
    progress(0.2, desc="processing audio")
    tubegpt.process_audio([url],save_dir,embeddings=embeddings)
    print("audio processing done")
    progress(0.5, desc="audio processing done")
    __tubegpt_dict[url] = tubegpt
    if(is_vision):
         if not file_desc:
              return "upload description file"
         progress(0.51, desc="processing vision data")
         time.sleep(0.1)
         tubegpt.process_vision_from_desc(file_path=file_desc.name,embeddings=embeddings)
         print("vision processing done")
         progress(0.8, desc="vision processing done, merging retreivers")
         time.sleep(0.5)
         tubegpt.merge_retreivers([tubegpt.audio_db.db.as_retriever(search_kwargs={"k": 1}), tubegpt.vision_db.db.as_retriever(search_kwargs={"k": 1})])
         print("retrievers merged")
         progress(0.9, desc="retrieivers merged")
    
    __tubegpt_dict[url] = tubegpt
    __url_list.append(url)
    print(__tubegpt_dict)
    print(__url_list)
    progress(1, desc="all processing done !")
    return ("all processing done",url)

def read_textfile(file):
    with open(file, "r") as file_handle:
        file_text = file_handle.read()
    return file_text


def format_chat_prompt(message, chat_history):
    prompt = ""
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt
def respond(url, message,is_vision,query_options, chat_history):
        formatted_prompt = format_chat_prompt(message, chat_history)
        bot_message = query(url,formatted_prompt,is_vision,query_options)
        # bot_message = client.generate(response,
        #                              max_new_tokens=1024,
        #                              stop_sequences=["\nUser:", "<|endoftext|>"]).generated_text
        chat_history.append((message, bot_message))
        return "", chat_history

if __name__ == '__main__': 
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
            with gr.Tab("Video Processing"): 
                yt_url =  gr.Textbox(
                    label="VideoURL",
                    value="Provide youtube URL here"
                )
                is_vision  = gr.Checkbox(label="vision", info="Process Vision Data?")
                #description_file = gr.File(label="Video Description File")
                file_text = gr.File(inputs='files', outputs='text',label="Upload Video Description file")
                upload = gr.Button("process")
                text=gr.Markdown()

                
            with gr.Tab("Questions"):
                chatbot = gr.Chatbot(height=350) #just to fit the notebook
                msg = gr.Textbox(label="Question for video?")
                url = gr.Textbox(label="provide youtube url")
                query_options = gr.Dropdown(["audio","vision","both"],info="Select Retreiver Model", label= "Retreiver Model")
                btn = gr.Button("Submit")
                #url = gr.Dropdown(__url_list)
                clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")
                btn.click(respond, inputs=[url,msg,is_vision,query_options, chatbot], outputs=[msg, chatbot])
                msg.submit(respond, inputs=[url, msg, is_vision,query_options, chatbot], outputs=[msg, chatbot]) #Press enter to submit

            upload.click(process,inputs=[yt_url,is_vision,file_text], outputs=[text,url], api_name="process")  #command=on_submit_click

#demo.launch(share=True, server_port=7680)
demo.queue(concurrency_count=20).launch(share=True, server_port=7680)