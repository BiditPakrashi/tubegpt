

from video.video_extractor import VideoExtractor
import numpy as np
import torch
#from lavis.models import load_model_and_preprocess
from PIL import Image
#import torchvision.transforms as transforms
import cv2
import pafy
import os


import youtube_dl


class SimpleVideoInfoExtractor(VideoExtractor):
    def __init__(self,keyframe_interval,caption_per_image) -> None:
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.caption_per_image = caption_per_image
        self.keyframe_interval = keyframe_interval

        # self.model, self.vis_processors, _ = load_model_and_preprocess(
        #     name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=self.device
        # )

    def video_to_text(self, url, file_path):
        if not (os.path.exists(file_path)):
            self.__download_youtube_video(url,file_path)
        
        return self.__caption_youtube_blip2_keyframe(file_path,self.keyframe_interval, self.caption_per_image)
    

    def __generate_caption(self,image, caption_type):
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)

        if caption_type == "Beam Search":
            caption = self.model.generate({"image": image})
        else:
            caption = self.model.generate(
                {"image": image}, use_nucleus_sampling=True, num_captions=2
            )

        caption = "\n".join(caption)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return caption
    
    def __extract_keyframes(self,video_path, interval_sec):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video file.")
            return

        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        interval_frames = int(frame_rate * interval_sec)
        keyframes = []

        ret, prev_frame = cap.read()
        frame_count = 1

        while ret:
            ret, curr_frame = cap.read()

            if frame_count % interval_frames == 0:
                keyframes.append(curr_frame)

            prev_frame = curr_frame
            frame_count += 1

        cap.release()

        return keyframes
    
    def __caption_youtube_blip2_keyframe(self,path,interval_sec, caption_per_image):
        keyframes = self.__extract_keyframes(path, interval_sec)
        combined_docs = []
        #prompt = " Notice All Human Activity carefully and Notice All numbers and words in images"
        for frame in keyframes:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            caption = self.__generate_caption(image,f"{caption_per_image} captions")
            if caption not in combined_docs:
                combined_docs.append(caption)
        return combined_docs
    
    def __download_youtube_video(url,file_path):

        ydl_opts = {
            'format': 'best',
            'outtmpl': file_path #'/content/drive/MyDrive/youtube/universal-video.mp4'
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])