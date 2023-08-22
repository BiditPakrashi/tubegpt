from abc import ABC, abstractmethod


class VideoExtractor(ABC):

    @abstractmethod
    def video_to_text(self,urls,save_dir):
        pass